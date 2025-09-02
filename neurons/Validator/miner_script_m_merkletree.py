#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sybil-compatible miner
----------------------
Modes:
  --mode gpu_info   → print JSON {"num_gpus", "gpu_names"}
  --mode benchmark  → 6-field line  (numGPUs vram size16 t16 size32 t32)
  --mode compute    → build C1=A·B, C2=B·A → Merkle → print ROOTS:/TIMINGS:
  --mode proof      → answer challenged rows + Merkle proofs
"""

import os, gc, time, json, argparse, struct, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

MASK32 = 0xFFFF_FFFF
MIX32  = 0x45D9F3B                # avalanche constant

# ──────────────────────────────────────────────────────────────────────
def xs32(x):
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5)  & MASK32
    return x & MASK32


def gen_matrix(seed: int, n: int, dev: torch.device) -> torch.Tensor:
    seed32 = torch.tensor(seed & MASK32, dtype=torch.int64, device=dev)
    i = torch.arange(n, device=dev).repeat_interleave(n)
    j = torch.arange(n, device=dev).repeat(n)
    s = seed32 + (i & MASK32) + j
    for _ in range(10):
        s = xs32(s)
    return (s.float() / float(MASK32)).reshape(n, n)


def row_hash_gpu(mat: torch.Tensor) -> torch.Tensor:
    """fast 32-bit xor/mix hash for each float32 row – runs on GPU"""
    w = (mat.view(torch.int32) & MASK32).to(torch.int64)   # promote for mul
    while w.shape[1] > 1:
        if w.shape[1] & 1:                                 # pad when odd
            w = torch.cat([w, w[:, -1:]], 1)
        w = w[:, 0::2] ^ w[:, 1::2]
        w = (w * MIX32) & MASK32
        w ^= w >> 16
    return w.squeeze(1).to(torch.int32)


# ───────────────────────── Merkle helpers ─────────────────────────────
def as_uint32_py(x):                        # torch → python int32 unsigned
    return int(np.uint32(x).item())


def build_merkle_cpu(hashes_i32: torch.Tensor):
    """
    Build SHA-256 Merkle tree where each leaf = sha256(little-endian uint32).
    Returns (root_hash_bytes, flat_bytes_of_all_nodes)
    """
    leaves = [
        hashlib.sha256(struct.pack("<I", as_uint32_py(v))).digest()
        for v in hashes_i32.cpu().numpy()
    ]
    tree = [leaves]
    while len(tree[-1]) > 1:
        lvl = tree[-1]
        if len(lvl) & 1:
            lvl = lvl + [lvl[-1]]
        nxt = [
            hashlib.sha256(lvl[i] + lvl[i + 1]).digest()
            for i in range(0, len(lvl), 2)
        ]
        tree.append(nxt)
    root = tree[-1][0]
    flat = b"".join(b for level in tree for b in level)
    return root, flat


def merkle_proof(flat: bytes, idx: int, total: int):
    """Return sibling list for row `idx` from `flat` node blob."""
    proof, off, width = [], 0, total
    while width > 1:
        sib = idx ^ 1
        if sib >= width:                   # padding → duplicate self
            sib = idx
        proof.append(flat[(off + sib) * 32 : (off + sib + 1) * 32])
        idx  //= 2
        off  += width
        width = (width + 1) // 2
    return proof

# ───────────────────────── file helpers ───────────────────────────────
def _read_lines(p):
    return [ln.strip() for ln in open(p) if ln.strip()]


def load_seeds():
    lines = _read_lines("/tmp/seeds.txt")
    n = int(lines[0]); seeds = {}
    for ln in lines[1:]:
        gid, a, b = map(int, ln.split())
        seeds[gid] = (a, b)
    return n, seeds


def load_idx():
    lines = _read_lines("/tmp/challenge_indices.txt")
    out = {}
    for ln in lines:
        gid, rest = ln.split(maxsplit=1)
        out[int(gid)] = [tuple(map(int, p.split(','))) for p in rest.split(';')]
    return out

# ───────────────────── benchmark / VRAM helpers ───────────────────────
def estimate_vram_size(buffer_factor=0.9, precision="fp16"):
    """
    Blind-probe allocation to estimate free VRAM (GB) on GPU 0.
    """
    dtype = torch.float16 if precision == "fp16" else torch.float32
    elem  = 2 if precision == "fp16" else 4
    n = 1024 * 1024
    try:
        while True:
            _ = torch.empty((n,), dtype=dtype, device="cuda")
            n *= 2
    except RuntimeError:
        n //= 2
    vram = n * elem / (buffer_factor * 1e9)
    return vram


def adjust_matrix_size(vram_gb, element_size=2, buffer_factor=0.8):
    usable = vram_gb * buffer_factor * 1e9
    max_sz = int((usable / (2 * element_size)) ** 0.5)
    return (max_sz // 32) * 32


def benchmark_matrix_mul(size, precision="fp16"):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    A = torch.randn(size, size, dtype=dtype, device="cuda")
    B = torch.randn(size, size, dtype=dtype, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    return time.time() - t0


def run_benchmark():
    g = torch.cuda.device_count()
    vram = estimate_vram_size(buffer_factor=1.0, precision="fp16")
    n16  = adjust_matrix_size(vram, element_size=2, buffer_factor=1.0)
    n32  = adjust_matrix_size(vram, element_size=4, buffer_factor=0.5)
    t16  = benchmark_matrix_mul(n16, "fp16")
    t32  = benchmark_matrix_mul(n32, "fp32")
    print(f"{g} {vram:.2f} {n16} {t16:.6f} {n32} {t32:.6f}")

# ─────────────────────────── GPU worker ───────────────────────────────
def gpu_job(gid, sA, sB, n):
    torch.cuda.set_device(gid)
    dev = torch.device(f"cuda:{gid}")
    T = {}                                      # timings

    t = time.time()
    A = gen_matrix(sA, n, dev)
    B = gen_matrix(sB, n, dev)
    torch.cuda.synchronize()
    T["build"] = time.time() - t; t = time.time()

    C1 = torch.matmul(A, B)
    C2 = torch.matmul(B, A)
    torch.cuda.synchronize()
    T["gemm"] = time.time() - t; t = time.time()

    rows = torch.cat((C1, C2))
    h32  = row_hash_gpu(rows)
    torch.cuda.synchronize()
    T["hash"] = time.time() - t; t = time.time()

    root, flat = build_merkle_cpu(h32)
    T["merkle"] = time.time() - t; T["n"] = n

    np.save(f"/dev/shm/flat_{gid}.npy",
            np.frombuffer(flat, dtype=np.uint8))
    np.save(f"/dev/shm/rows_{gid}.npy", rows.float().cpu().numpy())

    del A, B, C1, C2, rows, h32, flat
    torch.cuda.empty_cache(); gc.collect()
    return (gid, root.hex()), (gid, T)

# ────────────────────────── modes ─────────────────────────────────────
def run_compute():
    n, seeds = load_seeds()
    g = torch.cuda.device_count()
    roots, tims = [], []
    with ThreadPoolExecutor(max_workers=g) as ex:
        futs = [
            ex.submit(gpu_job, gid, *seeds[gid], n)
            for gid in range(g)
        ]
        for f in as_completed(futs):
            r, t = f.result(); roots.append(r); tims.append(t)
    print("ROOTS:"   + json.dumps(roots))
    print("TIMINGS:" + json.dumps(tims))


# ---- proof mode (fast: mmap rows / flat only once per GPU) ------------
_mmap_cache = {}                 # gid → (rows, flat_bytes)
def _get_mmaps(gid):
    if gid not in _mmap_cache:
        rows = np.load(f"/dev/shm/rows_{gid}.npy", mmap_mode="r")
        flat = np.load(f"/dev/shm/flat_{gid}.npy",
                       mmap_mode="r").astype(np.uint8).tobytes()
        _mmap_cache[gid] = (rows, flat)
    return _mmap_cache[gid]


def run_proof():
    idxs = load_idx()
    for gid, pairs in idxs.items():
        rows, flat = _get_mmaps(gid)
        total = rows.shape[0]
        resp = {"rows": [], "proofs": []}

        for (i, j) in pairs:
            resp["rows"].append(rows[i])                 # zero-copy slice
            resp["proofs"].append(merkle_proof(flat, i, total))

        np.save(f"/dev/shm/resp_{gid}.npy", resp, allow_pickle=True)

# ────────────────────────── gpu_info mode ─────────────────────────────
def gpu_info():
    if not torch.cuda.is_available():
        print(json.dumps({"num_gpus": 0, "gpu_names": []}))
    else:
        n = torch.cuda.device_count()
        print(
            json.dumps(
                {"num_gpus": n,
                 "gpu_names": [torch.cuda.get_device_name(i) for i in range(n)]}
            )
        )

# ───────────────────────────── CLI ────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", default="benchmark",
                    choices=("gpu_info", "benchmark", "compute", "proof"))
    mode = pa.parse_args().mode

    if mode == "gpu_info":
        gpu_info()
    elif mode == "benchmark":
        run_benchmark()
    elif mode == "compute":
        run_compute()
    elif mode == "proof":
        run_proof()
