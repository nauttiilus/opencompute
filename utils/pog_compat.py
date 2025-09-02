# utils/pog_compat.py
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np

MASK32 = 0xFFFF_FFFF
MIX32  = 0x45D9F3B

def xs32(x):
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5)  & MASK32
    return x & MASK32

def prng(seed, i, j):
    s = (seed + (i & MASK32) + j) & MASK32
    for _ in range(10):
        s = xs32(s)
    return s / float(MASK32)

def row_hash32_np(row: np.ndarray) -> int:
    words = np.ascontiguousarray(row, dtype=np.float32).view(np.uint32)
    while words.size > 1:
        if words.size & 1:
            words = np.append(words, words[-1])
        words = words[0::2] ^ words[1::2]
        words = (words.astype(np.uint64) * MIX32) & MASK32
        words ^= words >> np.uint64(16)
        words = words.astype(np.uint32)
    return int(words[0])

def leaf_digest(row: np.ndarray) -> bytes:
    return hashlib.sha256(struct.pack("<I", row_hash32_np(row))).digest()

def verify_merkle_proof_row(row, proof, root_hash, index, total_leaves, hash_func=hashlib.sha256):
    computed_hash = leaf_digest(row)
    idx = index
    for sibling_hash in proof:
        if idx % 2 == 0:
            computed_hash = hash_func(computed_hash + sibling_hash).digest()
        else:
            computed_hash = hash_func(sibling_hash + computed_hash).digest()
        idx //= 2
    return computed_hash == root_hash

# ---- miner script interactions

def compute_script_hash(script_path):
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def send_script_and_request_hash(ssh_client, script_path):
    sftp = ssh_client.open_sftp()
    try:
        sftp.put(script_path, "/tmp/miner_script.py")
    finally:
        sftp.close()
    hash_command = r"""/opt/conda/bin/python -c "import hashlib;print(hashlib.sha256(open('/tmp/miner_script.py','rb').read()).hexdigest())" """
    stdin, stdout, stderr = ssh_client.exec_command(hash_command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        raise RuntimeError(f"Hash computation failed: {err}")
    return out.splitlines()[-1]

def execute_script_on_miner(ssh_client, mode):
    execution_command = f"/opt/conda/bin/python /tmp/miner_script.py --mode {mode}"
    stdin, stdout, stderr = ssh_client.exec_command(execution_command)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        raise RuntimeError(f"Script execution failed: {err}")
    return out

def parse_benchmark_output(output):
    parts = output.strip().split()
    if len(parts) < 6:
        raise ValueError(f"Unexpected benchmark output: {output!r}")
    num_gpus = int(parts[0])
    vram = float(parts[1])
    size_fp16 = int(parts[2])
    time_fp16 = float(parts[3])
    size_fp32 = int(parts[4])
    time_fp32 = float(parts[5])
    return num_gpus, vram, size_fp16, time_fp16, size_fp32, time_fp32

def parse_merkle_output(output):
    lines = output.strip().split("\n")
    root_hashes_line = None
    timings_line = None
    for ln in lines:
        if ln.startswith("ROOTS:"):
            root_hashes_line = ln
        elif ln.startswith("TIMINGS:"):
            timings_line = ln
    if root_hashes_line is None or timings_line is None:
        raise ValueError("Output does not contain ROOTS or TIMINGS")
    root_hashes = json.loads(root_hashes_line.split(":", 1)[1])
    gpu_timings = json.loads(timings_line.split(":", 1)[1])
    return root_hashes, gpu_timings

def get_random_seeds(num_gpus):
    import secrets
    seeds = {}
    for gid in range(num_gpus):
        seeds[gid] = (secrets.randbits(32), secrets.randbits(32))
    return seeds

def send_seeds(ssh_client, seeds, n):
    lines = [str(n)]
    for gpu_id, (sA, sB) in seeds.items():
        lines.append(f"{gpu_id} {sA} {sB}")
    data = "\n".join(lines)
    with ssh_client.open_sftp() as sftp:
        with sftp.file("/tmp/seeds.txt", "w") as f:
            f.write(data)

def send_challenge_indices(ssh_client, indices):
    lines = []
    for gpu_id, pairs in indices.items():
        idxs = ";".join([f"{i},{j}" for (i, j) in pairs])
        lines.append(f"{gpu_id} {idxs}")
    data = "\n".join(lines)
    with ssh_client.open_sftp() as sftp:
        with sftp.file("/tmp/challenge_indices.txt", "w") as f:
            f.write(data)

def receive_responses(ssh_client, num_gpus):
    responses = {}
    with ssh_client.open_sftp() as sftp, tempfile.TemporaryDirectory() as td:
        for gid in range(num_gpus):
            remote_path = f"/dev/shm/resp_{gid}.npy"
            local_path = os.path.join(td, f"resp_{gid}.npy")
            try:
                sftp.get(remote_path, local_path)
                obj = np.load(local_path, allow_pickle=True)
                responses[gid] = obj.item()
            except Exception:
                responses[gid] = None
    return responses

def get_remote_gpu_info(ssh_client):
    cmd = "/opt/conda/bin/python /tmp/miner_script.py --mode gpu_info"
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        raise RuntimeError(f"Failed to get GPU info: {err}")
    return json.loads(out)

def verify_responses(seeds, root_hashes, responses, indices, n):
    verification_passed = True
    failed_gpus = []
    num_gpus = len(root_hashes.keys())
    required_passes = num_gpus if num_gpus <= 4 else int(np.ceil(0.75 * num_gpus))

    for gpu_id in root_hashes.keys():
        s_A, s_B = seeds[gpu_id]
        gpu_indices = indices[gpu_id]
        response = responses[gpu_id]
        if response is None:
            failed_gpus.append(gpu_id)
            continue
        root_hash = bytes.fromhex(root_hashes[gpu_id])
        total_leaves = 2 * n
        gpu_failed = False

        for idx, (i, j) in enumerate(gpu_indices):
            if i < n:
                exp = sum(prng(s_A, i, k) * prng(s_B, k, j) for k in range(n))
            else:
                ir = i - n
                exp = sum(prng(s_B, ir, k) * prng(s_A, k, j) for k in range(n))

            row_miner = response["rows"][idx]
            proof = response["proofs"][idx]
            value_miner = row_miner[j]

            if not np.isclose(value_miner, exp, atol=1e-4, rtol=1e-3):
                gpu_failed = True
                break
            if not verify_merkle_proof_row(row_miner, proof, root_hash, i, total_leaves):
                gpu_failed = True
                break

        if gpu_failed:
            failed_gpus.append(gpu_id)

    passed_gpus = num_gpus - len(failed_gpus)
    return passed_gpus >= required_passes

def adjust_matrix_size(vram, element_size=2, buffer_factor=0.8):
    """
    Calculate the matrix size based on available VRAM.
    """
    usable_vram = vram * buffer_factor * 1e9
    max_size = int((usable_vram / (2 * element_size)) ** 0.5)
    aligned_size = (max_size // 32) * 32
    return aligned_size

def identify_gpu(fp16_tflops, fp32_tflops, estimated_avram, gpu_data, reported_name=None, tolerance_pairs=None):
    """
    Identify GPU based on TFLOPS and AVRAM with a tolerance check for GPUs with similar fingerprints.
    """
    tolerance_pairs = tolerance_pairs or {}
    GPU_TFLOPS_FP16 = gpu_data["GPU_TFLOPS_FP16"]
    GPU_TFLOPS_FP32 = gpu_data["GPU_TFLOPS_FP32"]
    GPU_AVRAM = gpu_data["GPU_AVRAM"]

    combined_scores = []
    for gpu in GPU_TFLOPS_FP16.keys():
        fp16_theoretical = GPU_TFLOPS_FP16[gpu]
        fp32_theoretical = GPU_TFLOPS_FP32[gpu]
        avram_theoretical = GPU_AVRAM[gpu]
        fp16_deviation = abs(fp16_tflops - fp16_theoretical) / fp16_theoretical
        fp32_deviation = abs(fp32_tflops - fp32_theoretical) / fp32_theoretical
        avram_deviation = abs(estimated_avram - avram_theoretical) / avram_theoretical
        combined_score = (fp16_deviation + fp32_deviation + avram_deviation) / 3
        combined_scores.append((gpu, combined_score))
    identified_gpu = sorted(combined_scores, key=lambda x: x[1])[0][0]
    if reported_name:
        if identified_gpu in tolerance_pairs and reported_name == tolerance_pairs.get(identified_gpu):
            identified_gpu = reported_name
        elif reported_name in tolerance_pairs and identified_gpu == tolerance_pairs.get(reported_name):
            identified_gpu = reported_name
    return identified_gpu