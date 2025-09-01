# routes/benchmark.py
# Streaming benchmark runner with in-process allocate/deallocate (no self-HTTP).
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Body
from sse_starlette.sse import EventSourceResponse  # pip install sse-starlette
import yaml
import bittensor as bt

# ───────────────────────── path & imports matching rental_server ─────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow compute/* imports

from utils import RSAEncryption as rsa
from compute.protocol import Allocate
from compute.utils.parser import ComputeArgPaser

from utils.ssh_io import ssh_connect, ssh_close, scp_put
from utils.pog_compat import (
    compute_script_hash,
    send_script_and_request_hash,
    execute_script_on_miner,
    get_remote_gpu_info,
    parse_benchmark_output,
    parse_merkle_output,
    send_seeds,
    send_challenge_indices,
    receive_responses,
    verify_responses as verify_all,
    adjust_matrix_size,
    identify_gpu
)

import state  # reuse caches for W&B specs etc.

# ───────────────────────── Router ─────────────────────────
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# ───────────────────────── Wallets (Mainnet + Testnet) ─────────────────────────
from dotenv import load_dotenv
load_dotenv()

def _make_wallet(name: str, hotkey: str, netuid: int, network: str, suffix: str) -> bt.wallet:
    parser = ComputeArgPaser()
    cfg = parser.config
    cfg.wallet.name = name
    cfg.wallet.hotkey = hotkey
    cfg.netuid = netuid
    cfg.full_path = os.path.expanduser(
        f"~/.bittensor/logs/{name}/{hotkey}/netuid{netuid}/benchmark_{suffix}"
    )
    os.makedirs(cfg.full_path, exist_ok=True)
    w = bt.wallet(config=cfg)
    print(f"[BENCH][BOOT] Wallet ready: name={name} hotkey={hotkey} netuid={netuid} network={network} suffix={suffix}")
    return w

# Mainnet (Finney)
MAIN_WALLET_NAME   = os.getenv("MAIN_WALLET_NAME", "ni_core")
MAIN_WALLET_HOTKEY = os.getenv("MAIN_WALLET_HOTKEY", "ni_core1")
MAIN_NETUID        = int(os.getenv("MAIN_NETUID", 27))
MAIN_NETWORK       = os.getenv("MAIN_NETWORK", "finney")

# Testnet
TEST_WALLET_NAME   = os.getenv("TEST_WALLET_NAME", "ni_core_test")
TEST_WALLET_HOTKEY = os.getenv("TEST_WALLET_HOTKEY", "ni_core_test1")
TEST_NETUID        = int(os.getenv("TEST_NETUID", 27))
TEST_NETWORK       = os.getenv("TEST_NETWORK", "test")

wallet_main = _make_wallet(MAIN_WALLET_NAME, MAIN_WALLET_HOTKEY, MAIN_NETUID, MAIN_NETWORK, "mainnet")
wallet_test = _make_wallet(TEST_WALLET_NAME, TEST_WALLET_HOTKEY, TEST_NETUID, TEST_NETWORK, "testnet")

NETWORKS = {
    "Mainnet": {"network": MAIN_NETWORK, "netuid": MAIN_NETUID, "wallet": wallet_main},
    "Testnet": {"network": TEST_NETWORK, "netuid": TEST_NETUID, "wallet": wallet_test},
}

def _normalize_net_label(v: Optional[str]) -> str:
    s = (v or "").strip().lower()
    if s in ("mainnet", "main", "finney"):
        return "Mainnet"
    # default to Testnet if unknown
    return "Testnet"

# ───────────────────────── Job registry (SSE) ─────────────────────────
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_TTL_SECONDS = 3600

def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "status": "created",  # created|queued|running|done|error|cancelled
        "logs": [],           # list of {level,message,time,payload?}
        "result": None,
        "cancel": asyncio.Event(),
        "started_at": time.time(),
        "hotkey": None,
        "public_key": None,   # for dealloc
        "network": "Testnet", # default
    }
    return job_id

def _append(job_id: str, level: str, message: str, payload: dict | None = None):
    evt = {"level": level, "message": message, "time": time.time()}
    if payload is not None:
        evt["payload"] = payload
    # also print to server logs for debugging
    print(f"[BENCH][{level.upper()}] {message} {('-> ' + json.dumps(payload)) if payload else ''}")
    _JOBS[job_id]["logs"].append(evt)

async def _yield_events(job_id: str) -> AsyncGenerator[dict, None]:
    last = 0
    while True:
        job = _JOBS.get(job_id)
        if not job:
            yield {"event": "error", "data": json.dumps({"message": "job not found"})}
            return
        while last < len(job["logs"]):
            yield {"event": "message", "data": json.dumps(job["logs"][last])}
            last += 1
        if job["status"] in ("done", "error", "cancelled"):
            yield {"event": "end", "data": json.dumps({"status": job["status"], "result": job["result"]})}
            return
        await asyncio.sleep(0.2)

@router.get("/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="job not found")
    return EventSourceResponse(_yield_events(job_id))

@router.delete("/gc")
async def gc():
    now = time.time()
    for jid in list(_JOBS.keys()):
        if now - _JOBS[jid]["started_at"] > _JOB_TTL_SECONDS:
            _JOBS.pop(jid, None)
    return {"ok": True}

# ───────────────────────── Helpers ─────────────────────────
def _load_config_local() -> dict:
    cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
    print(f"[BENCH] Loading config: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def _rand_u32() -> int:
    return random.randint(0, 0xFFFF_FFFF)

# Run blocking function in a thread (keeps event loop responsive)
async def _bg(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

# Yield control so SSE can flush logs immediately
async def _tick():
    await asyncio.sleep(0)

async def _bt_metagraph_sync(network_label: str):
    """Sync metagraph for a given network label ('Mainnet'|'Testnet')."""
    spec = NETWORKS[network_label]
    subtensor = bt.subtensor(network=spec["network"])
    metagraph = subtensor.metagraph(spec["netuid"])
    metagraph.sync()
    return metagraph

# ───────────────────────── In-process Allocate / Deallocate ─────────────────────────
async def _allocate(
    hotkey: Optional[str],
    network_label: str,
    docker_image: str = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
) -> Dict[str, Any]:
    """Allocate a test container on the miner via Dendrite + Allocate protocol."""
    if not hotkey:
        raise RuntimeError("hotkey is required for allocation.")
    spec = NETWORKS[network_label]
    wallet = spec["wallet"]

    print(f"[BENCH][ALLOC] Requesting allocation for hotkey={hotkey} on {network_label}")
    metagraph = await _bt_metagraph_sync(network_label)
    try:
        uid = metagraph.hotkeys.index(hotkey)
    except ValueError:
        raise RuntimeError(f"Hotkey {hotkey} not found in metagraph ({network_label}).")

    axon = metagraph.axons[uid]

    # RSA for secure channel back from miner
    private_key, public_key = rsa.generate_key_pair()
    print("[BENCH][ALLOC] RSA keys generated.")

    # Minimal requirements for testing
    device_requirement = {
        "cpu":       {"count": 1},
        "gpu":       {"count": 1, "capacity": 0, "type": ""},
        "hard_disk": {"capacity": 1_073_741_824},
        "ram":       {"capacity": 1_073_741_824},
        "testing":   True,
    }
    docker_requirement = {"base_image": docker_image}

    MAX_TRIES = 5
    BASE_BACKOFF_S = 1

    for attempt in range(1, MAX_TRIES + 1):
        try:
            print(f"[BENCH][ALLOC] Attempt {attempt} → {axon.ip}:{axon.port} [{network_label}]")
            async with bt.dendrite(wallet=wallet) as dendrite:
                rsp = await dendrite(
                    axon,
                    Allocate(
                        timeline=1,
                        device_requirement=device_requirement,
                        checking=False,
                        public_key=public_key,
                        docker_requirement=docker_requirement,
                    ),
                    timeout=60,
                )

            if rsp and rsp.get("status", False):
                from base64 import b64decode
                decrypted = rsa.decrypt_data(private_key.encode(), b64decode(rsp["info"]))
                info = json.loads(decrypted)

                # read GPU info from our cached stats (best effort)
                s = state.stats.get(uid, {})
                gpu_name = (s.get("gpu_specs") or {}).get("gpu_name") or s.get("gpu_name") or "Unknown GPU"
                gpu_num  = (s.get("gpu_specs") or {}).get("num_gpus") or s.get("gpu_num", 0)

                details = {
                    "ssh": {
                        "host": axon.ip,
                        "port": info["port"],
                        "username": info["username"],
                        "password": info["password"],
                    },
                    "axon": {"ip": axon.ip, "port": axon.port},
                    "gpu_name": gpu_name,
                    "gpu_count": gpu_num,
                }
                print(f"[BENCH][ALLOC] Success on attempt {attempt} for hotkey={hotkey} [{network_label}]")
                return {
                    "status": True,
                    "uid": uid,
                    "hotkey": hotkey,
                    "public_key": public_key,
                    "details": details,
                }

            print(f"[BENCH][ALLOC] Busy/declined on attempt {attempt}.")
            return {"status": False, "detail": "Allocator busy or declined."}

        except bt.dendrite.exceptions.ServerDisconnectedError as e:
            print(f"[BENCH][ALLOC][WARN] Disconnected (attempt {attempt}): {e}")
        except ConnectionRefusedError as e:
            print(f"[BENCH][ALLOC][WARN] Refused (attempt {attempt}): {e}")
        except Exception as e:
            print(f"[BENCH][ALLOC][ERROR] Unexpected: {e}")
            return {"status": False, "detail": f"Exception: {e}"}

        if attempt < MAX_TRIES:
            await asyncio.sleep(BASE_BACKOFF_S * attempt)

    raise RuntimeError("All allocation attempts failed.")

async def _deallocate(hotkey: str, public_key: Optional[str], network_label: str) -> Dict[str, Any]:
    """Deallocate the test container from the miner (timeline=0)."""
    print(f"[BENCH][DEALLOC] Request for hotkey={hotkey} [{network_label}]")
    if not public_key:
        print("[BENCH][DEALLOC][WARN] public_key missing; attempting best-effort dealloc (may fail).")

    spec = NETWORKS[network_label]
    wallet = spec["wallet"]

    metagraph = await _bt_metagraph_sync(network_label)
    try:
        uid = metagraph.hotkeys.index(hotkey)
    except ValueError:
        print(f"[BENCH][DEALLOC][WARN] hotkey {hotkey} not in metagraph anymore.")
        return {"status": True}

    axon = metagraph.axons[uid]
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            async with bt.dendrite(wallet=wallet) as dendrite:
                rsp = await dendrite(
                    axon,
                    Allocate(
                        timeline=0,
                        checking=False,
                        public_key=public_key,
                    ),
                    timeout=30,
                )
                if rsp and rsp.get("status", True):
                    print(f"[BENCH][DEALLOC] Deallocated {hotkey} [{network_label}]")
                    return {"status": True}
                retry_count += 1
                print(f"[BENCH][DEALLOC] Failed (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(5)
        except Exception as e:
            retry_count += 1
            print(f"[BENCH][DEALLOC] Exception (attempt {retry_count}/{max_retries}): {e}")
            await asyncio.sleep(5)

    print(f"[BENCH][DEALLOC] Giving up on {hotkey}; treating as success to avoid leaks.")
    return {"status": True}

# ───────────────────────── API ─────────────────────────
@router.post("/start")
async def start_benchmark(
    payload: dict = Body(...),
    x_admin_key: str = Header(None),
):
    if not x_admin_key or x_admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Forbidden")

    hotkey = payload.get("hotkey")
    if not hotkey:
        raise HTTPException(status_code=400, detail="hotkey is required")

    network_label = _normalize_net_label(payload.get("network"))
    if network_label not in NETWORKS:
        raise HTTPException(status_code=400, detail="invalid network; use 'Mainnet' or 'Testnet'")

    job_id = _new_job()
    _JOBS[job_id]["status"] = "queued"
    _JOBS[job_id]["network"] = network_label
    print(f"[BENCH] start_benchmark called. job_id={job_id}, hotkey={hotkey}, network={network_label}")
    asyncio.create_task(_run_benchmark_job(job_id, hotkey, network_label))
    return {"job_id": job_id}

@router.post("/cancel/{job_id}")
async def cancel(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    print(f"[BENCH] cancel requested for job_id={job_id}")
    job["cancel"].set()
    return {"status": "cancelling"}

# ───────────────────────── Core runner ─────────────────────────
async def _run_benchmark_job(job_id: str, hotkey: Optional[str], network_label: str):
    job = _JOBS[job_id]
    cancel_event = job["cancel"]
    job["status"] = "running"

    ssh = None
    alloc: Dict[str, Any] | None = None
    remote_script_path = "/tmp/miner_script.py"

    def _check_cancel():
        if cancel_event.is_set():
            raise asyncio.CancelledError()

    try:
        # 0) Load config
        _append(job_id, "info", f"Loading config... (network={network_label})")
        await _tick()
        cfg = await _bg(_load_config_local)
        gpu_data = cfg.get("gpu_performance", {}) or {}
        gpu_tolerance_pairs = gpu_data.get("gpu_tolerance_pairs", {}) or {}
        merkle_cfg = cfg.get("merkle_proof", {}) or {}
        miner_script_path = merkle_cfg.get("miner_script_path", "")
        time_tol = float(merkle_cfg.get("time_tolerance", 5.0))
        if not miner_script_path or not os.path.exists(miner_script_path):
            raise RuntimeError("merkle_proof.miner_script_path missing or not found on server.")
        _append(job_id, "success", "Config loaded.", {"miner_script_path": miner_script_path, "time_tolerance": time_tol, "network": network_label})
        await _tick()
        _check_cancel()

        # 1) Allocate (in-process)
        _append(job_id, "info", "Allocating miner...")
        await _tick()
        alloc = await _allocate(hotkey=hotkey, network_label=network_label)
        if not alloc or not alloc.get("status"):
            raise RuntimeError(f"Allocation failed on {network_label}: {alloc}")
        details = alloc["details"]
        job["hotkey"] = alloc.get("hotkey")
        job["public_key"] = alloc.get("public_key")
        _append(job_id, "success", "Allocation successful.", {"hotkey": job["hotkey"]})
        await _tick()
        _check_cancel()

        # 2) SSH
        _append(job_id, "info", "Connecting over SSH...")
        await _tick()
        ssh = await _bg(
            ssh_connect,
            host=details["ssh"]["host"],
            port=details["ssh"]["port"],
            username=details["ssh"]["username"],
            password=details["ssh"]["password"],
            timeout=25,
        )
        _append(job_id, "success", "SSH connected.")
        await _tick()
        _check_cancel()

        # 3) Upload + integrity check
        _append(job_id, "info", "Uploading miner script and verifying integrity...")
        await _tick()
        await _bg(scp_put, ssh, miner_script_path, remote_script_path)
        local_hash  = await _bg(compute_script_hash, miner_script_path)
        remote_hash = await _bg(send_script_and_request_hash, ssh, miner_script_path)
        if local_hash != remote_hash:
            raise RuntimeError("Script integrity check failed (hash mismatch).")
        _append(job_id, "success", "Integrity check passed.", {"hash": local_hash})
        await _tick()
        _check_cancel()

        # 4) GPU info (driver-reported)
        _append(job_id, "info", "Querying GPU info...")
        await _tick()
        ginfo = await _bg(get_remote_gpu_info, ssh)  # {"num_gpus": N, "gpu_names": [...]}
        if int(ginfo.get("num_gpus", 0)) <= 0:
            raise RuntimeError("No GPUs detected.")
        reported_name = (ginfo.get("gpu_names") or [None])[0]
        _append(job_id, "success", "GPU identification (driver) received.", {
            "reported_name": reported_name,
            "num_gpus": ginfo.get("num_gpus")
        })
        await _tick()
        _check_cancel()

        # 5) Benchmark → identify GPU from performance (PoG-determined)
        _append(job_id, "info", "Running benchmark (FP16/FP32)...")
        await _tick()
        bench_out = await _bg(execute_script_on_miner, ssh, "benchmark")
        num_gpus, vram, size_fp16, time_fp16, size_fp32, time_fp32 = parse_benchmark_output(bench_out)

        # Compute achieved TFLOPs
        fp16_tflops = (2 * (size_fp16 ** 3)) / time_fp16 / 1e12
        fp32_tflops = (2 * (size_fp32 ** 3)) / time_fp32 / 1e12

        # Identify GPU by performance + avRAM using your table (with tolerance pairs)
        identified_name = identify_gpu(
            fp16_tflops=fp16_tflops,
            fp32_tflops=fp32_tflops,
            estimated_avram=vram,
            gpu_data=gpu_data,
            reported_name=reported_name,
            tolerance_pairs=gpu_tolerance_pairs
        )

        bench = dict(
            num_gpus=num_gpus, vram=vram,
            size_fp16=size_fp16, time_fp16=time_fp16,
            size_fp32=size_fp32, time_fp32=time_fp32,
            fp16_tflops=fp16_tflops, fp32_tflops=fp32_tflops,
            reported_gpu=reported_name, identified_gpu=identified_name
        )
        _append(job_id, "success", "Benchmark completed.", bench)
        await _tick()
        _check_cancel()

        # Enforce identity match
        if identified_name and reported_name and identified_name != reported_name:
            raise RuntimeError(
                f"GPU mismatch: reported='{reported_name}' vs identified='{identified_name}' "
                f"(fp16≈{fp16_tflops:.2f} TFLOPs, fp32≈{fp32_tflops:.2f} TFLOPs, avram≈{vram:.2f} GB)"
            )

        _append(job_id, "success", "GPU footprint matches reported GPU.", {
            "gpu": identified_name or reported_name
        })
        await _tick()

        # 6) Seeds
        buffer_factor = 0.4
        n = max(256, int(((vram * buffer_factor * 1e9) / (3 * 4)) ** 0.5 // 32 * 32))
        seeds = {gid: (_rand_u32(), _rand_u32()) for gid in range(num_gpus)}
        _append(job_id, "info", f"Preparing seeds (n={n})...")
        await _tick()
        await _bg(send_seeds, ssh, seeds, n)
        _check_cancel()

        # 7) PoG compute
        _append(job_id, "info", "Running PoG compute...")
        await _tick()
        compute_out = await _bg(execute_script_on_miner, ssh, "compute")
        root_hashes_list, gpu_timings_list = parse_merkle_output(compute_out)
        root_map: Dict[int, str] = {int(g): h for (g, h) in root_hashes_list}
        timing_map: Dict[int, Dict[str, float]] = {int(g): t for (g, t) in gpu_timings_list}
        _append(job_id, "success", "PoG compute completed.", {"n": n, "roots": root_map})
        await _tick()
        _check_cancel()

        # 8) Indices (1 per GPU for speed; increase for stronger checks)
        idx_map: Dict[int, List[tuple[int, int]]] = {}
        for gid in range(num_gpus):
            i = random.randint(0, 2 * n - 1)
            j = random.randint(0, n - 1)
            idx_map[gid] = [(i, j)]
        _append(job_id, "info", "Uploading challenge indices...")
        await _tick()
        await _bg(send_challenge_indices, ssh, idx_map)
        _check_cancel()

        # 9) Proof
        _append(job_id, "info", "Running PoG proof...")
        await _tick()
        _ = await _bg(execute_script_on_miner, ssh, "proof")
        _append(job_id, "success", "Proof data generated on miner.")
        await _tick()

        # 10) Verify
        _append(job_id, "info", "Downloading proof responses and verifying...")
        await _tick()
        responses = await _bg(receive_responses, ssh, num_gpus)
        verified = verify_all(seeds, root_map, responses, idx_map, n)

        gemm_times = [t.get("gemm", 0.0) for t in timing_map.values() if isinstance(t, dict)]
        avg_gemm = sum(gemm_times) / len(gemm_times) if gemm_times else 0.0
        timing_ok = (avg_gemm <= (time_fp32 + time_tol))

        if not verified or not timing_ok:
            raise RuntimeError(f"Verification failed (verified={verified}, timing_ok={timing_ok}, avg_gemm={avg_gemm:.4f}).")
        _append(job_id, "success", "PoG verified.", {"avg_gemm": avg_gemm, "timing_ok": timing_ok})
        await _tick()

                # 10.1) W&B active run + (Mainnet-only) specs
        _append(job_id, "info", "Checking W&B hardware specs / active run…")
        await _tick()
        try:
            # 1) Active W&B run (both networks)
            runs_map = (state.active_wandb_runs or {}).get(network_label, {}) or {}
            run_path = runs_map.get(job["hotkey"])

            if not run_path:
                # No active run for this hotkey on the selected network
                _append(
                    job_id, "error",
                    "No active W&B run found for this hotkey.",
                    {"network": network_label, "hotkey": job["hotkey"], "wandb_active": False}
                )
            else:
                payload = {"network": network_label, "hotkey": job["hotkey"], "wandb_active": True, "run_path": run_path}

                # 2) (Mainnet only) attach cached specs if present
                if network_label == "Mainnet":
                    specs_cache = state.hardware_specs_cache or {}

                    # Prefer UID returned by allocation, else find by hotkey
                    uid = (alloc or {}).get("uid")
                    if uid is None:
                        for u, entry in specs_cache.items():
                            if isinstance(entry, dict) and entry.get("hotkey") == job["hotkey"]:
                                uid = u
                                break

                    entry   = specs_cache.get(uid) if uid is not None else None
                    details = (entry or {}).get("details") or {}

                    if details:
                        def _gb(val, denom=1024**3):
                            try:
                                return round(float(val) / float(denom), 2)
                            except Exception:
                                return None

                        gpu  = details.get("gpu", {}) or {}
                        cpu  = details.get("cpu", {}) or {}
                        ram  = details.get("ram", {}) or {}
                        disk = details.get("hard_disk", {}) or {}

                        payload.update({
                            "uid": uid,
                            "gpu": {
                                "name": gpu.get("name"),
                                "count": gpu.get("count"),
                                "capacity_GiB": (
                                    round(float(gpu.get("capacity", 0)) / 1024.0, 2)
                                    if isinstance(gpu.get("capacity", 0), (int, float)) else None
                                ),
                            },
                            "cpu": {"count": cpu.get("count")},
                            "ram_GiB": _gb(ram.get("available")),
                            "disk_free_GiB": _gb(disk.get("free")),
                        })

                # Success log (active run; plus specs if mainnet had them)
                _append(job_id, "success", "W&B active run detected.", payload)

        except Exception as e:
            _append(job_id, "error", f"W&B check skipped: {e}")
        await _tick()

        # 11) Finalize
        result = {
            "hotkey": job["hotkey"],
            "network": network_label,
            "gpu_names": ginfo["gpu_names"],
            "gpu_count": num_gpus,
            "n": n,
            "fp16_tflops": fp16_tflops,
            "fp32_tflops": fp32_tflops,
            "bench_times": {"fp16_s": time_fp16, "fp32_s": time_fp32},
            "avg_gemm_s": avg_gemm,
        }
        _append(job_id, "info", "Finalizing results...")
        await _tick()
        _JOBS[job_id]["result"] = result
        _JOBS[job_id]["status"] = "done"
        _append(job_id, "success", "Benchmark complete.", result)
        await _tick()

    except asyncio.CancelledError:
        _JOBS[job_id]["status"] = "cancelled"
        _append(job_id, "warning", "Benchmark cancelled by user.")
        await _tick()
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _append(job_id, "error", f"{type(e).__name__}: {e}")
        await _tick()
    finally:
        # Clean up
        try:
            if ssh:
                await _bg(ssh_close, ssh)
        except Exception:
            pass
        try:
            hk = _JOBS[job_id].get("hotkey")
            pk = _JOBS[job_id].get("public_key")
            net = _JOBS[job_id].get("network", "Testnet")
            if hk:
                _append(job_id, "info", "Deallocating miner...")
                await _tick()
                await _deallocate(hotkey=hk, public_key=pk, network_label=net)
                _append(job_id, "success", "Deallocated.")
                await _tick()
        except Exception as e:
            _append(job_id, "error", f"Deallocation error: {e}")
            await _tick()