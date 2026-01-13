# routes/hardware_check.py
# Simplified hardware verification tool - replaces old benchmark.py
# Checks: Port accessibility, GPU performance, W&B active run
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException, Body, Request
from starlette.responses import JSONResponse
from starlette import status
from sse_starlette.sse import EventSourceResponse
import yaml
import bittensor as bt

# ───────────────────────── path & imports ─────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import RSAEncryption as rsa
from compute.protocol import Allocate
from compute.utils.parser import ComputeArgPaser
from utils.ssh_io import ssh_connect, ssh_connect_key, ssh_close, generate_ssh_keypair

import state

# ───────────────────────── Router ─────────────────────────
router = APIRouter(prefix="/hardware-check", tags=["hardware-check"])

# ───────────────────────── Wallets ─────────────────────────
from dotenv import load_dotenv
load_dotenv()

def _make_wallet(name: str, hotkey: str, netuid: int, network: str, suffix: str) -> bt.wallet:
    parser = ComputeArgPaser()
    cfg = parser.config
    cfg.wallet.name = name
    cfg.wallet.hotkey = hotkey
    cfg.netuid = netuid
    cfg.full_path = os.path.expanduser(
        f"~/.bittensor/logs/{name}/{hotkey}/netuid{netuid}/hwcheck_{suffix}"
    )
    os.makedirs(cfg.full_path, exist_ok=True)
    w = bt.wallet(config=cfg)
    print(f"[HWCHECK][BOOT] Wallet ready: name={name} hotkey={hotkey} netuid={netuid}")
    return w

# Mainnet
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

# ───────────────────────── Port Configuration ─────────────────────────
DEFAULT_PORTS = {
    "ssh": 4444,
    "test_ssh": 4445,
    "axon": 8091,
    "external": [27015, 27016, 27017, 27018],
}

# ───────────────────────── TCP Port Check ─────────────────────────
def tcp_port_check(host: str, port: int, timeout: float = 3.0) -> Tuple[bool, str]:
    """Check if a TCP port is open and accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return True, "open"
        else:
            return False, f"closed (errno={result})"
    except socket.timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)

def check_axon_port(host: str, ports_config: dict = None) -> Dict[str, Any]:
    """Check only the axon port (always available on miners)."""
    if ports_config is None:
        ports_config = DEFAULT_PORTS

    axon_port = ports_config["axon"]
    ok, msg = tcp_port_check(host, axon_port)

    return {
        "ports": {f"axon ({axon_port})": {"open": ok, "message": msg}},
        "all_open": ok
    }

def check_container_ports(host: str, ssh_port: int, ports_config: dict = None) -> Dict[str, Any]:
    """Check SSH port (only available after container allocation).

    Note: External ports (27015-27018) are not checked because they're just forwarded
    ports with no service bound to them until the user starts their own services.
    """
    if ports_config is None:
        ports_config = DEFAULT_PORTS

    results = {}
    all_open = True

    # Check allocated SSH port (dynamic from allocation response)
    # This is the only port with a guaranteed service (sshd) running
    ok, msg = tcp_port_check(host, ssh_port)
    results[f"ssh ({ssh_port})"] = {"open": ok, "message": msg}
    if not ok:
        all_open = False

    return {"ports": results, "all_open": all_open}

# ───────────────────────── GPU Detection via nvidia-smi ─────────────────────────
def get_gpu_info_nvidia_smi(ssh_client) -> dict:
    """Get GPU info using nvidia-smi (no script upload needed)."""
    # Query nvidia-smi for GPU name, count, and memory
    cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
    stdin, stdout, stderr = ssh_client.exec_command(cmd, timeout=30)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()

    if err and "NVIDIA-SMI" not in err:
        # Try alternative command
        cmd_alt = "nvidia-smi -L"
        stdin, stdout, stderr = ssh_client.exec_command(cmd_alt, timeout=30)
        out_alt = stdout.read().decode().strip()
        if out_alt:
            # Parse "GPU 0: NVIDIA GeForce RTX 4090 (UUID: ...)"
            lines = [l for l in out_alt.split("\n") if l.strip().startswith("GPU")]
            num_gpus = len(lines)
            if num_gpus > 0:
                # Extract GPU name from first line
                first = lines[0]
                # "GPU 0: NVIDIA GeForce RTX 4090 (UUID: ...)"
                name_part = first.split(":", 1)[1] if ":" in first else first
                name_part = name_part.split("(UUID")[0].strip() if "(UUID" in name_part else name_part.strip()
                return {
                    "num_gpus": num_gpus,
                    "gpu_names": [name_part] * num_gpus,
                    "vram_mb": None,  # Can't get VRAM from -L output
                }
        raise RuntimeError(f"nvidia-smi failed: {err}")

    if not out:
        raise RuntimeError("No GPUs detected (nvidia-smi returned empty)")

    # Parse CSV output: "NVIDIA GeForce RTX 4090, 24564"
    lines = [l.strip() for l in out.split("\n") if l.strip()]
    num_gpus = len(lines)
    gpu_names = []
    vram_total = 0

    for line in lines:
        parts = line.split(",")
        name = parts[0].strip() if parts else "Unknown"
        vram = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0
        gpu_names.append(name)
        vram_total += vram

    return {
        "num_gpus": num_gpus,
        "gpu_names": gpu_names,
        "vram_mb": vram_total,
        "vram_gb": round(vram_total / 1024, 2) if vram_total else None,
    }

# ───────────────────────── Job Registry ─────────────────────────
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_TTL_SECONDS = 3600

def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "status": "created",
        "logs": [],
        "result": None,
        "cancel": asyncio.Event(),
        "started_at": time.time(),
        "hotkey": None,
        "reserved_hotkey": None,
        "public_key": None,
        "ssh_private_key": None,  # SSH private key for container access
        "network": "Testnet",
    }
    return job_id

def _append(job_id: str, level: str, message: str, payload: dict | None = None):
    evt = {"level": level, "message": message, "time": time.time()}
    if payload is not None:
        evt["payload"] = payload
    print(f"[HWCHECK][{level.upper()}] {message} {('-> ' + json.dumps(payload)) if payload else ''}")
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

async def _tick():
    await asyncio.sleep(0)

# ───────────────────────── Rate Limiting ─────────────────────────
from collections import deque, defaultdict
from time import monotonic

HWCHECK_WORKERS   = int(os.getenv("HWCHECK_WORKERS", "3"))
HWCHECK_QUEUE_MAX = int(os.getenv("HWCHECK_QUEUE_MAX", "50"))
HWCHECK_PER_KEY   = os.getenv("HWCHECK_PER_KEY", "3/min")
HWCHECK_PER_IP    = os.getenv("HWCHECK_PER_IP", "10/min")
HWCHECK_GLOBAL    = os.getenv("HWCHECK_GLOBAL", "30/min")

def _parse_limit(s: str) -> Tuple[int, float]:
    n, per = s.strip().split("/", 1)
    n = int(n)
    per = per.strip().lower()
    if per in ("s", "sec", "second", "seconds"):
        window = 1.0
    elif per in ("m", "min", "minute", "minutes"):
        window = 60.0
    elif per in ("h", "hour", "hours"):
        window = 3600.0
    else:
        window = 60.0
    return n, window

_LIM_PER_KEY = _parse_limit(HWCHECK_PER_KEY)
_LIM_PER_IP  = _parse_limit(HWCHECK_PER_IP)
_LIM_GLOBAL  = _parse_limit(HWCHECK_GLOBAL)

_calls_per_key: Dict[str, deque[float]] = defaultdict(deque)
_calls_per_ip:  Dict[str, deque[float]] = defaultdict(deque)
_calls_global:  deque[float] = deque()

def _rate_ok(dq: deque[float], limit: Tuple[int, float]) -> bool:
    now = monotonic()
    n, window = limit
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= n:
        return False
    dq.append(now)
    return True

def _retry_after(dq: deque[float], limit: Tuple[int, float]) -> int:
    if not dq:
        return 1
    now = monotonic()
    _, window = limit
    oldest = dq[0]
    remaining = max(0.0, window - (now - oldest))
    return max(1, int(remaining))

# Queue and workers
_job_queue: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue(maxsize=HWCHECK_QUEUE_MAX)
_workers_started = False
_running_workers = 0
_busy_hotkeys: Dict[str, str] = {}

async def _start_workers():
    global _workers_started
    if _workers_started:
        return
    for _ in range(HWCHECK_WORKERS):
        asyncio.create_task(_queue_worker())
    _workers_started = True
    print(f"[HWCHECK][POOL] started {HWCHECK_WORKERS} workers; queue max={HWCHECK_QUEUE_MAX}")

async def _queue_worker():
    global _running_workers
    while True:
        job_id, hotkey, network_label = await _job_queue.get()
        _running_workers += 1
        try:
            await _run_hardware_check(job_id, hotkey, network_label)
        finally:
            _running_workers -= 1
            _job_queue.task_done()

def _client_ip(request: Request) -> str:
    xfwd = request.headers.get("x-forwarded-for")
    if xfwd:
        return xfwd.split(",")[0].strip()
    return (request.client.host if request.client else "unknown")

def _normalize_net_label(v: Optional[str]) -> str:
    s = (v or "").strip().lower()
    if s in ("mainnet", "main", "finney"):
        return "Mainnet"
    return "Testnet"

def _load_config_local() -> dict:
    cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

# ───────────────────────── Allocation Helpers ─────────────────────────
async def _bt_metagraph_sync(network_label: str):
    spec = NETWORKS[network_label]
    subtensor = bt.subtensor(network=spec["network"])
    metagraph = subtensor.metagraph(spec["netuid"])
    metagraph.sync()
    return metagraph

async def _allocate(hotkey: str, network_label: str,
                    docker_image: str = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime") -> Dict[str, Any]:
    """Allocate a test container on the miner."""
    spec = NETWORKS[network_label]
    wallet = spec["wallet"]

    print(f"[HWCHECK][ALLOC] Requesting allocation for hotkey={hotkey} on {network_label}")
    metagraph = await _bt_metagraph_sync(network_label)
    try:
        uid = metagraph.hotkeys.index(hotkey)
    except ValueError:
        raise RuntimeError(f"Hotkey {hotkey} not found in metagraph ({network_label}).")

    axon = metagraph.axons[uid]
    private_key, public_key = rsa.generate_key_pair()

    # Generate SSH keypair for container access (key-based auth, not password)
    ssh_private_key, ssh_public_key = generate_ssh_keypair()
    print(f"[HWCHECK][ALLOC] Generated SSH keypair for container access")

    device_requirement = {
        "cpu": {"count": 1},
        "gpu": {"count": 1, "capacity": 0, "type": ""},
        "hard_disk": {"capacity": 1_073_741_824},
        "ram": {"capacity": 1_073_741_824},
        "testing": True,
    }
    docker_requirement = {
        "base_image": docker_image,
        "ssh_key": ssh_public_key,  # Pass SSH public key to miner
    }

    MAX_TRIES = 3
    RETRY_DELAY = 1.5  # 1-2 seconds
    last_error = None

    for attempt in range(1, MAX_TRIES + 1):
        try:
            print(f"[HWCHECK][ALLOC] Attempt {attempt}/{MAX_TRIES} -> {axon.ip}:{axon.port}")
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

            print(f"[HWCHECK][ALLOC] Response type: {type(rsp)}")

            # Handle different response formats
            if rsp is None:
                print(f"[HWCHECK][ALLOC] Response is None")
                last_error = "No response from miner"
                if attempt < MAX_TRIES:
                    await asyncio.sleep(RETRY_DELAY)
                continue

            # If rsp is a Synapse object, extract the output dict
            if hasattr(rsp, 'output') and rsp.output:
                rsp_data = rsp.output
                print(f"[HWCHECK][ALLOC] Extracted from synapse.output")
            elif hasattr(rsp, '__dict__'):
                # Try to get status from synapse attributes
                rsp_data = {}
                if hasattr(rsp, 'status'):
                    rsp_data['status'] = rsp.status
                if hasattr(rsp, 'info'):
                    rsp_data['info'] = rsp.info
                print(f"[HWCHECK][ALLOC] Extracted from synapse attrs: status={rsp_data.get('status')}")
            elif isinstance(rsp, dict):
                rsp_data = rsp
            else:
                rsp_data = rsp

            status_val = rsp_data.get("status", False) if isinstance(rsp_data, dict) else getattr(rsp_data, 'status', False)

            if status_val:
                from base64 import b64decode
                info_val = rsp_data.get("info") if isinstance(rsp_data, dict) else getattr(rsp_data, 'info', None)
                if not info_val:
                    print(f"[HWCHECK][ALLOC] No 'info' in response")
                    last_error = "No SSH info in response"
                    if attempt < MAX_TRIES:
                        await asyncio.sleep(RETRY_DELAY)
                    continue

                decrypted = rsa.decrypt_data(private_key.encode(), b64decode(info_val))
                info = json.loads(decrypted)
                print(f"[HWCHECK][ALLOC] Decrypted info keys: {list(info.keys())}")

                # Handle missing fields gracefully
                ssh_port = info.get("port")
                ssh_user = info.get("username", "root")

                if not ssh_port:
                    print(f"[HWCHECK][ALLOC] Missing SSH port in response")
                    last_error = "No SSH port in response"
                    if attempt < MAX_TRIES:
                        await asyncio.sleep(RETRY_DELAY)
                    continue

                # SSH key-based auth - we use the private key we generated, not a password from miner
                s = state.stats.get(uid, {})
                gpu_name = (s.get("gpu_specs") or {}).get("gpu_name") or s.get("gpu_name") or "Unknown GPU"
                gpu_num = (s.get("gpu_specs") or {}).get("num_gpus") or s.get("gpu_num", 0)

                details = {
                    "ssh": {
                        "host": axon.ip,
                        "port": ssh_port,
                        "username": ssh_user,
                        "auth": "ssh_key",  # Key-based auth indicator
                    },
                    "ssh_private_key": ssh_private_key,  # Paramiko Ed25519Key object
                    "axon": {"ip": axon.ip, "port": axon.port},
                    "gpu_name": gpu_name,
                    "gpu_count": gpu_num,
                }
                print(f"[HWCHECK][ALLOC] Success! SSH port={ssh_port}, user={ssh_user}, auth=ssh_key")
                return {"status": True, "uid": uid, "hotkey": hotkey, "public_key": public_key, "details": details}
            else:
                print(f"[HWCHECK][ALLOC] status=False or missing in response")
                last_error = "Miner declined allocation"
                if attempt < MAX_TRIES:
                    await asyncio.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"[HWCHECK][ALLOC][ERROR] Attempt {attempt}: {type(e).__name__}: {e}")
            last_error = str(e)
            if attempt < MAX_TRIES:
                await asyncio.sleep(RETRY_DELAY)

    raise RuntimeError(f"All allocation attempts failed. Last error: {last_error}")

async def _deallocate(hotkey: str, public_key: Optional[str], network_label: str) -> Dict[str, Any]:
    """Deallocate the test container."""
    print(f"[HWCHECK][DEALLOC] Request for hotkey={hotkey}")
    spec = NETWORKS[network_label]
    wallet = spec["wallet"]

    metagraph = await _bt_metagraph_sync(network_label)
    try:
        uid = metagraph.hotkeys.index(hotkey)
    except ValueError:
        return {"status": True}

    axon = metagraph.axons[uid]
    for attempt in range(3):
        try:
            async with bt.dendrite(wallet=wallet) as dendrite:
                rsp = await dendrite(
                    axon,
                    Allocate(timeline=0, checking=False, public_key=public_key),
                    timeout=30,
                )
                if rsp and rsp.get("status", True):
                    return {"status": True}
        except Exception as e:
            print(f"[HWCHECK][DEALLOC] Attempt {attempt+1}: {e}")
        await asyncio.sleep(5)

    return {"status": True}

# ───────────────────────── API Endpoints ─────────────────────────
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

@router.post("/start")
async def start_hardware_check(
    request: Request,
    payload: dict = Body(...),
    x_admin_key: str = Header(None),
):
    """Start a hardware check for a miner."""
    if not x_admin_key or x_admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Forbidden")

    ip = _client_ip(request)
    key = x_admin_key or "anon"

    # Rate limiting
    if not _rate_ok(_calls_per_key[key], _LIM_PER_KEY):
        ra = _retry_after(_calls_per_key[key], _LIM_PER_KEY)
        return JSONResponse({"detail": "Per-key rate limit exceeded"}, status_code=429, headers={"Retry-After": str(ra)})
    if not _rate_ok(_calls_per_ip[ip], _LIM_PER_IP):
        ra = _retry_after(_calls_per_ip[ip], _LIM_PER_IP)
        return JSONResponse({"detail": "Per-IP rate limit exceeded"}, status_code=429, headers={"Retry-After": str(ra)})
    if not _rate_ok(_calls_global, _LIM_GLOBAL):
        ra = _retry_after(_calls_global, _LIM_GLOBAL)
        return JSONResponse({"detail": "Global rate limit exceeded"}, status_code=429, headers={"Retry-After": str(ra)})

    hotkey = payload.get("hotkey")
    if not hotkey:
        raise HTTPException(status_code=400, detail="hotkey is required")

    network_label = _normalize_net_label(payload.get("network"))

    await _start_workers()

    # Check if hotkey already has a running job
    existing_job_id = _busy_hotkeys.get(hotkey)
    if existing_job_id:
        existing = _JOBS.get(existing_job_id, {})
        if existing and existing.get("status") in {"created", "queued", "running"}:
            return JSONResponse(
                {"detail": "Hotkey already busy", "job_id": existing_job_id, "status": existing.get("status")},
                status_code=409
            )
        _busy_hotkeys.pop(hotkey, None)

    job_id = _new_job()
    _JOBS[job_id]["status"] = "queued"
    _JOBS[job_id]["network"] = network_label
    _JOBS[job_id]["reserved_hotkey"] = hotkey
    _busy_hotkeys[hotkey] = job_id

    try:
        _job_queue.put_nowait((job_id, hotkey, network_label))
    except asyncio.QueueFull:
        if _busy_hotkeys.get(hotkey) == job_id:
            _busy_hotkeys.pop(hotkey, None)
        _JOBS.pop(job_id, None)
        return JSONResponse(
            {"detail": "Queue full. Try later."},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": "15"}
        )

    return {"job_id": job_id}

@router.post("/cancel/{job_id}")
async def cancel(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    job["cancel"].set()
    return {"status": "cancelling"}

@router.get("/stats")
async def stats():
    queued = _job_queue.qsize()
    running = _running_workers
    busy = {hk: jid for hk, jid in _busy_hotkeys.items()
            if _JOBS.get(jid, {}).get("status") in {"created", "queued", "running"}}
    return {"queued": queued, "running": running, "busy_hotkeys": busy}

# ───────────────────────── Core Hardware Check Runner ─────────────────────────
async def _run_hardware_check(job_id: str, hotkey: str, network_label: str):
    """
    Run hardware check:
    1. Sync metagraph to find miner IP
    2. Check PoG stats (gpu_name, num_gpus > 0, score > 0)
    3. Check axon port
    4. Allocate container
    5. Check container ports (SSH)
    6. SSH connect
    7. Get GPU info via nvidia-smi
    8. Check W&B active run
    9. Deallocate
    """
    job = _JOBS[job_id]
    cancel_event = job["cancel"]
    job["status"] = "running"

    ssh = None
    alloc: Dict[str, Any] | None = None

    def _check_cancel():
        if cancel_event.is_set():
            raise asyncio.CancelledError()

    try:
        # 1) Get metagraph to find miner IP
        _append(job_id, "info", "Syncing metagraph...")
        await _tick()
        metagraph = await _bt_metagraph_sync(network_label)
        try:
            uid = metagraph.hotkeys.index(hotkey)
        except ValueError:
            raise RuntimeError(f"Hotkey {hotkey} not found in metagraph")

        axon = metagraph.axons[uid]
        miner_ip = axon.ip
        _append(job_id, "success", "Metagraph synced.", {"uid": uid, "miner_ip": miner_ip})
        await _tick()
        _check_cancel()

        # 2) Check PoG stats (quick check if miner passed PoG3)
        _append(job_id, "info", "Checking PoG stats...")
        await _tick()
        miner_stats = state.stats.get(uid, {}) or {}
        gpu_specs = miner_stats.get("gpu_specs") or {}
        pog_gpu_name = gpu_specs.get("gpu_name") or miner_stats.get("gpu_name")
        pog_num_gpus = gpu_specs.get("num_gpus")
        if pog_num_gpus is None:
            pog_num_gpus = miner_stats.get("gpu_num", 0)
        pog_score = miner_stats.get("score")

        pog_pass = bool(
            pog_gpu_name
            and isinstance(pog_num_gpus, (int, float)) and pog_num_gpus > 0
            and pog_score is not None and pog_score > 0
        )

        pog_info = {
            "gpu_name": pog_gpu_name or "N/A",
            "num_gpus": pog_num_gpus if isinstance(pog_num_gpus, (int, float)) else 0,
            "score": pog_score if pog_score is not None else "N/A",
            "pog_pass": pog_pass,
        }

        if pog_pass:
            _append(job_id, "success", "PoG stats verified.", pog_info)
        else:
            _append(job_id, "warning", "PoG stats indicate miner has NOT passed PoG3.", pog_info)
        await _tick()
        _check_cancel()

        # 3) Pre-allocation port check (only axon - SSH/external require container)
        _append(job_id, "info", f"Checking axon port on {miner_ip}...")
        await _tick()
        axon_result = await asyncio.to_thread(check_axon_port, miner_ip)

        axon_open = axon_result["all_open"]
        if axon_open:
            _append(job_id, "success", "Axon port accessible.", axon_result["ports"])
        else:
            _append(job_id, "error", "Axon port not accessible - miner may be offline.", axon_result["ports"])
            raise RuntimeError("Miner axon port not reachable")
        await _tick()
        _check_cancel()

        # 4) Allocate container
        _append(job_id, "info", "Allocating container...")
        await _tick()
        alloc = await _allocate(hotkey=hotkey, network_label=network_label)
        if not alloc or not alloc.get("status"):
            raise RuntimeError(f"Allocation failed: {alloc}")

        details = alloc["details"]
        job["hotkey"] = alloc.get("hotkey")
        job["public_key"] = alloc.get("public_key")
        job["ssh_private_key"] = details.get("ssh_private_key")  # Store for SSH connection
        _append(job_id, "success", "Allocation successful.", {"uid": alloc.get("uid"), "auth": "ssh_key"})
        await _tick()
        _check_cancel()

        # 5) Post-allocation port check (SSH + external)
        ssh_port = details["ssh"]["port"]
        _append(job_id, "info", f"Checking container ports (SSH:{ssh_port})...")
        await _tick()

        # Wait briefly for container to start
        await asyncio.sleep(2)

        container_ports = await asyncio.to_thread(check_container_ports, details["ssh"]["host"], ssh_port)
        port_summary = {name: info["open"] for name, info in container_ports["ports"].items()}
        # Merge with axon result
        port_summary.update({name: info["open"] for name, info in axon_result["ports"].items()})

        if container_ports["all_open"]:
            _append(job_id, "success", "Container ports accessible.", container_ports["ports"])
        else:
            _append(job_id, "warning", "Some container ports not accessible.", container_ports["ports"])
        await _tick()
        _check_cancel()

        # 6) SSH connect (using SSH key-based auth)
        _append(job_id, "info", "Connecting via SSH (key-based auth)...")
        await _tick()
        ssh_private_key = details.get("ssh_private_key")
        if ssh_private_key:
            # Key-based auth
            ssh = await asyncio.to_thread(
                ssh_connect_key,
                host=details["ssh"]["host"],
                port=ssh_port,
                username=details["ssh"]["username"],
                private_key=ssh_private_key,
                timeout=25,
            )
        else:
            # Fallback to password auth (legacy)
            ssh = await asyncio.to_thread(
                ssh_connect,
                host=details["ssh"]["host"],
                port=ssh_port,
                username=details["ssh"]["username"],
                password=details["ssh"].get("password", ""),
                timeout=25,
            )
        _append(job_id, "success", "SSH connected.")
        await _tick()
        _check_cancel()

        # 7) Get GPU info via nvidia-smi (no script upload needed)
        _append(job_id, "info", "Querying GPU info via nvidia-smi...")
        await _tick()
        ginfo = await asyncio.to_thread(get_gpu_info_nvidia_smi, ssh)
        num_gpus = ginfo.get("num_gpus", 0)
        if num_gpus <= 0:
            raise RuntimeError("No GPUs detected.")

        gpu_name = (ginfo.get("gpu_names") or ["Unknown"])[0]
        vram_gb = ginfo.get("vram_gb")

        _append(job_id, "success", "GPU detected.", {
            "gpu_name": gpu_name,
            "num_gpus": num_gpus,
            "vram_gb": vram_gb,
        })
        await _tick()
        _check_cancel()

        # 8) Check W&B active run
        _append(job_id, "info", "Checking W&B active run...")
        await _tick()
        try:
            runs_map = (state.active_wandb_runs or {}).get(network_label, {}) or {}
            run_path = runs_map.get(job["hotkey"])

            if not run_path:
                _append(job_id, "warning", "No active W&B run found.", {"hotkey": job["hotkey"]})
            else:
                wandb_info = {"run_path": run_path, "wandb_active": True}

                # Add cached specs for Mainnet
                if network_label == "Mainnet":
                    specs_cache = state.hardware_specs_cache or {}
                    entry = specs_cache.get(alloc.get("uid")) if alloc else None
                    if entry:
                        details = (entry or {}).get("details") or {}
                        gpu = details.get("gpu", {}) or {}
                        wandb_info["specs"] = {
                            "gpu_name": gpu.get("name"),
                            "gpu_count": gpu.get("count"),
                            "gpu_capacity_gib": round(float(gpu.get("capacity", 0)) / 1024.0, 2) if gpu.get("capacity") else None,
                        }

                _append(job_id, "success", "W&B active run found.", wandb_info)
        except Exception as e:
            _append(job_id, "error", f"W&B check failed: {e}")
        await _tick()

        # 9) Finalize
        all_ports_open = axon_open and container_ports["all_open"]
        result = {
            "hotkey": job["hotkey"],
            "network": network_label,
            "uid": alloc.get("uid") if alloc else None,
            "miner_ip": miner_ip,
            "pog_stats": pog_info,
            "ports": port_summary,
            "ports_all_open": all_ports_open,
            "gpu": {
                "name": gpu_name,
                "count": num_gpus,
                "vram_gb": vram_gb,
            },
            "wandb_active": bool(run_path) if 'run_path' in dir() else False,
        }
        _JOBS[job_id]["result"] = result
        _JOBS[job_id]["status"] = "done"
        _append(job_id, "success", "Hardware check complete.", result)
        await _tick()

    except asyncio.CancelledError:
        _JOBS[job_id]["status"] = "cancelled"
        _append(job_id, "warning", "Hardware check cancelled.")
        await _tick()
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _append(job_id, "error", f"{type(e).__name__}: {e}")
        await _tick()
    finally:
        # Cleanup
        try:
            if ssh:
                await asyncio.to_thread(ssh_close, ssh)
        except Exception:
            pass
        try:
            hk = _JOBS[job_id].get("hotkey")
            pk = _JOBS[job_id].get("public_key")
            net = _JOBS[job_id].get("network", "Testnet")
            if hk:
                _append(job_id, "info", "Deallocating...")
                await _tick()
                await _deallocate(hotkey=hk, public_key=pk, network_label=net)
                _append(job_id, "success", "Deallocated.")
                await _tick()
        except Exception as e:
            _append(job_id, "error", f"Deallocation error: {e}")
            await _tick()
        finally:
            try:
                reserved = _JOBS[job_id].get("reserved_hotkey")
                if reserved and _busy_hotkeys.get(reserved) == job_id:
                    _busy_hotkeys.pop(reserved, None)
            except Exception:
                pass
