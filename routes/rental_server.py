# rental_server.py
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import json
import base64
import os
import bittensor as bt
from compute.wandb.wandb import ComputeWandb

# Path fix to import from compute/ even though it's in SN27/
import sys
from pathlib import Path

# Add root dir to path to allow: from opencompute.server import ...
sys.path.append(str(Path(__file__).resolve().parents[1]))

print("[DEBUG] sys.path:", sys.path)  # Debug print

# Import from custom utils and your compute library
from utils import RSAEncryption as rsa
from compute.protocol import Allocate
from compute.axon import ComputeSubnetSubtensor
from utils.wandb_sync import update_allocations_in_wandb
from utils.sn27_db import update_allocation_db

# Import global caches from server
import state

# Initialize the database once at startup
from db.rental_db import init_rental_db, add_rental, get_rental, remove_rental, list_all_rentals
init_rental_db()

# Create FastAPI router
router = APIRouter()

# ---------------------------- Wallet Initialization ----------------------------
from compute.utils.parser import ComputeArgPaser

# Load from .env if available
from dotenv import load_dotenv
load_dotenv()

WALLET_NAME = os.getenv("WALLET_NAME", "ni_core")
WALLET_HOTKEY = os.getenv("WALLET_HOTKEY", "ni_core1")
NETUID = int(os.getenv("NETUID", 27))

# Create a parser and generate config from scratch
parser = ComputeArgPaser()
config = parser.config

# Override config with values from env
config.wallet.name = WALLET_NAME
config.wallet.hotkey = WALLET_HOTKEY
config.netuid = NETUID

# Set default logging path
config.full_path = os.path.expanduser(
    f"~/.bittensor/logs/{WALLET_NAME}/{WALLET_HOTKEY}/netuid{NETUID}/allocator"
)
os.makedirs(config.full_path, exist_ok=True)

wallet = bt.wallet(config=config)

# WanDB
WANDB_RUN_PATH = os.getenv("WANDB_RUN_PATH")
wandb_instance = ComputeWandb(config, wallet, "validator.py")

# ---------------------------- Models ----------------------------
class AllocateRequest(BaseModel):
    hotkey: str
    timeline: int | None = 1
    docker_image: str | None = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
    ssh_public_key: str | None = None  # Optional: client-provided SSH public key


# Default SSH public key for rentals (can be overridden via env var or request)
DEFAULT_RENTAL_SSH_PUBLIC_KEY = os.getenv(
    "RENTAL_SSH_PUBLIC_KEY",
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJt//D5Ourn8ttCFUBYM9oafO4Ht8il8VrPkgfF9Qiq+ fabianmatthias_mueller@gmail.com"
)

class DeallocateRequest(BaseModel):
    hotkey: str
    public_key: str

# ---------------------------- GPU Label Mapping ----------------------------
GPU_LABELS = {
    "NVIDIA H100 80GB HBM3": "H100 80GB HBM3",
    "NVIDIA H100": "H100 80GB PCIE",
    "NVIDIA H100 PCIe": "H100 80GB PCIE",
    "NVIDIA H100 NVL": "H100 NVL",
    "NVIDIA A100-SXM4-80GB": "A100 80GB SXM4",
    "NVIDIA A100-SXM4-40GB": "A100 40GB SXM4",
    "NVIDIA A100 80GB PCIe": "A100 80GB PCIE",
    "NVIDIA L40S": "L40S",
    "NVIDIA L40": "L40",
    "NVIDIA A40": "A40",
    "NVIDIA RTX 6000 Ada Generation": "RTX 6000 Ada",
    "NVIDIA RTX A6000": "RTX A6000",
    "NVIDIA RTX A5000": "RTX A5000",
    "NVIDIA RTX A4500": "RTX A4500",
    "NVIDIA RTX A4000": "RTX A4000",
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA GeForce RTX 3090": "RTX 3090",
    "NVIDIA L4": "L4",
}
OTHER_LABEL = "Other GPUs"

# ---------------------------- Helpers ----------------------------
def _normalize_gpu_label(raw: str) -> str:
    if not raw:
        return OTHER_LABEL
    return GPU_LABELS.get(raw, raw)

def _is_pog_pass(stats_obj: dict) -> bool:
    if not isinstance(stats_obj, dict):
        return False
    name = (stats_obj.get("gpu_specs") or {}).get("gpu_name") or stats_obj.get("gpu_name")
    num = (stats_obj.get("gpu_specs") or {}).get("num_gpus")
    if num is None:
        num = stats_obj.get("gpu_num", 0)
    return bool(name) and isinstance(num, (int, float)) and num > 0

def _miner_summary(uid: int, s: dict, axon: dict) -> dict:
    gpu_name = (s.get("gpu_specs") or {}).get("gpu_name") or s.get("gpu_name") or "Unknown GPU"
    gpu_num  = (s.get("gpu_specs") or {}).get("num_gpus") or s.get("gpu_num", 0)
    label = _normalize_gpu_label(gpu_name)
    return {
        "uid": uid,
        "hotkey": s.get("hotkey"),
        "gpu_label": label,
        "gpu_name": gpu_name,
        "gpu_count": gpu_num,
        "score": s.get("score"),
        "reliability": s.get("reliability_score"),
        "axon_ip": axon.ip,
        "axon_port": axon.port,
        "version": axon.version
    }

@router.get("/rent/available")
async def list_rentable_miners() -> Dict[str, Any]:
    stats = state.stats
    metagraph_cache = state.metagraph_cache
    allocated_hotkeys_cache = state.allocated_hotkeys_cache
    penalized_hotkeys_cache = state.penalized_hotkeys_cache

    if not metagraph_cache:
        print("[DEBUG] Metagraph cache is empty")
        return {"groups": []}

    groups: Dict[str, list] = {}
    hotkeys = metagraph_cache.get("hotkeys", [])
    axons   = metagraph_cache.get("axons", [])

    print(f"[DEBUG] Total hotkeys: {len(hotkeys)}")
    print(f"[DEBUG] Total stats entries: {len(stats)}")

    rentals = list_all_rentals()
    rented_hotkeys = {r["hotkey"] for r in rentals}

    for uid, s in stats.items():
        try:
            if not isinstance(uid, int):
                print(f"[SKIP] UID={uid} is not an int")
                continue

            if not isinstance(s, dict):
                print(f"[SKIP] UID={uid} stats is not a dict: {s}")
                continue

            hk = s.get("hotkey")
            if not hk:
                print(f"[SKIP] UID={uid} has no hotkey")
                continue

            if hk in allocated_hotkeys_cache:
                print(f"[SKIP] UID={uid} is already allocated")
                continue

            if hk in penalized_hotkeys_cache:
                print(f"[SKIP] UID={uid} is penalized")
                continue

            if hk in rented_hotkeys:
                print(f"[SKIP] UID={uid} is already rented")
                continue

            if not _is_pog_pass(s):
                print(f"[SKIP] UID={uid} did not pass POG check")
                continue

            score_val = s.get("score")
            if score_val is None or score_val <= 0:
                print(f"[SKIP] UID={uid} has score=0")
                continue

            ax = axons[uid] or {}
            summ = _miner_summary(uid, s, ax)
            label = summ.get("gpu_label", "Other")
            groups.setdefault(label, []).append(summ)

        except Exception as e:
            print(f"[ERROR] UID={uid} raised exception: {e}")
            continue

    grouped = []
    for label, miners in groups.items():
        miners_sorted = sorted(
            miners,
            key=lambda m: (m.get("gpu_count") or 0, m.get("reliability") or 0),
            reverse=True
        )
        grouped.append({
            "gpu_label": label,
            "count": len(miners_sorted),
            "miners": miners_sorted
        })

    grouped.sort(key=lambda g: g["count"], reverse=True)
    print(f"[DEBUG] Returning {sum(len(g['miners']) for g in grouped)} rentable miners in {len(grouped)} groups")
    return {"groups": grouped}

@router.post("/rent/allocate")
async def allocate_miner(req: AllocateRequest = Body(...)) -> Dict[str, Any]:
    allocated_hotkeys_cache = state.allocated_hotkeys_cache
    penalized_hotkeys_cache = state.penalized_hotkeys_cache

    hk = req.hotkey
    print(f"[DEBUG] Allocation request for hotkey: {hk}")

    if not hk:
        raise HTTPException(status_code=400, detail="Hotkey is required.")

    # if hk in penalized_hotkeys_cache:
    #     print(f"[DEBUG] Hotkey {hk} is penalized.")
    #     raise HTTPException(status_code=409, detail="Miner is penalized.")

    if hk in allocated_hotkeys_cache:
        print(f"[DEBUG] Hotkey {hk} is already allocated.")
        raise HTTPException(status_code=409, detail="Miner is already allocated.")

    try:
        subtensor = bt.subtensor(network="finney")
        metagraph = subtensor.metagraph(27)
        metagraph.sync()
        print("[DEBUG] Metagraph synced successfully.")
    except Exception as e:
        print(f"[ERROR] Metagraph sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync metagraph: {e}")

    try:
        uid = metagraph.hotkeys.index(hk)
        print(f"[DEBUG] UID for hotkey {hk}: {uid}")
    except ValueError:
        print(f"[ERROR] Hotkey {hk} not found in metagraph.")
        raise HTTPException(status_code=404, detail="Hotkey not found.")

    axon = metagraph.axons[uid]

    try:
        private_key, public_key = rsa.generate_key_pair()
        print("[DEBUG] RSA keys generated.")
    except Exception as e:
        print(f"[ERROR] RSA key generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"RSA key error: {e}")

    device_requirement = {
        "cpu":       {"count": 1},
        "gpu":       {"count": 1, "capacity": 0, "type": ""},
        "hard_disk": {"capacity": 1_073_741_824},
        "ram":       {"capacity": 1_073_741_824},
        "testing":   False,
    }
    # Use client-provided SSH key or fall back to default
    ssh_key_to_use = req.ssh_public_key or DEFAULT_RENTAL_SSH_PUBLIC_KEY

    docker_requirement = {
        "base_image": req.docker_image or "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
        "ssh_key": ssh_key_to_use,
    }

    MAX_TRIES = 5
    BASE_BACKOFF_S = 1

    for attempt in range(1, MAX_TRIES + 1):
        try:
            print(f"[DEBUG] Attempt {attempt} - sending Allocate to {axon.ip}:{axon.port}")
            async with bt.dendrite(wallet=wallet) as dendrite:
                rsp = await dendrite(
                    axon,
                    Allocate(
                        timeline=req.timeline,
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

                s = state.stats.get(uid, {})
                gpu_name = (s.get("gpu_specs") or {}).get("gpu_name") or s.get("gpu_name") or "Unknown GPU"
                gpu_num  = (s.get("gpu_specs") or {}).get("num_gpus") or s.get("gpu_num", 0)

                rental_details = {
                    "ssh": {
                        "host": axon.ip,
                        "port": info["port"],
                        "username": info["username"],
                        "ssh_key": ssh_key_to_use,  # SSH key-based auth (no password)
                    },
                    "axon": {"ip": axon.ip, "port": axon.port},
                    "docker_image": docker_requirement["base_image"],
                    "gpu_name": gpu_name,
                    "gpu_count": gpu_num,
                }

                add_rental(
                    uid=uid,
                    hotkey=hk,
                    public_key = public_key,
                    ssh_key= "ssh_key_placeholder",
                    rented_by="placeholder_user_id",
                    details=rental_details
                )

                print(f"[DEBUG] Attempt {attempt} - Allocated {axon.ip}:{axon.port}")
                update_allocations_in_wandb(wandb_instance, WANDB_RUN_PATH, hotkey=hk, action="add")    # after allocation
                update_allocation_db(hk, rental_details, True)  # update SN27 validator allocation DB

                return {
                    "status": True,
                    "hotkey": hk,
                    "uid": uid,
                    "ssh": {
                        "host": axon.ip,
                        "port": info["port"],
                        "username": info["username"],
                        "auth": "ssh_key",  # Auth method indicator
                    },
                    "public_key": public_key,
                    "ssh_public_key": ssh_key_to_use,
                }

            print(f"[DEBUG] Attempt {attempt} - Allocator busy or declined.")
            return {"status": False, "detail": "Allocator busy or declined."}

        except bt.dendrite.exceptions.ServerDisconnectedError as e:
            print(f"[WARN] Disconnected on attempt {attempt}: {e}")
        except ConnectionRefusedError as e:
            print(f"[WARN] Connection refused on attempt {attempt}: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error during allocation: {e}")
            return {"status": False, "detail": f"Exception: {e}"}

        if attempt < MAX_TRIES:
            await asyncio.sleep(BASE_BACKOFF_S * attempt)

    raise HTTPException(status_code=503, detail="All allocation attempts failed.")

@router.post("/rent/deallocate")
async def deallocate_miner(req: DeallocateRequest = Body(...)) -> Dict[str, Any]:
    hk = req.hotkey
    public_key = req.public_key

    print(f"[DEBUG] Deallocation request for hotkey: {hk}")

    try:
        subtensor = bt.subtensor(network="finney")
        metagraph = subtensor.metagraph(27)
        metagraph.sync()
    except Exception as e:
        print(f"[ERROR] Failed to sync metagraph: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync metagraph")

    try:
        uid = metagraph.hotkeys.index(hk)
        axon = metagraph.axons[uid]
    except ValueError:
        print(f"[ERROR] Hotkey {hk} not found in metagraph.")
        raise HTTPException(status_code=404, detail="Hotkey not found")
    
    # Retrieve private key from DB
    if not public_key:
        try:
            rental_entry = get_rental(uid)
            if not rental_entry:
                raise Exception("Rental not found in DB")

            private_key = rental_entry.get("ssh_key")
            public_key = rental_entry.get("public_key")
            if not public_key:
                raise Exception("Public key not found in DB")
        except Exception as e:
            print(f"[ERROR] Could not retrieve keys for deallocation: {e}")
            raise HTTPException(status_code=500, detail="Missing keys for deallocation")

    retry_count = 0
    max_retries = 3
    allocation_status = True
        
    while allocation_status and retry_count < max_retries:
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
                    print(f"[DEBUG] Deallocated miner {hk}")
                    remove_rental(uid)
                    update_allocations_in_wandb(wandb_instance, WANDB_RUN_PATH, hotkey=hk, action="remove") # after deallocation
                    update_allocation_db(hk, {}, False)  # remove from SN27 validator allocation DB
                    return {"status": True, "message": f"Deallocated miner {hk}"}

                retry_count += 1
                print(f"[DEBUG] Failed to deallocate {hk} (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(5)

        except Exception as e:
            retry_count += 1
            print(f"[DEBUG] Exception during deallocation of {hk} (attempt {retry_count}/{max_retries}): {e}")
            await asyncio.sleep(5)

    # Final return if all retries fail
    print(f"[DEBUG] Final failure: Could not deallocate {hk} after {max_retries} attempts.")
    print(f"[DEBUG] Resetting allocation status of miner {hk}.")
    remove_rental(uid)
    update_allocations_in_wandb(wandb_instance, WANDB_RUN_PATH, hotkey=hk, action="remove") # after deallocation
    update_allocation_db(hk, {}, False)  # remove from SN27 validator allocation DB
    return {"status": True, "message": f"Resetted allocation status of miner {hk}."}

@router.get("/rent/rentals")
async def get_rentals():
    try:
        rentals = list_all_rentals()
        return {"rentals": rentals}
    except Exception as e:
        print(f"[ERROR] Failed to get rentals: {e}")
        return {"rentals": []}