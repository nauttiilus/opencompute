from fastapi import FastAPI, Header, HTTPException
from typing import Dict, List, Any
import bittensor as bt
import wandb
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
# Import the synchronous Metagraph class
from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
import argparse
import json
import yaml
import httpx
import copy
import random
from typing import Dict, List, Any

app = FastAPI()

# Load environment variables
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
ENABLE_CONFIG_WRITE = os.getenv("ENABLE_CONFIG_WRITE", "false").strip().lower() in ("1","true","yes","on")


# Constants for W&B
PUBLIC_WANDB_NAME = "opencompute"
PUBLIC_WANDB_ENTITY = "neuralinternet"

# Global caches
hardware_specs_cache: Dict[int, Dict[str, Any]] = {}
allocated_hotkeys_cache: List[str] = []
penalized_hotkeys_cache: List[str] = []
hotkeys_cache: List[str] = []
metagraph_cache: Dict[str, Any] = {}
subnet_cache: Dict[str, Any] = {}
price_cache: Dict[str, Any] = {}

# Validator stats
stats: Dict[int, Any] = {}

# Thread pool
executor = ThreadPoolExecutor(max_workers=4)

def fetch_validator_stats(api, run_path: str) -> Dict[int, Any]:
    """Fetches the 'stats' dict from a validator run, converting string keys to integer keys."""
    try:
        run = api.run(run_path)
        if run:
            run_config = run.config
            raw_stats = run_config.get("stats", {})
            converted = {}
            for k, v in raw_stats.items():
                try:
                    converted[int(k)] = v
                except:
                    pass
            return converted
    except Exception as e:
        print(f"Error fetching validator stats from {run_path}: {e}")
    return {}

def interpolate_details(
    donor_details: Dict[str, Any],
    donor_gpu_count: int,
    target_gpu_count: int
) -> Dict[str, Any]:
    """
    Scales certain fields in 'donor_details' from 'donor_gpu_count' to 'target_gpu_count':
      - cpu.count
      - gpu.count
      - gpu.capacity
      - gpu.details[*].capacity
      - ram.{total,used,free,available}
    Also applies a random multiplier (1.1 to 1.9) to disk space fields.
    Leaves cpu.frequency, disk read/write speeds, etc., unchanged.

    Returns a deep copy, not altering the original.
    """
    new_data = copy.deepcopy(donor_details)

    # If donor or target is invalid or the same, no scaling
    if donor_gpu_count <= 0 or target_gpu_count <= 0 or donor_gpu_count == target_gpu_count:
        # We still apply the disk multiplier, even if GPU counts match
        pass
    else:
        scale_factor = float(target_gpu_count) / float(donor_gpu_count)

        # ----- Scale CPU.count -----
        cpu_section = new_data.get("cpu", {})
        if isinstance(cpu_section.get("count"), (int, float)):
            old_val = cpu_section["count"]
            cpu_section["count"] = old_val * scale_factor
            # If you prefer an integer, do: cpu_section["count"] = int(old_val * scale_factor)

        # ----- Scale GPU -----
        gpu_section = new_data.get("gpu", {})
        if isinstance(gpu_section, dict):
            # Force GPU count to the new target
            gpu_section["count"] = target_gpu_count

            # Scale total GPU capacity if present
            if "capacity" in gpu_section and isinstance(gpu_section["capacity"], (int, float)):
                gpu_section["capacity"] *= scale_factor

            # Scale capacity of each GPU in 'details'
            details_list = gpu_section.get("details", [])
            if isinstance(details_list, list):
                for g in details_list:
                    if isinstance(g, dict) and isinstance(g.get("capacity"), (int, float)):
                        g["capacity"] *= scale_factor

        # ----- Scale certain RAM fields -----
        ram_section = new_data.get("ram", {})
        if isinstance(ram_section, dict):
            for field in ["total", "free", "used", "available"]:
                val = ram_section.get(field)
                if isinstance(val, (int, float)) and val > 0:
                    ram_section[field] = val * scale_factor

    # ----- Random multiplier for hard disk fields (total, free, used) -----
    hard_disk_section = new_data.get("hard_disk", {})
    if isinstance(hard_disk_section, dict):
        disk_factor = random.uniform(1.1, 1.9)
        for field in ["total", "used", "free"]:
            val = hard_disk_section.get(field)
            if isinstance(val, (int, float)) and val > 0:
                hard_disk_section[field] = val * disk_factor

    return new_data

def _fetch_hardware_specs(api, hotkeys: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    1) Pull all W&B runs once (project path).
       - store hotkey -> specs in 'wandb_hotkey_map'.

    2) For each UID in stats:
       A) If we have W&B details for that hotkey, use them directly.
       B) Otherwise, find any other hotkey in stats that has the 
          same GPU name *and* W&B details. Copy & scale if needed.
       C) If no fallback found, use empty {}.

    3) Return a dict keyed by UID with {hotkey, details, stats}.
    4) Print a single line for debugging each UID.

    Now includes a random disk multiplier between 1.1 and 1.9.
    """
    global stats
    db_specs_dict: Dict[int, Dict[str, Any]] = {}
    project_path = f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_NAME}"

    # --- 1) Build hotkey -> specs map from W&B
    wandb_hotkey_map: Dict[str, Dict[str, Any]] = {}
    try:
        runs = api.runs(project_path)
        for run in runs:
            run_config = getattr(run, 'config', None)
            if not run_config or not isinstance(run_config, dict):
                continue
            if run_config.get("role") != "miner":
                continue

            details = run_config.get("specs")
            if not isinstance(details, dict):
                continue

            run_hotkey = run_config.get("hotkey")
            if isinstance(run_hotkey, str):
                wandb_hotkey_map[run_hotkey] = details
    except Exception as e:
        print(f"An error occurred while fetching runs from wandb: {e}")

    # --- 2) Identify all hotkeys in stats that DO have W&B details, grouped by GPU name
    hotkeys_by_gpu_name = {}  # gpu_name -> list of hotkeys in W&B
    stats_gpu_map = {}        # hotkey -> (gpu_name, gpu_count)

    for uid, stat_data in stats.items():
        if not isinstance(stat_data, dict):
            continue
        st_hotkey = stat_data.get("hotkey")
        gpu_info = stat_data.get("gpu_specs", {})
        if not isinstance(st_hotkey, str) or not isinstance(gpu_info, dict):
            continue

        gpu_name = gpu_info.get("gpu_name")
        gpu_count = gpu_info.get("num_gpus")
        stats_gpu_map[st_hotkey] = (gpu_name, gpu_count)

        # If this hotkey is in W&B, add it to fallback donors for that gpu_name
        if st_hotkey in wandb_hotkey_map and isinstance(gpu_name, str):
            hotkeys_by_gpu_name.setdefault(gpu_name, []).append(st_hotkey)

    # --- 3) For each UID, build final details
    for uid, stat_data in stats.items():
        if not isinstance(stat_data, dict):
            db_specs_dict[uid] = {
                "hotkey": None,
                "details": {},
                "stats": stat_data
            }
            print(f"UID={uid} | hotkey=None | gpu_name=None | num_gpus=None")
            continue

        st_hotkey = stat_data.get("hotkey")
        gpu_info = stat_data.get("gpu_specs", {})
        if not isinstance(st_hotkey, str) or not isinstance(gpu_info, dict):
            # No valid hotkey or GPU info
            db_specs_dict[uid] = {
                "hotkey": st_hotkey,
                "details": {},
                "stats": stat_data
            }
            print(f"UID={uid} | hotkey={st_hotkey} | gpu_name=None | num_gpus=None")
            continue

        needed_gpu_name = gpu_info.get("gpu_name")
        needed_gpu_count = gpu_info.get("num_gpus")

        final_details = {}

        # (A) Direct W&B match for this hotkey?
        if st_hotkey in wandb_hotkey_map:
            final_details = copy.deepcopy(wandb_hotkey_map[st_hotkey])

            # If GPU counts differ, scale
            donor_gpu_name, donor_gpu_count = stats_gpu_map[st_hotkey]
            if isinstance(donor_gpu_count, int) and isinstance(needed_gpu_count, int):
                final_details = interpolate_details(final_details, donor_gpu_count, needed_gpu_count)

        else:
            # (B) Fallback: same GPU name => pick any "donor" that has W&B
            if isinstance(needed_gpu_name, str) and (needed_gpu_name in hotkeys_by_gpu_name):
                donor_list = hotkeys_by_gpu_name[needed_gpu_name]
                if donor_list:
                    donor_hotkey = donor_list[0]  # pick the first or refine logic if needed
                    donor_details = wandb_hotkey_map[donor_hotkey]
                    final_details = copy.deepcopy(donor_details)

                    # Scale if needed
                    donor_gpu_name, donor_gpu_count = stats_gpu_map[donor_hotkey]
                    if isinstance(donor_gpu_count, int) and isinstance(needed_gpu_count, int):
                        final_details = interpolate_details(final_details, donor_gpu_count, needed_gpu_count)

        db_specs_dict[uid] = {
            "hotkey": st_hotkey,
            "details": final_details,
            "stats": stat_data
        }

        print(f"UID={uid} | hotkey={st_hotkey} | gpu_name={needed_gpu_name} | num_gpus={needed_gpu_count}")

    return db_specs_dict

def fetch_hardware_specs(api, hotkeys: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Pull W&B runs for miners and map their specs to hotkeys.
    If a UID's hotkey has a run, use its details.
    Otherwise, leave details as {}.
    """
    global stats
    db_specs_dict: Dict[int, Dict[str, Any]] = {}
    project_path = f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_NAME}"

    # --- 1) Build hotkey -> specs map from W&B
    wandb_hotkey_map: Dict[str, Dict[str, Any]] = {}
    try:
        runs = api.runs(project_path)
        for run in runs:
            run_config = getattr(run, 'config', None)
            if not run_config or not isinstance(run_config, dict):
                continue
            if run_config.get("role") != "miner":
                continue

            details = run_config.get("specs")
            if not isinstance(details, dict):
                continue

            run_hotkey = run_config.get("hotkey")
            if isinstance(run_hotkey, str):
                wandb_hotkey_map[run_hotkey] = details
    except Exception as e:
        print(f"An error occurred while fetching runs from wandb: {e}")

    # --- 2) For each UID, use details if hotkey exists in W&B
    for uid, stat_data in stats.items():
        if not isinstance(stat_data, dict):
            db_specs_dict[uid] = {
                "hotkey": None,
                "details": {},
                "stats": stat_data
            }
            print(f"UID={uid} | hotkey=None")
            continue

        st_hotkey = stat_data.get("hotkey")
        if isinstance(st_hotkey, str) and st_hotkey in wandb_hotkey_map:
            final_details = copy.deepcopy(wandb_hotkey_map[st_hotkey])
        else:
            final_details = {}

        db_specs_dict[uid] = {
            "hotkey": st_hotkey,
            "details": final_details,
            "stats": stat_data
        }

        print(f"UID={uid} | hotkey={st_hotkey}")

    return db_specs_dict

def get_allocated_hotkeys(api, run_path: str) -> List[str]:
    """
    Fetch allocated hotkeys from a validator run 
    by looking at the stats for where 'allocated' == True.
    """
    allocated_hotkeys = []
    try:
        run = api.run(run_path)
        if run:
            run_config = run.config
            stats_local = run_config.get("stats", {})
            for _, data in stats_local.items():
                if isinstance(data, dict) and data.get("allocated", False):
                    hotkey = data.get("hotkey")
                    if hotkey:
                        allocated_hotkeys.append(hotkey)
    except Exception as e:
        print(f"Error fetching allocated hotkeys from stats for run {run_path}: {e}")
    return allocated_hotkeys

def get_penalized_hotkeys_id(api, run_path: str) -> List[str]:
    """
    Fetch the penalized hotkeys from a validator run's 
    'penalized_hotkeys_checklist' in the config.
    """
    penalized_keys_list: List[str] = []
    try:
        run = api.run(run_path)
        if not run:
            print(f"No run info found for ID {run_path}.")
            return []
        run_config = run.config
        penalized_hotkeys_checklist = run_config.get('penalized_hotkeys_checklist', [])
        for entry in penalized_hotkeys_checklist:
            hotkey = entry #.get('hotkey')
            if hotkey:
                penalized_keys_list.append(hotkey)
    except Exception as e:
        print(f"Run path: {run_path}, Error: {e}")
    return penalized_keys_list

def get_metagraph():

    subtensor = bt.subtensor(network="finney")
    
    # Fetch and sync the metagraph for subnet 27
    metagraph = subtensor.metagraph(netuid=27)
    metagraph.sync()

    return metagraph

def get_subnet_alpha_price():
    try:
        subtensor = bt.subtensor(network="finney")
        subnet_info = subtensor.subnet(27)
        return subnet_info.price, subnet_info.emission
    except Exception as e:
        print(f"Error fetching subnet info: {e}")
        return None, None

async def fetch_tao_price():
    """Fetches the latest TAO price from CoinGecko asynchronously."""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price_cache["tao_price"] = data["bittensor"]["usd"]
                #print(f"✅ Updated TAO/USD Price: ${price_cache['tao_price']}")
            else:
                print(f"⚠️ Failed to fetch TAO price: {response.status_code}")
    except Exception as e:
        print(f"❌ Error fetching TAO price: {e}")

async def sync_data_periodically():
    """
    Background task that periodically:
     1) Syncs the metagraph 
     2) Fetches the validator stats 
     3) Fetches miner specs 
     4) Fetches allocated & penalized hotkeys
    """
    validator_run_path = "neuralinternet/opencompute/0djlnjjs"  # Example
    validator_run_path2 = "neuralinternet/opencompute/ckig4h3x"  # Example

    global metagraph_cache, hardware_specs_cache, allocated_hotkeys_cache, penalized_hotkeys_cache, stats, subnet_cache, price_cache

    # Create the wandb API object once
    api = wandb.Api()

    while True:
        try:
            loop = asyncio.get_running_loop()

            await loop.run_in_executor(executor, api.flush)

            # 1) Metagraph sync. Because it's synchronous, we do it in a thread executor.
            print("[DEBUG] Starting metagraph sync in background task ...")
            metagraph = await loop.run_in_executor(executor, get_metagraph)
            print("[DEBUG] Metagraph is up-to-date.")

            # 2) Store all metagraph details in cache
            def build_metagraph_cache(m):
                return {
                    "version": m.version.tolist(),
                    "n": m.n.tolist(),
                    "block": m.block.tolist(),
                    "stake": m.S.tolist(),
                    "total_stake": m.total_stake.tolist(),
                    "ranks": m.R.tolist(),
                    "trust": m.T.tolist(),
                    "consensus": m.C.tolist(),
                    "validator_trust": m.validator_trust.tolist(),
                    "incentive": m.I.tolist(),
                    "emission": m.E.tolist(),
                    "dividends": m.D.tolist(),
                    "active": m.active.tolist(),
                    "last_update": m.last_update.tolist(),
                    "validator_permit": m.validator_permit.tolist(),
                    "weights": m.weights.tolist(),
                    "bonds": m.bonds.tolist(),
                    "uids": m.uids.tolist(),
                    "hotkeys": list(m.hotkeys),
                    "axons": [axon for axon in m.axons],
                }

            metagraph_cache = await loop.run_in_executor(executor, build_metagraph_cache, metagraph)

            alpha_price, alpha_emission = await loop.run_in_executor(executor, get_subnet_alpha_price)
            await loop.run_in_executor(executor, api.flush)

            if alpha_price is not None and alpha_emission is not None:
                subnet_cache = {
                    "alpha_price": float(alpha_price),
                    "alpha_emission": float(alpha_emission),
                }
                #print(f"Alpha Price (τ/α): {alpha_price}")
                #print(f"Alpha Emission (α/block): {alpha_emission}")
            else:
                subnet_cache = {}
            
            # 2) Load validator stats (executor again, but for CPU-bound or blocking I/O calls)
            new_stats = await loop.run_in_executor(
                executor,
                fetch_validator_stats,
                api,
                validator_run_path2
            )
            stats = new_stats

            # 3) Load hardware specs from miners
            hotkeys = metagraph.hotkeys
            hardware_specs = await loop.run_in_executor(
                executor,
                fetch_hardware_specs,
                api,
                hotkeys
            )
            hardware_specs_cache = hardware_specs
            hotkeys_cache = hotkeys

            # 4) Fetch allocated hotkeys from second validator run
            allocated = await loop.run_in_executor(
                executor,
                get_allocated_hotkeys,
                api,
                validator_run_path
            )
            allocated_hotkeys_cache = allocated

            # 5) Fetch penalized hotkeys from first validator run
            penalized = await loop.run_in_executor(
                executor,
                get_penalized_hotkeys_id,
                api,
                validator_run_path2
            )
            penalized_hotkeys_cache = penalized

            asyncio.create_task(fetch_tao_price())

        except Exception as e:
            print(f"An error occurred during periodic sync: {e}")

        # Sleep 10 minutes between syncs
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    # Login to wandb once at startup
    wandb.login(key=api_key)
    # Kick off the background sync task
    asyncio.create_task(sync_data_periodically())

@app.get("/keys")
async def get_keys() -> Dict[str, List[str]]:

    return {"keys": hotkeys_cache}

@app.get("/specs")
async def get_specs() -> Dict[str, Dict[int, Dict[str, Any]]]:
    return {"specs": hardware_specs_cache}

@app.get("/allocated_keys")
async def get_allocated_keys() -> Dict[str, List[str]]:
    return {"allocated_keys": allocated_hotkeys_cache}

@app.get("/penalized_keys")
async def get_penalized_keys() -> Dict[str, List[str]]:
    return {"penalized_keys": penalized_hotkeys_cache}

@app.get("/subnet")
async def get_subnet_data() -> Dict[str, Any]:
    return {"subnet": subnet_cache}

@app.get("/metagraph")
async def get_metagraph_data() -> Dict[str, Any]:
    """
    API endpoint to fetch the latest metagraph state.
    Returns hotkeys, stake, trust, ranks, incentive, emission, consensus, and dividends.
    """
    if not metagraph_cache:
        return {"error": "Metagraph data not available. Try again later."}

    return {"metagraph": metagraph_cache}

@app.get("/price")
async def get_tao_price() -> Dict[str, Any]:
    """API Endpoint to get the latest TAO/USD price."""
    return {"tao_price": price_cache.get("tao_price", "N/A")}

@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Returns the entire contents of config.yaml for display/editing.
    """
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="config.yaml not found")
    return {"config": cfg}

@app.put("/config")
async def update_config(
    payload: Dict[str, Any],
    x_admin_key: str = Header(None)
) -> Dict[str, Any]:
    if not ENABLE_CONFIG_WRITE:
        raise HTTPException(status_code=403, detail="Config writing disabled by server policy")
    if x_admin_key != ADMIN_KEY or not ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        with open("config.yaml", "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")
    return {"status": "ok"}

# To run the server:
# uvicorn server:app --reload --host 0.0.0.0 --port 8316
# or:
# pm2 start uvicorn --interpreter python3 --name opencompute_server -- --host 0.0.0.0 --port 8000 server:app
