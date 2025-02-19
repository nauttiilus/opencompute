from fastapi import FastAPI
from typing import Dict, List, Any
import bittensor as bt
import wandb
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Load environment variables
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")

# Constants for W&B
PUBLIC_WANDB_NAME = "opencompute"
PUBLIC_WANDB_ENTITY = "neuralinternet"

# Initialize the Bittensor metagraph with the specified netuid
metagraph = bt.metagraph(netuid=27)

# Global caches
hardware_specs_cache: Dict[int, Dict[str, Any]] = {}
allocated_hotkeys_cache: List[str] = []
penalized_hotkeys_cache: List[str] = []

# IMPORTANT: We'll store the validator's stats here
stats: Dict[int, Any] = {}

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)


def fetch_validator_stats(api, run_path: str) -> Dict[int, Any]:
    """
    Fetches the 'stats' dict from the specified validator run (run_path), 
    and converts string keys to integer keys. 
    """
    try:
        run = api.run(run_path)  
        if run:
            run_config = run.config
            raw_stats = run_config.get("stats", {})  # This is the big dict of stats
            converted = {}
            # Convert keys like '0','1','2' to integers
            for k, v in raw_stats.items():
                try:
                    converted[int(k)] = v
                except:
                    # if k isn't an integer string, skip or handle
                    pass
            return converted
    except Exception as e:
        print(f"Error fetching validator stats from {run_path}: {e}")
    return {}


def fetch_hardware_specs(api, hotkeys: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Fetch hardware specs from all miner runs in W&B, 
    and attach corresponding stats from the global 'stats' dict.
    """
    global stats
    db_specs_dict: Dict[int, Dict[str, Any]] = {}
    project_path = f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_NAME}"

    # Grab *all* runs in the opencompute project
    runs = api.runs(project_path)
    try:
        for run in runs:
            run_config = run.config
            hotkey = run_config.get('hotkey')
            role = run_config.get('role')
            details = run_config.get('specs')

            # We only care about miners
            if hotkey in hotkeys and role == 'miner' and isinstance(details, dict):
                index = hotkeys.index(hotkey)
                db_specs_dict[index] = {
                    "hotkey": hotkey,
                    "details": details,
                    # Pull the stats for that index (UID) from the global stats
                    "stats": stats.get(index)
                }
    except Exception as e:
        print(f"An error occurred while getting specs from wandb: {e}")

    return db_specs_dict


def get_allocated_hotkeys(api, run_path: str) -> List[str]:
    """
    Fetch allocated hotkeys from a single validator run
    by looking at the stats.  We only gather those where 'allocated' == True.
    """
    allocated_hotkeys = []
    try:
        run = api.run(run_path)  
        if run:
            run_config = run.config
            stats_local = run_config.get("stats", {})  
            for uid_str, data in stats_local.items():
                # If data is a dict and 'allocated' is True:
                if isinstance(data, dict) and data.get("allocated", False):
                    hotkey = data.get("hotkey")
                    if hotkey:
                        allocated_hotkeys.append(hotkey)
    except Exception as e:
        print(f"Error fetching allocated hotkeys from stats for run {run_path}: {e}")

    return allocated_hotkeys


def get_penalized_hotkeys_id(api, run_path: str) -> List[str]:
    """
    Fetch the penalized hotkeys from a *specific* validator run's 
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
        if penalized_hotkeys_checklist:
            # If the checklist is a list, each item might be a dict with "hotkey"
            for entry in penalized_hotkeys_checklist:
                # e.g. entry = {"hotkey": "...", ...}
                hotkey = entry.get('hotkey')
                if hotkey:
                    penalized_keys_list.append(hotkey)

    except Exception as e:
        print(f"Run path: {run_path}, Error: {e}")

    return penalized_keys_list


async def sync_data_periodically():
    """
    Background task that periodically:
     1) Syncs the metagraph 
     2) Fetches the validator stats 
     3) Fetches miner specs 
     4) Fetches allocated & penalized hotkeys
    """
    global hardware_specs_cache, allocated_hotkeys_cache, penalized_hotkeys_cache, stats
    validator_run_path = "neuralinternet/opencompute/8etd9d95"  # Example
    validator_run_path2 = "neuralinternet/opencompute/0djlnjjs"  # Example

    while True:
        try:
            metagraph.sync()

            # Use the W&B API in a background thread
            loop = asyncio.get_event_loop()
            wandb.login(key=api_key)
            api = wandb.Api()

            # 1) Load validator stats for the specified run
            new_stats = await loop.run_in_executor(
                executor,
                fetch_validator_stats,
                api,
                validator_run_path
            )
            # Update the global stats
            stats = new_stats

            # 2) Load hardware specs from miners
            hotkeys = metagraph.hotkeys
            hardware_specs = await loop.run_in_executor(
                executor,
                fetch_hardware_specs,
                api,
                hotkeys
            )
            hardware_specs_cache = hardware_specs

            # 3) Fetch allocated hotkeys from the same validator run
            allocated = await loop.run_in_executor(
                executor,
                get_allocated_hotkeys,
                api,
                validator_run_path2
            )
            allocated_hotkeys_cache = allocated

            # 4) Fetch penalized hotkeys from that validator run
            penalized = await loop.run_in_executor(
                executor,
                get_penalized_hotkeys_id,
                api,
                validator_run_path
            )
            penalized_hotkeys_cache = penalized

        except Exception as e:
            print(f"An error occurred during periodic sync: {e}")

        # Sleep 10 minutes between syncs
        await asyncio.sleep(600)


@app.on_event("startup")
async def startup_event():
    # Kick off the background sync task
    asyncio.create_task(sync_data_periodically())


@app.get("/keys")
async def get_keys() -> Dict[str, List[str]]:
    hotkeys = metagraph.hotkeys
    return {"keys": hotkeys}


@app.get("/specs")
async def get_specs() -> Dict[str, Dict[int, Dict[str, Any]]]:
    return {"specs": hardware_specs_cache}


@app.get("/allocated_keys")
async def get_allocated_keys() -> Dict[str, List[str]]:
    return {"allocated_keys": allocated_hotkeys_cache}


@app.get("/penalized_keys")
async def get_penalized_keys() -> Dict[str, List[str]]:
    return {"penalized_keys": penalized_hotkeys_cache}


# To run the server:
# uvicorn server:app --reload --host 0.0.0.0 --port 8316
# pm2 start uvicorn --interpreter python3 --name opencompute_server -- \
#      --host 0.0.0.0 --port 8000 server:app
