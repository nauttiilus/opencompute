from fastapi import FastAPI
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
import httpx

app = FastAPI()

# Load environment variables
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")

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

def fetch_hardware_specs(api, hotkeys: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Fetch hardware specs by looking up W&B runs and matching them to the 
    hotkey in stats (ignoring the given hotkeys list). For each UID in stats:

      1. If the hotkey is found in W&B, use that run's details.
      2. Otherwise, compare only the GPU name (gpu_name).
         - Among all runs with matching gpu_name, find the run whose num_gpus 
           is the smallest >= the needed num_gpus.
         - If none are >=, pick the largest smaller one.
         - If no runs match the gpu_name, details = {}.

    This ensures every UID in stats is returned, including UID=0,
    and prints a single line for each added UID.

    We add safeguards to skip broken runs or missing data so we don't 
    get 'NoneType' object has no attribute 'get'.
    """
    global stats
    db_specs_dict: Dict[int, Dict[str, Any]] = {}
    project_path = f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_NAME}"

    try:
        runs = api.runs(project_path)
    except Exception as e:
        print(f"An error occurred while fetching runs from wandb: {e}")
        runs = []

    # Lookups
    hotkey_run_details: Dict[str, Dict[str, Any]] = {}
    # Instead of storing (gpu_name, num_gpus)->details, we store:
    #   gpu_name -> [ (num_gpus, details), (num_gpus, details), ... ]
    gpu_runs_map: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

    for run in runs:
        try:
            run_config = getattr(run, 'config', None)
            if not run_config or not isinstance(run_config, dict):
                # Skip runs if config is not a valid dict
                continue

            role = run_config.get("role", None)
            if role != "miner":
                continue

            details = run_config.get("specs", {})
            if not isinstance(details, dict):
                continue

            run_hotkey = run_config.get("hotkey", None)
            if isinstance(run_hotkey, str):
                hotkey_run_details[run_hotkey] = details

            gpu_name = details.get("gpu_name", None)
            num_gpus = details.get("num_gpus", None)

            if gpu_name and isinstance(num_gpus, int):
                if gpu_name not in gpu_runs_map:
                    gpu_runs_map[gpu_name] = []
                gpu_runs_map[gpu_name].append((num_gpus, details))

        except Exception as e_run:
            print(f"Skipping a run due to unexpected error: {e_run}")
            continue

    # Sort each gpu_name's list by ascending num_gpus for nearest-logic
    for gname in gpu_runs_map:
        gpu_runs_map[gname].sort(key=lambda x: x[0])

    # Iterate over stats to build the final db_specs_dict
    for uid, stat_data in stats.items():
        if not isinstance(stat_data, dict):
            db_specs_dict[uid] = {
                "hotkey": None,
                "details": {},
                "stats": stat_data
            }
            print(f"UID={uid} | hotkey=None | gpu_name=None | num_gpus=None")
            continue

        # Pull hotkey/gpu info from stats
        hotkey = stat_data.get("hotkey")
        gpu_specs = stat_data.get("gpu_specs", {})
        if not isinstance(gpu_specs, dict):
            gpu_specs = {}

        gpu_name = gpu_specs.get("gpu_name", None)
        num_gpus = gpu_specs.get("num_gpus", None)

        # 1) If there's a W&B run with this hotkey, take that
        details = {}
        if hotkey and hotkey in hotkey_run_details:
            details = hotkey_run_details[hotkey]
        else:
            # 2) Otherwise fallback to GPU-based logic: same gpu_name
            if gpu_name and gpu_name in gpu_runs_map and isinstance(num_gpus, int):
                # We have a sorted list of (num_gpus_run, details)
                candidates = gpu_runs_map[gpu_name]

                # Find the smallest run_num_gpus >= actual num_gpus
                chosen_details = None
                for (run_num_gpus, run_details) in candidates:
                    if run_num_gpus >= num_gpus:
                        chosen_details = run_details
                        break

                if not chosen_details:
                    # If none are >=, pick the largest smaller one
                    chosen_details = candidates[-1][1]  # last item in sorted list

                details = chosen_details

        db_specs_dict[uid] = {
            "hotkey": hotkey,
            "details": details,
            "stats": stat_data
        }

        # Print a single line for clarity
        print(f"UID={uid} | hotkey={hotkey} | gpu_name={gpu_name} | num_gpus={num_gpus}")

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
    subtensor = bt.subtensor(network="finney")
    
    # Fetch subnet information directly without metagraph
    subnet_info = subtensor.subnet(27)

    # Get the current alpha price (τ/α) and emission rate (α/block)
    alpha_price = subnet_info.price
    alpha_emission = subnet_info.emission

    return alpha_price, alpha_emission

async def fetch_tao_price():
    """Fetches the latest TAO price from CoinGecko asynchronously."""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price_cache["tao_price"] = data["bittensor"]["usd"]
                print(f"✅ Updated TAO/USD Price: ${price_cache['tao_price']}")
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
    validator_run_path = "neuralinternet/opencompute/8etd9d95"  # Example
    validator_run_path2 = "neuralinternet/opencompute/0djlnjjs"  # Example

    global metagraph_cache, hardware_specs_cache, allocated_hotkeys_cache, penalized_hotkeys_cache, stats, subnet_cache, price_cache

    # Create the wandb API object once
    api = wandb.Api()

    while True:
        try:
            loop = asyncio.get_running_loop()

            api.flush()

            # 1) Metagraph sync. Because it's synchronous, we do it in a thread executor.
            print("[DEBUG] Starting metagraph sync in background task ...")
            metagraph = await loop.run_in_executor(executor, get_metagraph)
            print("[DEBUG] Metagraph is up-to-date.")

            # 2) Store all metagraph details in cache
            metagraph_cache = {
                "version": metagraph.version.tolist(),
                "n": metagraph.n.tolist(),
                "block": metagraph.block.tolist(),
                "stake": metagraph.S.tolist(),
                "total_stake": metagraph.total_stake.tolist(),
                "ranks": metagraph.R.tolist(),
                "trust": metagraph.T.tolist(),
                "consensus": metagraph.C.tolist(),
                "validator_trust": metagraph.validator_trust.tolist(),
                "incentive": metagraph.I.tolist(),
                "emission": metagraph.E.tolist(),
                "dividends": metagraph.D.tolist(),
                "active": metagraph.active.tolist(),
                "last_update": metagraph.last_update.tolist(),
                "validator_permit": metagraph.validator_permit.tolist(),
                "weights": metagraph.weights.tolist(),
                "bonds": metagraph.bonds.tolist(),
                "uids": metagraph.uids.tolist(),
                "hotkeys": metagraph.hotkeys,  # Ensure deep copy for thread safety
                "axons": [axon for axon in metagraph.axons],
                #"neurons": [neuron.to_dict() for neuron in metagraph.neurons]
            }

            alpha_price, alpha_emission = get_subnet_alpha_price()
            print(f"Alpha Price (τ/α): {alpha_price}")
            print(f"Alpha Emission (α/block): {alpha_emission}")

            subnet_cache = {
                "alpha_price": alpha_price,
                "alpha_emission": alpha_emission,
            }
            
            # 2) Load validator stats (executor again, but for CPU-bound or blocking I/O calls)
            new_stats = await loop.run_in_executor(
                executor,
                fetch_validator_stats,
                api,
                validator_run_path
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
                validator_run_path2
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

# To run the server:
# uvicorn server:app --reload --host 0.0.0.0 --port 8316
# or:
# pm2 start uvicorn --interpreter python3 --name opencompute_server -- --host 0.0.0.0 --port 8000 server:app
