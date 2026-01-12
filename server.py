from fastapi import FastAPI, Header, HTTPException
from typing import Dict, List, Any, Tuple, Union
import bittensor as bt
import wandb
import os
import shutil
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
import argparse
import json
import yaml
import httpx
import copy
import random
import state

app = FastAPI()

# Load environment variables
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
ENABLE_CONFIG_WRITE = os.getenv("ENABLE_CONFIG_WRITE", "false").strip().lower() in ("1","true","yes","on")


# Constants for W&B
PUBLIC_WANDB_NAME = "opencompute"
PUBLIC_WANDB_ENTITY = "neuralinternet"

# Thread pool
executor = ThreadPoolExecutor(max_workers=4)

# ─────────────────────── CONFIG VALIDATION & MERGE ───────────────────────
# Type definitions for config validation (allows unknown fields for flexibility)
CONFIG_TYPES = {
    "attestation_layer": {
        "url": str, "auth_token": str,
        "connect_timeout_sec": (int, float), "read_timeout_sec": (int, float),
        "retry_max": int, "retry_base_sec": (int, float), "retry_max_sec": (int, float),
        "retry_statuses": str, "cache_ttl_sec": int,
    },
    "chain": {
        "network": str, "rpc_http": str, "rpc_wss": str,
        "request_timeout_sec": (int, float), "block_hash_cache": int,
    },
    "miner": {
        "block_interval_validator_update": int, "block_interval_specs_update": int,
        "block_interval_sync_status": int, "block_interval_allocation_check": int,
        "wallet_name": str, "hotkey_name": str, "run_root_path": str, "poll_sec": (int, float),
    },
    "validator": {
        "block_interval_hardware_info": int, "block_interval_miner_check": int,
        "block_interval_sync_status": int, "block_interval_token_refresh": int,
        "pubsub_timeout_sec": (int, float), "ssh_timeout_sec": (int, float),
        "allocation_timeout_sec": (int, float), "deallocation_timeout_sec": (int, float),
        "allocation_max_retries": int, "deallocation_max_retries": int,
        "retry_backoff_sec": (int, float), "server_ip": str, "server_port": str, "pull_interval": int,
    },
    "pog": {
        "exec_every_blocks": int, "finalized_cache_ttl_sec": int, "rpc_backoff_sec": (int, float),
        "allocation_grace_sec": int, "pog_interval_blocks": int, "batch_size": int,
        "c_open_rows": int, "c_open_cols": int, "n_cap": int,
        "anchor_kind": str, "buffer_micro": (int, float), "buffer_heavy": (int, float),
        "stream_micro_priority": int, "stream_heavy_priority": int,
        "cc_attestation_enabled": bool, "cc_attestation_mode": str,
        "miner_script_path": str, "time_tolerance": int, "submatrix_size": int,
        "buffer_factor": (int, float), "spot_per_gpu": int, "hash_algorithm": str,
        "pog_retry_limit": int, "pog_retry_interval": int, "max_workers": int, "max_random_delay": int,
    },
    # Complex sections with dynamic keys - allow any dict structure
    "delta_model": dict,
    "gpu_performance": dict,
    "subnet_config": dict,
}

def deep_merge(base: dict, updates: dict) -> dict:
    """Recursively merge updates into base dict. Only updates provided fields."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def validate_config_types(payload: dict) -> list:
    """Validate that payload values match expected types. Returns list of errors."""
    errors = []
    for section, fields in payload.items():
        if section not in CONFIG_TYPES:
            # Allow unknown sections (for flexibility with existing config)
            continue
        if not isinstance(fields, dict):
            continue
        type_spec = CONFIG_TYPES[section]
        if not isinstance(type_spec, dict):
            continue
        for field, value in fields.items():
            if field not in type_spec:
                continue  # Allow unknown fields within known sections
            expected = type_spec[field]
            if isinstance(expected, tuple):
                if not isinstance(value, expected):
                    errors.append(f"{section}.{field}: expected {expected}, got {type(value).__name__}")
            else:
                if not isinstance(value, expected):
                    errors.append(f"{section}.{field}: expected {expected.__name__}, got {type(value).__name__}")
    return errors

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
    Pull W&B runs for miners and map their specs to hotkeys.
    Only considers runs that are currently running (state=='running').
    If a UID's hotkey has no active run, leave details as {}.
    """
    db_specs_dict: Dict[int, Dict[str, Any]] = {}
    project_path = f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_NAME}"

    # --- 1) Build hotkey -> specs map from W&B (only state=='running') ---
    wandb_hotkey_map: Dict[str, Dict[str, Any]] = {}
    try:
        runs = api.runs(project_path)
        for run in runs:
            # Only accept active runs
            if getattr(run, "state", None) != "running":
                continue

            run_config = getattr(run, "config", None)
            if not run_config or not isinstance(run_config, dict):
                continue
            if run_config.get("role") != "miner":
                continue

            details = run_config.get("specs")
            if not isinstance(details, dict) or not details:
                continue

            run_hotkey = run_config.get("hotkey")
            if isinstance(run_hotkey, str) and run_hotkey:
                wandb_hotkey_map[run_hotkey] = details

    except Exception as e:
        print(f"An error occurred while fetching runs from wandb: {e}")

    # --- 2) Map to your state.stats ---
    for uid, stat_data in state.stats.items():
        if not isinstance(stat_data, dict):
            db_specs_dict[uid] = {
                "hotkey": None,
                "details": {},
                "stats": stat_data,
            }
            continue

        st_hotkey = stat_data.get("hotkey")
        if isinstance(st_hotkey, str) and st_hotkey in wandb_hotkey_map:
            final_details = copy.deepcopy(wandb_hotkey_map[st_hotkey])
        else:
            final_details = {}

        db_specs_dict[uid] = {
            "hotkey": st_hotkey,
            "details": final_details,
            "stats": stat_data,
        }

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
            hotkey = entry
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

def get_metagraph_test():
    subtensor = bt.subtensor(network="test")
    # Fetch and sync the metagraph for subnet 15 on testnet
    metagraph = subtensor.metagraph(netuid=15)
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
                state.price_cache["tao_price"] = data["bittensor"]["usd"]
            else:
                print(f"⚠️ Failed to fetch TAO price: {response.status_code}")
    except Exception as e:
        print(f"❌ Error fetching TAO price: {e}")

def fetch_active_wandb_runs(api, entity: str, project: str) -> Dict[str, str]:
    """
    Return a mapping of hotkey -> '<entity>/<project>/runs/<run_id>' for ACTIVE (state=='running')
    miner runs in the given W&B project.
    """
    project_path = f"{entity}/{project}"
    mapping: Dict[str, str] = {}
    try:
        runs = api.runs(project_path)
        for run in runs:
            # Only active runs
            if getattr(run, "state", None) != "running":
                continue
            cfg = getattr(run, "config", None)
            if not isinstance(cfg, dict):
                continue
            if cfg.get("role") != "miner":
                continue
            hotkey = cfg.get("hotkey")
            if not isinstance(hotkey, str) or not hotkey:
                continue
            run_id = getattr(run, "id", "")
            mapping[hotkey] = f"{project_path}/runs/{run_id}"
    except Exception as e:
        print(f"An error occurred while fetching active runs from wandb ({project_path}): {e}")
    return mapping

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

    api = wandb.Api()

    while True:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, api.flush)

            print("[DEBUG] Starting metagraph sync in background task ...")
            metagraph = await loop.run_in_executor(executor, get_metagraph)

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

            state.metagraph_cache = await loop.run_in_executor(executor, build_metagraph_cache, metagraph)

            metagraph_test = await loop.run_in_executor(executor, get_metagraph_test)

            def build_metagraph_test_cache(m):
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

            state.metagraph_test_cache = await loop.run_in_executor(executor, build_metagraph_test_cache, metagraph_test)

            alpha_price, alpha_emission = await loop.run_in_executor(executor, get_subnet_alpha_price)
            await loop.run_in_executor(executor, api.flush)

            if alpha_price is not None and alpha_emission is not None:
                state.subnet_cache = {
                    "alpha_price": float(alpha_price),
                    "alpha_emission": float(alpha_emission),
                }
            else:
                state.subnet_cache = {}

            new_stats = await loop.run_in_executor(
                executor,
                fetch_validator_stats,
                api,
                validator_run_path2
            )
            state.stats = new_stats

            hotkeys = metagraph.hotkeys
            hardware_specs = await loop.run_in_executor(
                executor,
                fetch_hardware_specs,
                api,
                hotkeys
            )
            state.hardware_specs_cache = hardware_specs
            state.hotkeys_cache = hotkeys

            hotkeys_test = metagraph_test.hotkeys
            hardware_specs_test = await loop.run_in_executor(
                executor,
                fetch_hardware_specs,
                api,
                hotkeys_test
            )
            state.hardware_specs_test_cache = hardware_specs_test

            allocated = await loop.run_in_executor(
                executor,
                get_allocated_hotkeys,
                api,
                validator_run_path
            )
            state.allocated_hotkeys_cache = allocated

            penalized = await loop.run_in_executor(
                executor,
                get_penalized_hotkeys_id,
                api,
                validator_run_path2
            )
            state.penalized_hotkeys_cache = penalized

            main_entity = os.getenv("PUBLIC_WANDB_ENTITY", "neuralinternet")
            main_project = os.getenv("PUBLIC_WANDB_NAME", "opencompute")
            test_entity = os.getenv("PUBLIC_WANDB_TEST_ENTITY", os.getenv("PUBLIC_WANDB_ENTITY", "neuralinternet"))
            test_project = os.getenv("PUBLIC_WANDB_TEST_NAME", os.getenv("PUBLIC_WANDB_NAME", "opencompute"))

            active_main = await loop.run_in_executor(
                executor, fetch_active_wandb_runs, api, main_entity, main_project
            )
            active_test = await loop.run_in_executor(
                executor, fetch_active_wandb_runs, api, test_entity, test_project
            )

            state.active_wandb_runs["Mainnet"] = active_main
            state.active_wandb_runs["Testnet"] = active_test

            asyncio.create_task(fetch_tao_price())

        except Exception as e:
            print(f"An error occurred during periodic sync: {e}")

        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    wandb.login(key=api_key)
    asyncio.create_task(sync_data_periodically())

@app.get("/keys")
async def get_keys() -> Dict[str, List[str]]:
    return {"keys": state.hotkeys_cache}

@app.get("/specs")
async def get_specs() -> Dict[str, Dict[int, Dict[str, Any]]]:
    return {"specs": state.hardware_specs_cache}

@app.get("/allocated_keys")
async def get_allocated_keys() -> Dict[str, List[str]]:
    return {"allocated_keys": state.allocated_hotkeys_cache}

@app.get("/penalized_keys")
async def get_penalized_keys() -> Dict[str, List[str]]:
    return {"penalized_keys": state.penalized_hotkeys_cache}

@app.get("/subnet")
async def get_subnet_data() -> Dict[str, Any]:
    return {"subnet": state.subnet_cache}

@app.get("/metagraph")
async def get_metagraph_data() -> Dict[str, Any]:
    if not state.metagraph_cache:
        return {"error": "Metagraph data not available. Try again later."}
    return {"metagraph": state.metagraph_cache}

@app.get("/metagraph_test")
async def get_metagraph__test_data() -> Dict[str, Any]:
    if not state.metagraph_test_cache:
        return {"error": "Metagraph data not available. Try again later."}
    return {"metagraph": state.metagraph_test_cache}

@app.get("/price")
async def get_tao_price() -> Dict[str, Any]:
    return {"tao_price": state.price_cache.get("tao_price", "N/A")}

@app.get("/config")
async def get_config() -> Dict[str, Any]:
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
    """
    Update config with MERGE strategy for robustness.
    - Only updates provided fields, preserves others
    - Creates backup before writing
    - Uses atomic write (temp file + rename)
    - Validates types before any write
    """
    if not ENABLE_CONFIG_WRITE:
        raise HTTPException(status_code=403, detail="Config writing disabled by server policy")
    if x_admin_key != ADMIN_KEY or not ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    # 1. Validate payload types
    errors = validate_config_types(payload)
    if errors:
        raise HTTPException(status_code=400, detail=f"Invalid config: {'; '.join(errors)}")

    try:
        # 2. Read current config
        try:
            with open("config.yaml", "r") as f:
                current_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            current_config = {}

        # 3. Backup current config before any modification
        if os.path.exists("config.yaml"):
            shutil.copy("config.yaml", "config.yaml.bak")

        # 4. Merge updates into current config
        new_config = deep_merge(current_config, payload)

        # 5. Atomic write: write to temp file, then rename
        with open("config.yaml.tmp", "w") as f:
            yaml.safe_dump(new_config, f, sort_keys=False)
        os.replace("config.yaml.tmp", "config.yaml")

    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists("config.yaml.tmp"):
            try:
                os.remove("config.yaml.tmp")
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")

    return {"status": "ok"}

from routes.rental_server import router as rental_router
app.include_router(rental_router)

# server.py (where you included rental_router)
from routes.benchmark import router as benchmark_router
app.include_router(benchmark_router)

# To run the server:
# uvicorn server:app --reload --host 0.0.0.0 --port 8316
# or:
# pm2 start uvicorn --interpreter python3 --name opencompute_server -- --host 0.0.0.0 --port 8000 server:app