import wandb
import bittensor as bt
from typing import Literal

def update_allocations_in_wandb(
    wandb_instance,
    wandb_run_path: str,
    hotkey: str,
    action: Literal["add", "remove"]
):
    """
    Update allocated hotkeys and reflect the changes in the stats (allocated flag) in the WandB config.

    Args:
        wandb_run_path (str): WandB run path (e.g. 'neuralinternet/opencompute/xyz123')
        hotkey (str): The hotkey to add/remove
        action (str): "add" or "remove"
    """
    try:
        wandb_instance.api.flush()

        api = wandb.Api()
        api.flush()
        run = api.run(wandb_run_path)
        config = run.config

        # Step 1: Pull data
        allocated = config.get("allocated_hotkeys", [])
        stats = config.get("stats", {})
        penalized = config.get("penalized_hotkeys_checklist", [])

        # Normalize hotkeys
        hotkey = hotkey.strip()
        allocated = [str(hk).strip() for hk in allocated]

        # Step 2: Update allocated_hotkeys
        if action == "add" and hotkey not in allocated:
            allocated.append(hotkey)
        elif action == "remove":
            allocated = [hk for hk in allocated if hk != hotkey]

        hotkey_set = set(allocated)

        # Step 3: Update 'allocated' flag in stats
        for uid, data in stats.items():
            data["allocated"] = data.get("hotkey") in hotkey_set

        # Step 4: Flatten gpu_specs
        flat_stats = {}
        for uid, data in stats.items():
            flat_stats[uid] = {
                "uid":               uid,
                "hotkey":            data.get("hotkey"),
                "gpu_name":          data.get("gpu_name"),
                "gpu_num":           data.get("gpu_num"),
                "score":             data.get("score"),
                "allocated":         data.get("allocated"),
                "own_score":         data.get("own_score"),
                "reliability_score": data.get("reliability_score"),
                "created_at":        data.get("created_at"),
            }

        # Step 5: Push to WandB config
        wandb_instance.run.config.update({
            "allocated_hotkeys": allocated,
            "penalized_hotkeys_checklist": penalized,
            "stats": flat_stats
        }, allow_val_change=True)
        wandb_instance.api.flush()

        print(f"[WandB] Updated allocation: {action} '{hotkey}'")

    except Exception as e:
        print(f"[WandB] Failed to update allocations ({action}): {e}")
