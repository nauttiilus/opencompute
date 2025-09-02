# opencompute/state.py

hardware_specs_cache: dict[int, dict] = {}
hardware_specs_test_cache: dict[int, dict] = {}
allocated_hotkeys_cache: list[str] = []
penalized_hotkeys_cache: list[str] = []
hotkeys_cache: list[str] = []
metagraph_cache: dict = {}
metagraph_test_cache: dict = {}
subnet_cache: dict = {}
price_cache: dict = {}
stats: dict[int, dict] = {}
active_wandb_runs = {}