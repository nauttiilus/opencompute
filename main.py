import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import matplotlib.pyplot as plt
import base64
import json, requests, threading
import time
import plotly.graph_objects as go  # type: ignore
import plotly.express as px        # type: ignore
import plotly.colors as pc

# ────────────────────────── ENV & GLOBALS ────────────────────────────
load_dotenv()

def get_config(key, default=None):
    """
    Priority:
    1. Local .env (os.getenv)
    2. Streamlit Cloud secrets (st.secrets)
    3. Default value
    """
    # 1. Check .env / local environment
    env_value = os.getenv(key)
    if env_value not in (None, ""):
        return env_value

    # 2. Check Streamlit secrets (only if they exist)
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        return st.secrets.get(key, default)

    # 3. Use default
    return default

# Config values
NI_ADMIN_USERNAME = get_config("NI_ADMIN_USERNAME", "ni-admin")
NI_ADMIN_PASSWORD = get_config("NI_ADMIN_PASSWORD", "")
NI_MASTER_USERNAME = get_config("NI_MASTER_USERNAME", "ni-master")
NI_MASTER_PASSWORD = get_config("NI_MASTER_PASSWORD", "")

ADMIN_PASSWORD = get_config("ADMIN_PASSWORD", "")  # kept for compatibility (unused in new auth flow)
ADMIN_KEY      = get_config("ADMIN_KEY", "")
SERVER_IP      = get_config("SERVER_IP", "65.108.33.88")
SERVER_PORT    = get_config("SERVER_PORT", "8000")

SERVER_URL     = f"http://{SERVER_IP}:{SERVER_PORT}"
BLUE           = "#1976D2"

st.set_page_config(page_title="NI-Compute", layout="wide", page_icon="Icon_White_crop.png")

# ─────────────────────────── CSS ─────────────────────────────
st.markdown(f"""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
  img[data-testid="stLogo"] {{
    height: 8rem !important;
    width: auto !important;
  }}
  .icon {{
    font-family: 'Material Icons';
    font-size: 18px;
    vertical-align: middle;
    margin-right: 5px;
    color: {BLUE} !important;
  }}
  a {{
    color: {BLUE} !important;
    text-decoration: none;
            font-weight: light;
  }}
  a:hover {{
    text-decoration: underline;
  }}
  .stProgress > div > div > div > div {{
    background-color: {BLUE} !important;
  }}
</style>
""", unsafe_allow_html=True)

st.logo("Neural_Internet_White_crop.png", size="large")

# ───────────────────────── SERVER CALL ────────────────────────
def get_data_from_server(endpoint: str):
    try:
        r = requests.get(f"{SERVER_URL}/{endpoint}", timeout=15)
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        st.error(f"Error fetching {endpoint}: {e}")
        return {}

# ─────────────────────── SIDEBAR LOGIN (username + password) ────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "role" not in st.session_state:
    st.session_state.role = None  # "admin" | "master" | None

def _login_cb():
    """Called when the user presses Enter in the password box."""
    user = st.session_state.get("admin_user", "").strip()
    pwd  = st.session_state.get("admin_pwd", "")

    # Master has full access
    if user == NI_MASTER_USERNAME and pwd == NI_MASTER_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.role = "master"
    # Admin has rentals + benchmark mainnet
    elif user == NI_ADMIN_USERNAME and pwd == NI_ADMIN_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.role = "admin"
    else:
        st.session_state.authenticated = False
        st.session_state.role = None
        st.sidebar.error("❌ Invalid username or password")

with st.sidebar:
    if not st.session_state.authenticated:
        # username first, then password; pressing Enter in password triggers login
        st.markdown("##### Admin Login")  # smaller heading (H5 size)
        st.text_input("Username", key="admin_user")
        st.text_input(
            "Password",
            type="password",
            key="admin_pwd",
            on_change=_login_cb
        )
    else:
        st.caption(f"Logged in as **{st.session_state.role}**")
        if st.button("Logout"):
            # 1) log them out
            st.session_state.authenticated = False
            st.session_state.role = None
            # 2) clear the saved creds so the boxes are empty on next show
            for k in ("admin_pwd", "admin_user"):
                if k in st.session_state:
                    del st.session_state[k]
            # 3) rerun so the login box immediately pops back
            st.rerun()

# ───────────────────────── HELPERS ───────────────────────────
def display_hardware_specs(specs_details, allocated_keys, penalized_keys, metagraph_data):
    column_headers = [
        "UID", "Hotkey", "Incentive", "GPU Name", "GPU Capacity (GiB)", "GPU Count",
        "CPU Count", "RAM (GiB)", "Disk Space (GiB)", "Status", "PoG Status", "Penalized"
    ]
    incentives = dict(zip(
        metagraph_data.get("uids", []),
        metagraph_data.get("incentive", [])
    ))
    table_data = []

    for idx in sorted(specs_details.keys(), key=int):
        item = specs_details.get(idx, {}) or {}
        stats = item.get("stats", {}) or {}
        det   = item.get("details", {}) or {}
        hotkey = item.get("hotkey", "unknown")

        # Raw values from stats
        raw_gpu_name = stats.get("gpu_name")
        raw_gpu_num = stats.get("gpu_num")

        # Display values (fallback to "N/A")
        gpu_name = raw_gpu_name if raw_gpu_name else "N/A"
        gpu_num = raw_gpu_num if isinstance(raw_gpu_num, (int, float)) and raw_gpu_num > 0 else "N/A"

        cap = (det.get("gpu", {}) or {}).get("capacity", 0) / 1024
        cpu = (det.get("cpu", {}) or {}).get("count", 0)
        ram = (det.get("ram", {}) or {}).get("available", 0) / (1024**3)
        disk = (det.get("hard_disk", {}) or {}).get("free", 0) / (1024**3)

        status = "Res." if hotkey in allocated_keys else "Avail."
        pog = "Pass" if raw_gpu_name and raw_gpu_num else "Fail"
        pen = "Yes" if hotkey in penalized_keys else "No"

        table_data.append([
            int(idx),
            hotkey,
            round(incentives.get(int(idx), 0), 6),
            gpu_name,
            round(cap, 2),
            gpu_num,
            cpu, round(ram, 2), round(disk, 2),
            status, pog, pen
        ])

    df = pd.DataFrame(table_data, columns=column_headers).set_index("UID")
    st.dataframe(df.style.format({
        "GPU Capacity (GiB)": "{:.2f}",
        "RAM (GiB)": "{:.2f}",
        "Disk Space (GiB)": "{:.2f}",
        "Incentive": "{:.6f}",
    }), use_container_width=True, height=1000)

def plot_incentive_vs_uid(metagraph_data):
    if not metagraph_data or "uids" not in metagraph_data or "incentive" not in metagraph_data:
        st.error("Incomplete metagraph data. Try again later.")
        return
    df = pd.DataFrame({
        "UID": metagraph_data["uids"],
        "Incentive": metagraph_data["incentive"]
    }).sort_values("Incentive", ascending=True)
    fig = go.Figure(go.Scatter(
        x=list(range(len(df))),
        y=df["Incentive"],
        mode="markers",
        marker=dict(size=8, color=df["Incentive"], colorscale="Blues", showscale=True)
    ))
    fig.update_layout(
        title="Incentive Distribution",
        xaxis=dict(title="Miners", showticklabels=False),
        yaxis=dict(title="Incentive", gridcolor="gray"),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────── PAGES ─────────────────────────────
def dashboard():
    st.title("Dashboard – NI Compute")
    with st.spinner("Loading..."):
        specs = get_data_from_server("specs").get("specs", {})
        alloc = get_data_from_server("allocated_keys").get("allocated_keys", [])
        pen   = get_data_from_server("penalized_keys").get("penalized_keys", [])
        meta  = get_data_from_server("metagraph").get("metagraph", {})
    plot_incentive_vs_uid(meta)
    display_hardware_specs(specs, alloc, pen, meta)

def metagraph():
    st.title("Metagraph Data")
    with st.spinner("Loading..."):
        data = get_data_from_server("metagraph").get("metagraph", {})
    if not data or "uids" not in data:
        st.error("Metagraph data not available."); return
    uids   = data["uids"]
    hot    = data["hotkeys"]
    act    = data["active"]
    stake  = data["stake"]
    trust  = data["trust"]
    v_trust= data["validator_trust"]
    v_perm = data["validator_permit"]
    axons  = data["axons"]
    miner_ver = {}
    val_ver   = {}
    rows = []
    for i, uid in enumerate(uids):
        ax = axons[i] if i < len(axons) else {}
        ip = ax.get("ip", "N/A")
        port = ax.get("port", "N/A")
        ver  = ax.get("version", "N/A")
        if v_trust[i] == 0:
            miner_ver[ver] = miner_ver.get(ver, 0) + 1
        else:
            val_ver[ver] = val_ver.get(ver, 0) + 1
        rows.append([
            uid, hot[i], act[i],
            round(stake[i],6), round(trust[i],6),
            v_perm[i], round(v_trust[i],6),
            ip, port, ver
        ])
    cols = ['UID','Hotkey','Active','Stake','Trust','V_Permit','V_Trust','IP','Port','Version']
    df = pd.DataFrame(rows, columns=cols)
    st.dataframe(df.style.format({
        "Stake":"{:.6f}", "Trust":"{:.6f}", "V_Trust":"{:.6f}"
    }), use_container_width=True, height=800)
    vtot = sum(val_ver.values()); mtot = sum(miner_ver.values())
    col1, col2 = st.columns(2)
    val_df = pd.DataFrame([
        {"Version":k,"Count":c,"Percentage":c/vtot*100}
        for k,c in val_ver.items()
    ])
    min_df = pd.DataFrame([
        {"Version":k,"Count":c,"Percentage":c/mtot*100}
        for k,c in miner_ver.items()
    ])
    col2.write("### Validator Version Summary")
    col2.write(f"Total Validators: {vtot}")
    col2.dataframe(val_df.style.format({"Percentage":"{:.2f}%"}), use_container_width=True)
    col1.write("### Miner Version Summary")
    col1.write(f"Total Miners: {mtot}")
    col1.dataframe(min_df.style.format({"Percentage":"{:.2f}%"}), use_container_width=True)

# Define supported GPU models
SUPPORTED_GPUS = {
    "NVIDIA B200": "B200",
    "NVIDIA H200": "H200",
    "NVIDIA H100 80GB HBM3": "H100 80GB HBM3",
    "NVIDIA H100": "H100 80GB PCIE",           # generic match for PCIe variant
    "NVIDIA H100 PCIe": "H100 80GB PCIE",
    "NVIDIA H100 NVL": "H100 NVL",
    "NVIDIA A100-SXM4-80GB": "A100 80GB SXM4",
    "NVIDIA A100-SXM4-40GB": "A100 40GB SXM4",
    "NVIDIA A100 80GB PCIe": "A100 80GB PCIE",
    "NVIDIA L40S": "L40S",
    "NVIDIA L40": "L40",
    "NVIDIA A40": "A40",
    "NVIDIA GeForce RTX 5090": "RTX 5090",
    "NVIDIA RTX 6000 Ada Generation": "RTX 6000 Ada",
    "NVIDIA RTX A6000": "RTX A6000",
    "NVIDIA RTX A5000": "RTX A5000",
    "NVIDIA RTX A4500": "RTX A4500",
    "NVIDIA RTX 4000 Ada Generation": "RTX 4000 Ada",
    "NVIDIA RTX A4000": "RTX A4000",
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA GeForce RTX 3090": "RTX 3090",
    "NVIDIA L4": "L4",
}

OTHER_GPU_LABEL = "Other GPUs"

def stats():
    st.title("Network Statistics")
    with st.spinner("Loading..."):
        alloc = get_data_from_server("allocated_keys").get("allocated_keys", [])
        specs = get_data_from_server("specs").get("specs", {})
        cfg   = get_data_from_server("config").get("config", {})
    if not cfg:
        st.error("No config available."); return

    # ── GPU Count Distribution (from network) ──
    gpu_counts = {}
    rented_gpu = {}
    total = 0
    for itm in specs.values():
        sd = itm.get("stats", {}) or {}
        name = sd.get("gpu_name", "Unknown GPU")
        raw_cnt = sd.get("gpu_num")
        cnt = int(raw_cnt) if isinstance(raw_cnt, (int, float)) else 0
        status = "Res." if itm.get("hotkey", "") in alloc else "Avail."
        lbl = SUPPORTED_GPUS.get(name, OTHER_GPU_LABEL)
        gpu_counts[lbl] = gpu_counts.get(lbl, 0) + cnt
        total += cnt
        if status == "Res.":
            rented_gpu[lbl] = rented_gpu.get(lbl, 0) + cnt

    gpu_data = pd.DataFrame([
        {"GPU Model": g, "Count": c,
         "Rented": rented_gpu.get(g, 0),
         "Percentage": (c / total * 100) if total else 0}
        for g, c in gpu_counts.items()
    ]).sort_values("Count", ascending=False)

    # ── GPU Weights from Config ONLY ──
    gpu_weights_cfg = (cfg.get("subnet_config", {}) or {}).get("gpu_weights", {})
    total_miner_emission = float((cfg.get("subnet_config", {}) or {}).get("total_miner_emission", 0.0))

    total_weight = sum(gpu_weights_cfg.values())
    if total_weight == 0:
        total_weight = 1  # avoid div by zero

    # Map raw GPU names from config to display labels
    weight_rows = []
    for raw_name, w in gpu_weights_cfg.items():
        label = SUPPORTED_GPUS.get(raw_name, raw_name)  # fallback to raw if not mapped
        weight_pct = (w / total_weight * 100)
        emission_pct = (w / total_weight * total_miner_emission * 100)
        weight_rows.append({
            "GPU Model": label,
            "Weight %": weight_pct,
            "Emission %": emission_pct
        })

    weight_df = pd.DataFrame(weight_rows).sort_values("Weight %", ascending=False)

    # ── Use distinct qualitative colors ──
    distinct_colors = px.colors.qualitative.Set3

    # ── Layout: now config pie is Column 1 ──
    c1, c2 = st.columns([1.5, 1.5])

    # Left column → GPU Weight & Emission (from config) + Rentals
    with c1:
        st.markdown("#### GPU Emission Share Distribution")
        fig1 = px.pie(weight_df, names="GPU Model", values="Weight %",
                        color_discrete_sequence=distinct_colors)
        fig1.update_layout(width=800, height=600, showlegend=True,
                           legend=dict(orientation="h", yanchor="bottom", y=-0.5,
                                       xanchor="center", x=0.5))
        st.plotly_chart(fig1, use_container_width=True)
        st.dataframe(weight_df.style.format({
            "Weight %": "{:.2f}%",
            "Emission %": "{:.2f}%"
        }), use_container_width=True)

        # GPU Rentals moved here
        if not gpu_data.empty:
            st.markdown("#### GPU Rentals")
            for g in gpu_counts:
                if g == OTHER_GPU_LABEL:
                    continue
                tot = gpu_counts[g]
                ren = rented_gpu.get(g, 0)
                pct = ren / tot if tot else 0
                st.text(f"{g}: {ren}/{tot} rented")
                st.progress(pct)
        else:
            st.info("No GPU count data available.")

    # Right column → GPU Count Distribution
    with c2:
        st.markdown("#### GPU Live Distribution (Count)")
        if not gpu_data.empty:
            fig2 = px.pie(gpu_data, names="GPU Model", values="Count",
                        color_discrete_sequence=distinct_colors)
            fig2.update_layout(width=800, height=600, showlegend=True,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.5,
                                           xanchor="center", x=0.5))
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(gpu_data.style.format({"Percentage": "{:.2f}%"}),
                         use_container_width=True)
        else:
            st.info("No GPU count data available.")

def search():
    """Search Tool for Hotkeys"""
    st.title("Search Tool")

    # Fetch all data
    with st.spinner("Fetching data..."):
        hotkeys_response = get_data_from_server("keys")
        specs_response = get_data_from_server("specs")
        metagraph_response = get_data_from_server("metagraph")
        allocated_keys_response = get_data_from_server("allocated_keys")
        penalized_keys_response = get_data_from_server("penalized_keys")
        subnet_response = get_data_from_server("subnet")
        price_response = get_data_from_server("price")

    # Extract responses
    specs_details = specs_response.get("specs", {})
    metagraph_data = metagraph_response.get("metagraph", {})
    allocated_keys = allocated_keys_response.get("allocated_keys", [])
    penalized_keys = penalized_keys_response.get("penalized_keys", [])
    subnet_data = subnet_response.get("subnet", {})
    tao_price_usd = price_response.get("tao_price", 0)

    # Hotkey search with autocomplete
    hotkeys = metagraph_data.get("hotkeys", [])

    # Single-column layout for clarity
    col, _ = st.columns([1, 1])  # Use only the first column (50% width)
    hotkey_input = col.selectbox("Select or enter a hotkey to search:", [""] + hotkeys)

    if hotkey_input:
        if hotkey_input in hotkeys:
            uid = metagraph_data["hotkeys"].index(hotkey_input)
            hardware_info = specs_details.get(str(uid), {})  # Convert uid to string

            # Single-column layout for clarity
            col, _ = st.columns([1, 1])  # Use only the first column (50% width)

            # 🚨 **Handle Missing GPU Data Case**
            if not hardware_info or "stats" not in hardware_info or not hardware_info.get("stats", {}).get("gpu_name"):
                col.error("⚠️ This hotkey is registered but inactive. It did not pass Proof-of-GPU (PoG) verification.")
                return  # Stop execution to avoid further errors

            # ✅ **Display Hardware Information**
            col.subheader("Hardware Information")
            stats = hardware_info.get("stats", {})
            details = hardware_info.get("details", {})

            gpu_name = stats.get("gpu_name", "N/A")
            gpu_num = stats.get("gpu_num", 0)
            cpu_specs = details.get("cpu", {})
            gpu_miner = details.get("gpu", {})
            ram_specs = details.get("ram", {})
            disk_specs = details.get("hard_disk", {})

            hardware_table = {
                "GPU Name": gpu_name,
                "GPU Count": gpu_num,
                "Total GPU Capacity (GiB)": gpu_miner.get("capacity", 0) / 1024 if gpu_miner.get("capacity") else "N/A",
                "CPU Count": cpu_specs.get("count", "N/A"),
                "RAM Available (GiB)": ram_specs.get("available", 0) / (1024.0**3),
                "Disk Space Free (GiB)": disk_specs.get("free", 0) / (1024.0**3),
            }

            df_hardware = pd.DataFrame.from_dict(hardware_table, orient="index", columns=["Value"])
            df_hardware["Value"] = df_hardware["Value"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            col.dataframe(df_hardware, use_container_width=True)

            # ✅ **Display Metagraph Information**
            col.subheader("Metagraph Information")
            axon_info = metagraph_data["axons"][uid]
            metagraph_info = {
                "UID": uid,
                "Incentive": metagraph_data["incentive"][uid],
                "Stake": metagraph_data["stake"][uid],
                "Trust": metagraph_data["trust"][uid],
                "Consensus": metagraph_data["consensus"][uid],
                "Validator Trust": metagraph_data["validator_trust"][uid],
                "Emission": metagraph_data["emission"][uid],
                "Dividends": metagraph_data["dividends"][uid],
                "Axon IP": axon_info.get("ip", "N/A"),
                "Axon Port": axon_info.get("port", "N/A"),
                "Version": axon_info.get("version", "N/A"),
            }

            df_metagraph = pd.DataFrame.from_dict(metagraph_info, orient="index", columns=["Value"])
            for key in ["Incentive", "Stake", "Trust", "Consensus", "Validator Trust", "Emission", "Dividends"]:
                df_metagraph.at[key, "Value"] = f"{df_metagraph.at[key, 'Value']:.6f}"
            col.dataframe(df_metagraph, use_container_width=True)

            # ✅ **Revenue Calculation**
            alpha_price = subnet_data["alpha_price"]
            alpha_per_day = metagraph_data["emission"][uid] * 20
            tao_per_day = alpha_per_day * alpha_price
            revenue_per_day_usd = alpha_price * alpha_per_day * tao_price_usd

            # ✅ **Display Scoring Metrics**
            col.subheader("Scoring Metrics")
            scoring_metrics = {
                "PoG Status": "Pass" if gpu_name not in (None, "", "N/A") else "Fail",
                "Penalized": "Yes" if hotkey_input in penalized_keys else "No",
                "Rented (Allocated)": "Yes" if hotkey_input in allocated_keys else "No",
                "Performance Score": stats.get("score", "N/A"),
                "Reliability Score": stats.get("reliability_score", "N/A"),
                "Revenue λ/day": alpha_per_day,
                "Revenue τ/day": tao_per_day,
                "Revenue $/day": revenue_per_day_usd,
            }

            df_scoring = pd.DataFrame.from_dict(scoring_metrics, orient="index", columns=["Value"])
            df_scoring["Value"] = df_scoring["Value"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            col.dataframe(df_scoring, use_container_width=True)

        else:
            st.error("❌ Hotkey not found. Please try again.")

def benchmark():
    import time  # only needed for the 4s message pause

    st.title("Benchmark Tool")

    st.markdown(
        "This tool runs a controlled benchmark on a selected miner hotkey to verify GPU performance, "
        "integrity, and configuration in a process closely aligned with the network’s validation routine."
    )

    # ── session flags ─────────────────────────────────────────────
    if "bench_running" not in st.session_state:
        st.session_state.bench_running = False
    if "bench_hotkey" not in st.session_state:
        st.session_state.bench_hotkey = None
    if "bench_network" not in st.session_state:
        st.session_state.bench_network = "Testnet"

    # Allow admin or master to operate (including Mainnet). Others disabled.
    disabled_flag = (st.session_state.role not in ("admin", "master")) or st.session_state.bench_running

    net_cols = st.columns([1, 3])
    with net_cols[0]:
        st.radio(
            "Network",
            ["Testnet", "Mainnet"],
            key="bench_network",   # bind directly to session_state
            horizontal=True,
            help="Choose which metagraph to list hotkeys from (only available for admins).",
            disabled=disabled_flag,
        )

        # Warning directly BELOW the radio when Mainnet is selected
        if st.session_state.bench_network == "Mainnet":
            st.warning(
                "Executing the benchmark tool for hotkeys on **Mainnet** may interfere with the "
                "**validation routine** and reduce miner performance if run frequently. "
                "Use sparingly and only when necessary.",
                icon="⚠️",
            )

    # ── load metagraph depending on selected network ─────────────
    meta_endpoint = "metagraph" if st.session_state.bench_network == "Mainnet" else "metagraph_test"
    meta = get_data_from_server(meta_endpoint).get("metagraph", {})
    hotkeys = meta.get("hotkeys", [])

    # ── controls (Start button disabled while running) ───────────
    hotkey = st.selectbox("Target hotkey", hotkeys)
    start = st.button("Start Benchmark", type="primary", disabled=st.session_state.bench_running)

    st.markdown("---")

    console = st.empty()
    status_box = st.empty()   # placeholder; cleared at end
    spinner_ph = st.empty()

    # ---------- kick off ----------
    if start:
        if not hotkey:
            st.error("Please select a hotkey first.")
            return
        st.session_state.bench_running = True
        st.session_state.bench_hotkey = hotkey
        st.rerun()

    # ---------- active run ----------
    if st.session_state.bench_running:
        hotkey = st.session_state.bench_hotkey

        # spinner immediately
        spinner_ph.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
  <div style="
    width:18px;height:18px;border:3px solid rgba(0,0,0,0.1);
    border-top-color:#1976D2;border-radius:50%;
    animation:spin 0.8s linear infinite;"></div>
  <div><strong>Preparing benchmark…</strong></div>
</div>
<style>@keyframes spin { to { transform: rotate(360deg); } }</style>
        """, unsafe_allow_html=True)

        # 1) start job  (only modification: handle 429 nicely)
        try:
            resp = requests.post(
                f"{SERVER_URL}/benchmark/start",
                json={"hotkey": hotkey, "network": st.session_state.bench_network},
                headers={"X-Admin-Key": ADMIN_KEY},
                timeout=15
            )

            # Graceful rate-limit / queue-full handling
            if resp.status_code == 429:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
                detail = data.get("detail") or "System is busy. Please retry later."
                retry_after = resp.headers.get("Retry-After")

                spinner_ph.empty()
                st.session_state.bench_running = False  # let user click again
                if retry_after:
                    st.warning(f"⏳ {detail} (retry after ~{retry_after}s)")
                else:
                    st.warning(f"⏳ {detail}")
                st.info(
                    "Tips:\n"
                    "- Avoid rapid repeated clicks.\n"
                    "- If multiple jobs are queued, wait for capacity to free up."
                )
                time.sleep(4)  # show message for ~4s, then refresh
                st.rerun()
                return

            # Normal success/error path (unchanged)
            resp.raise_for_status()
            job_id = resp.json().get("job_id")
        except Exception as e:
            spinner_ph.empty()
            st.session_state.bench_running = False
            st.error(f"Failed to start benchmark: {e}")
            st.rerun()

        # 2) stream logs (SSE)
        log_text = ""
        first_event_seen = False
        error_seen = False
        last_error_msg = None
        end_status = None  # "done" | "error" | "cancelled"

        try:
            with requests.get(f"{SERVER_URL}/benchmark/stream/{job_id}", stream=True, timeout=900) as r:
                r.raise_for_status()

                buffer_event, buffer_data = None, None

                for raw in r.iter_lines(decode_unicode=True):
                    if raw is None:
                        continue
                    raw = raw.strip()

                    if not raw:
                        # dispatch a complete event
                        if buffer_event is not None and buffer_data is not None:
                            # first SSE event → update spinner text
                            if not first_event_seen:
                                first_event_seen = True
                                spinner_ph.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
  <div style="
    width:18px;height:18px;border:3px solid rgba(0,0,0,0.1);
    border-top-color:#1976D2;border-radius:50%;
    animation:spin 0.8s linear infinite;"></div>
  <div><strong>Benchmark in progress – live logs</strong></div>
</div>
<style>@keyframes spin { to { transform: rotate(360deg); } }</style>
                                """, unsafe_allow_html=True)

                            if buffer_event == "message":
                                try:
                                    evt = json.loads(buffer_data)
                                    lvl = evt.get("level", "info")
                                    msg = evt.get("message", "")
                                    payload = evt.get("payload")

                                    # classify
                                    if lvl == "error":
                                        error_seen = True
                                        last_error_msg = msg or "Unknown error"

                                    prefix = {"success": "[OK]", "error": "[ERR]", "warning": "[WARN]"}.get(lvl, "[..]")
                                    line = f"{prefix} {msg}"

                                    # compact payload preview
                                    if isinstance(payload, dict) and payload:
                                        interesting = (
                                            # general / IDs
                                            "hotkey", "uid", "wandb_active",

                                            # GPU identity
                                            "num_gpus", "gpu_count", "gpu_names", "reported_name", "identified_gpu",

                                            # benchmark metrics
                                            "vram", "size_fp16", "time_fp16", "size_fp32", "time_fp32",
                                            "fp16_tflops", "fp32_tflops",

                                            # PoG/merkle
                                            "n", "roots", "avg_gemm", "avg_gemm_s", "timing_ok", "bench_times",

                                            # W&B hardware specs
                                            "gpu", "cpu", "ram_GiB", "disk_free_GiB", "run_path"
                                        )
                                        compact = {k: payload[k] for k in interesting if k in payload}
                                        if "reported_name" not in compact and "gpu_names" in payload:
                                            try:
                                                if isinstance(payload["gpu_names"], list) and payload["gpu_names"]:
                                                    compact["reported_name"] = payload["gpu_names"][0]
                                            except Exception:
                                                pass
                                        if compact:
                                            line += "\n" + "`" + json.dumps(compact, ensure_ascii=False) + "`"

                                except Exception:
                                    line = buffer_data

                                log_text += ("\n" if log_text else "") + line
                                console.code(log_text, language="bash")

                            elif buffer_event == "end":
                                try:
                                    end_payload = json.loads(buffer_data)
                                    end_status = end_payload.get("status")
                                except Exception:
                                    end_status = None
                                break

                            elif buffer_event == "error":
                                error_seen = True
                                last_error_msg = (buffer_data or "").strip() or "Stream error"
                                line = f"[ERR] {last_error_msg}"
                                log_text += ("\n" if log_text else "") + line
                                console.code(log_text, language="bash")
                                break

                        # reset for next event
                        buffer_event, buffer_data = None, None
                        continue

                    # accumulate sse fields
                    if raw.startswith("event:"):
                        buffer_event = raw[len("event:"):].strip()
                    elif raw.startswith("data:"):
                        data_line = raw[len("data:"):].strip()
                        buffer_data = data_line if buffer_data is None else buffer_data + "\n" + data_line

        except Exception as e:
            spinner_ph.empty()
            st.session_state.bench_running = False
            st.error(f"Benchmark failed ❌\n\nStream error: {e}")
            st.info(
                "How to resolve:\n"
                "- Check server health and logs.\n"
                "- Verify ports and firewall rules.\n"
                "- Ensure the benchmark service is running.\n"
                "- Try again."
            )
            #st.rerun()

        # 3) end-of-run UI
        spinner_ph.empty()
        status_box.empty()

        if error_seen or (end_status == "error"):
            msg = last_error_msg or "An error occurred."
            fix = "How to resolve:\n"

            lm = (msg or "").lower()

            if ("cuda out of memory" in lm or
                "torch.outofmemoryerror" in lm or
                "out of memory" in lm):
                fix += "- Ensure there are no other processes using the GPU; additional load is not allowed and will be detected.\n" \
                    "- Stop any background GPU tasks (training, inference, etc.) before re-running.\n" \
                    "- Reboot the miner if memory does not free up after stopping processes.\n" 

            elif "gpu mismatch" in lm:
                fix += "- Ensure the reported GPU matches the actual hardware performance (no spoofing).\n" \
                    "- Stop any other GPU workloads and re-run; additional load is not allowed on this subnet.\n"

            elif "no gpus detected" in lm:
                fix += "- Verify that NVIDIA drivers and CUDA are properly installed and working.\n" \
                    "- Confirm that the container has GPU access (e.g., run `nvidia-smi`).\n" \
                    "- Reboot the miner if needed.\n"

            elif "verification failed" in lm or "pog" in lm:
                fix += "- Check GPU clocks and thermals to avoid instability.\n" \
                    "- Reboot the miner and re-run the benchmark.\n"

            elif "ssh" in lm or "novalidconnectionserror" in lm or "connection refused" in lm:
                fix += "- Confirm that the SSH port (set with `ssh.port`) is open (default = 4444).\n" \
                    "- Check network connectivity to the miner host/port.\n" \
                    "- Ensure the container is still running and not terminated by the system after creation.\n"

            elif "integrity" in lm or "hash" in lm:
                fix += "- Ensure Docker has root access and can read/write to the `/tmp` directory.\n" \
                    "- Do not attempt spoofing — it will be detected.\n"

            elif "wandb" in lm or "w&b" in lm:
                fix += "- Ensure the miner runs a Weights & Biases (W&B) client under [OpenCompute W&B](https://wandb.ai/neuralinternet/opencompute/) \n" \
                    "- Verify that `WANDB_API_KEY` is set in the `.env` file.\n" \
                    "- Confirm the run config includes `role='miner'`, the correct `hotkey`, and a valid `specs` dictionary.\n"

            elif ("busy" in lm or "allocator" in lm or "declined" in lm or "hotkey" in lm or
                  ("allocate" in lm and "tried to allocate" not in lm and "out of memory" not in lm)):
                fix += "- Check that the axon port (set with `axon.port`) is open and reachable from the network.\n" \
                    "- The miner may already be allocated or performing another test; wait and retry later.\n" \
                    "- Verify allocator availability before retrying.\n"

            else:
                fix += "- Check server logs for more details.\n" \
                    "- Verify miner health and configuration.\n" \
                    "- Re-run the benchmark.\n"

            st.error(f"Benchmark failed ❌\n\n{msg}")
            st.info(fix)

        elif end_status == "cancelled":
            st.warning("Benchmark was cancelled by user.")
        else:
            st.success("Benchmark complete ✅")
            st.info("This miner passed all checks and is expected to validate successfully on the selected network.")

        # allow new runs and refresh UI (re-enable Start)
        st.session_state.bench_running = False

def help_links():
    st.title("Help & Resources")
    st.subheader("Neural Dashboard")
    st.markdown('- <span class="icon">dashboard</span> <a href="https://app.neuralinternet.ai/" target="_blank">Dashboard</a>',unsafe_allow_html=True)
    st.subheader("Docs")
    st.markdown('- <span class="icon">menu_book</span> <a href="https://docs.neuralinternet.ai/" target="_blank">User Guide</a>\n- <span class="icon">article</span> <a href="https://github.com/neuralinternet/ni-compute" target="_blank">GitHub</a>',unsafe_allow_html=True)
    st.subheader("Community")
    st.markdown('- <span class="icon">public</span> <a href="https://neuralinternet.ai/" target="_blank">Website</a>\n- <span class="icon">chat</span> <a href="https://discord.com/" target="_blank">Discord</a>\n- <span class="icon">campaign</span> <a href="https://x.com/neural_internet" target="_blank">X/Twitter</a>',unsafe_allow_html=True)
    st.subheader("Monitoring")
    st.markdown('- <span class="icon">bar_chart</span> <a href="https://wandb.ai/neuralinternet/opencompute" target="_blank">WandB</a>\n- <span class="icon">insights</span> <a href="https://taostats.io/subnets/27" target="_blank">Taostats</a>',unsafe_allow_html=True)

def chat():
    st.title("Chat Support")
    st.markdown("---")
    st.write("Ask common questions:")
    st.markdown('- <span class="icon">help</span> **What is NI?**\n- <span class="icon">computer</span> **How to mine?**\n- <span class="icon">warning</span> **Why no emissions?**\n- <span class="icon">check_circle</span> **Check miner health**',unsafe_allow_html=True)
    st.link_button("Open Chat", "https://chatgpt.com/g/...", type="secondary", icon=":material/chat:")

def config_page():
    # Only show to logged-in masters
    if st.session_state.role != "master":
        st.stop()

    st.title("Configuration")

    # Fetch the entire config.yaml from the server
    resp = get_data_from_server("config")
    cfg  = resp.get("config", {})
    sc   = cfg.get("subnet_config", {})
    gp   = cfg.get("gpu_performance", {})
    tm   = cfg.get("gpu_time_models", {})
    mp   = cfg.get("merkle_proof", {})

    ### ── Subnet Configuration ────────────────────────────────
    with st.expander("Subnet Configuration", expanded=True):
        with st.form("subnet_form"):
            # Fetch all hotkeys for the multiselect
            all_keys = get_data_from_server("keys").get("keys", [])
            if not all_keys:
                # fallback to metagraph hotkeys
                mg = get_data_from_server("metagraph").get("metagraph", {})
                all_keys = mg.get("hotkeys", [])

            col1, col2 = st.columns(2)
            # Left: GPU weights
            with col1:
                st.write("**GPU Weights**")
                new_gpu_weights = {}
                for gpu, wt in sc.get("gpu_weights", {}).items():
                    new_gpu_weights[gpu] = st.slider(
                        label=gpu,
                        min_value=0.0, max_value=100.0,
                        value=float(wt), step=0.1,
                        help="Relative priority weight"
                    )

            # Right: general subnet parameters + sybil list + reliability + treasury
            with col2:
                st.write("**General Subnet Parameters**")
                total_em     = st.number_input(
                    "Total Miner Emission",
                    min_value=0.0, max_value=1.0,
                    value=float(sc.get("total_miner_emission", 0.05)),
                    step=0.01,
                    help="Fraction of total emission that goes to miners (0..1)."
                )
                blocks_epoch = st.number_input(
                    "Blocks per Epoch",
                    min_value=1, max_value=1000,
                    value=int(sc.get("blocks_per_epoch", 360)),
                    step=1
                )
                max_chg      = st.number_input(
                    "Max Challenge Blocks",
                    min_value=1, max_value=25,
                    value=int(sc.get("max_challenge_blocks", 11)),
                    step=1
                )
                rand_delay   = st.number_input(
                    "Rand Delay Blocks Max",
                    min_value=0, max_value=10,
                    value=int(sc.get("rand_delay_blocks_max", 5)),
                    step=1
                )
                allow_sybil  = st.checkbox(
                    "Allow Fake Sybil Slot",
                    value=bool(sc.get("allow_fake_sybil_slot", True))
                )

                existing_sybil = sc.get("sybil_check_eligible_hotkeys", [])
                new_sybil = st.multiselect(
                    "Select hotkeys eligible for Sybil check",
                    options=all_keys,
                    default=existing_sybil
                )

                reliability_weight = st.number_input(
                    "Reliability Weight",
                    min_value=0.0, max_value=1.0,
                    value=float(sc.get("reliability_weight", 0.5)),
                    step=0.05,
                    help="Blend between no effect (0.0) and full effect (1.0) when applying reliability to scores."
                )

                treasury_wallet_hotkey = st.selectbox(
                    "Treasury Wallet Hotkey",
                    options=[""] + all_keys,
                    index=([""] + all_keys).index(sc.get("treasury_wallet_hotkey", ""))
                        if sc.get("treasury_wallet_hotkey", "") in ([""] + all_keys) else 0,
                    help="Select the wallet hotkey to receive the treasury allocation. Leave empty to disable."
                )

                treasury_emission_share = st.number_input(
                    "Treasury Emission Share",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(sc.get("treasury_emission_share", 0.5)),
                    step=0.01,
                    help="Fraction of the total emission allocated to the treasury (0–1)."
                )

            submitted1 = st.form_submit_button("Save Subnet Configuration")
            msg1       = st.empty()
            if submitted1:
                cfg["subnet_config"] = {
                    "total_miner_emission":       total_em,
                    "blocks_per_epoch":           blocks_epoch,
                    "max_challenge_blocks":       max_chg,
                    "rand_delay_blocks_max":      rand_delay,
                    "allow_fake_sybil_slot":      allow_sybil,
                    "sybil_check_eligible_hotkeys": new_sybil,
                    "gpu_weights":                new_gpu_weights,
                    "reliability_weight":       reliability_weight,
                    "treasury_wallet_hotkey":   treasury_wallet_hotkey,
                    "treasury_emission_share":  treasury_emission_share,
                }
                r = requests.put(
                    f"{SERVER_URL}/config",
                    json=cfg,
                    headers={"X-Admin-Key": ADMIN_KEY},
                    timeout=10
                )
                if r.status_code == 200:
                    msg1.success("✅ Subnet configuration updated")
                else:
                    msg1.error(f"❌ Subnet update failed (status {r.status_code})")

    ### ── PoG - GPU Performance ────────────────────────────────
    with st.expander("PoG - GPU Performance", expanded=False):
        gp16 = gp.get("GPU_TFLOPS_FP16", {})
        gp32 = gp.get("GPU_TFLOPS_FP32", {})
        av   = gp.get("GPU_AVRAM", {})
        scs  = gp.get("gpu_scores", {})

        with st.form("perf_form"):
            # header
            c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
            c1.write("**GPU Name**"); c2.write("**FP16**"); c3.write("**FP32**"); c4.write("**VRAM**"); c5.write("**Score**")

            new_perf = {}
            for gpu in gp16:
                v16 = float(gp16.get(gpu, 0.0))
                v32 = float(gp32.get(gpu, 0.0))
                vr  = float(av.get(gpu,    0.0))
                ss  = float(scs.get(gpu,   0.0))

                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
                c1.write(gpu)
                f16 = c2.number_input(f"{gpu}_fp16", value=v16, format="%.2f", label_visibility="collapsed")
                f32 = c3.number_input(f"{gpu}_fp32", value=v32, format="%.2f", label_visibility="collapsed")
                vrb = c4.number_input(f"{gpu}_vram", value=vr, format="%.2f", label_visibility="collapsed")
                scr = c5.number_input(f"{gpu}_score", value=ss, format="%.2f", label_visibility="collapsed")
                new_perf[gpu] = {"fp16": f16, "fp32": f32, "vram": vrb, "score": scr}

            submitted2 = st.form_submit_button("Save GPU Performance")
            msg2       = st.empty()
            if submitted2:
                updated_gp = cfg.get("gpu_performance", {}).copy()
                updated_gp.update({
                    "GPU_TFLOPS_FP16": { g: new_perf[g]["fp16"] for g in new_perf },
                    "GPU_TFLOPS_FP32": { g: new_perf[g]["fp32"] for g in new_perf },
                    "GPU_AVRAM":        { g: new_perf[g]["vram"] for g in new_perf },
                    "gpu_scores":       { g: new_perf[g]["score"] for g in new_perf },
                })
                cfg["gpu_performance"] = updated_gp

                r2 = requests.put(
                    f"{SERVER_URL}/config",
                    json=cfg,
                    headers={"X-Admin-Key": ADMIN_KEY},
                    timeout=10
                )
                if r2.status_code == 200:
                    msg2.success("✅ GPU performance updated")
                else:
                    msg2.error(f"❌ GPU performance update failed (status {r2.status_code})")

    ### ── PoG - Quadratic GPU Timing Models ───────────────────
    with st.expander("PoG - Quadratic GPU Timing Models", expanded=False):
        with st.form("timing_form"):
            c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
            c1.write("**GPU Name**"); c2.write("**a0**"); c3.write("**a1**"); c4.write("**a2**"); c5.write("**tol**")
            new_tm = {}
            for gpu, params in tm.items():
                a0 = float(params.get("a0", 0.0))
                a1 = float(params.get("a1", 0.0))
                a2 = float(params.get("a2", 0.0))
                tl = float(params.get("tol", 1.0))

                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
                c1.write(gpu)
                na0 = c2.number_input(f"{gpu}_a0", value=a0, format="%.2f", label_visibility="collapsed")
                na1 = c3.number_input(f"{gpu}_a1", value=a1, format="%.2f", label_visibility="collapsed")
                na2 = c4.number_input(f"{gpu}_a2", value=a2, format="%.2f", label_visibility="collapsed")
                ntl = c5.number_input(f"{gpu}_tol", value=tl, format="%.2f", label_visibility="collapsed")
                new_tm[gpu] = {"a0": na0, "a1": na1, "a2": na2, "tol": ntl}

            submitted3 = st.form_submit_button("Save GPU Timing Models")
            msg3       = st.empty()
            if submitted3:
                cfg["gpu_time_models"] = new_tm
                r3 = requests.put(
                    f"{SERVER_URL}/config",
                    json=cfg,
                    headers={"X-Admin-Key": ADMIN_KEY},
                    timeout=10
                )
                if r3.status_code == 200:
                    msg3.success("✅ GPU timing models updated")
                else:
                    msg3.error(f"❌ GPU timing models update failed (status {r3.status_code})")

    ### ── PoG - Merkle Proof Settings ────────────────────────
    with st.expander("PoG - Merkle Proof Settings", expanded=False):
        with st.form("merkle_form"):
            col1, col2 = st.columns(2)
            miner_path  = col1.text_input("Miner Script Path", value=mp.get("miner_script_path", ""))
            hash_algo   = col1.text_input("Hash Algorithm", value=mp.get("hash_algorithm", ""))
            pog_spots   = col1.number_input("Spots per GPU", min_value=1, max_value=100, value=int(mp.get("spot_per_gpu", 3)))
            retry_int   = col1.number_input("PoG Retry Interval (s)", min_value=1, max_value=3600, value=int(mp.get("pog_retry_interval", 60)))

            time_tol    = col2.number_input("Time Tolerance (s)", min_value=0, max_value=300, value=int(mp.get("time_tolerance", 5)))
            submat      = col2.number_input("Submatrix Size", min_value=1, max_value=10_000, value=int(mp.get("submatrix_size", 512)))
            buf_factor  = col2.number_input("Buffer Factor", min_value=0.0, max_value=1.0, value=float(mp.get("buffer_factor", 0.45)), format="%.2f")
            retry_lim   = col2.number_input("PoG Retry Limit", min_value=1, max_value=100, value=int(mp.get("pog_retry_limit", 20)))
            max_workers = col2.number_input("Max Workers", min_value=1, max_value=1024, value=int(mp.get("max_workers", 64)))
            max_delay   = col2.number_input("Max Random Delay (s)", min_value=0, max_value=3600, value=int(mp.get("max_random_delay", 600)))

            submitted4 = st.form_submit_button("Save Merkle Settings")
            msg4       = st.empty()
            if submitted4:
                cfg["merkle_proof"] = {
                    "miner_script_path":  miner_path,
                    "hash_algorithm":     hash_algo,
                    "time_tolerance":     time_tol,
                    "submatrix_size":     submat,
                    "buffer_factor":      buf_factor,
                    "spot_per_gpu":       pog_spots,
                    "pog_retry_limit":    retry_lim,
                    "pog_retry_interval": retry_int,
                    "max_workers":        max_workers,
                    "max_random_delay":   max_delay
                }
                r4 = requests.put(
                    f"{SERVER_URL}/config",
                    json=cfg,
                    headers={"X-Admin-Key": ADMIN_KEY},
                    timeout=10
                )
                if r4.status_code == 200:
                    msg4.success("✅ Merkle proof settings updated")
                else:
                    msg4.error(f"❌ Merkle settings update failed (status {r4.status_code})")


# add next to get_data_from_server()
def post_to_server(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{SERVER_URL}/{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        st.error(f"Error POST {endpoint}: {e}")
        return {}


# ---------------------- Rent Compute Page (Miner-Centric Cards) ----------------------
def build_specs_map(specs):
    """Build hotkey → specs map for quick lookup."""
    specs_map = {}
    for idx, item in specs.items():
        hotkey = item.get("hotkey")
        if not hotkey:
            continue
        det = item.get("details", {}) or {}
        gpu_cap = (det.get("gpu", {}).get("capacity") or 0) / 1024
        cpu_cnt = det.get("cpu", {}).get("count") or 0
        ram_gb  = (det.get("ram", {}).get("available") or 0) / (1024**3)
        disk_gb = (det.get("hard_disk", {}).get("free") or 0) / (1024**3)
        specs_map[hotkey] = {
            "vram": round(gpu_cap, 2),
            "cpu": int(cpu_cnt),
            "ram": round(ram_gb, 2),
            "disk": round(disk_gb, 2),
        }
    return specs_map


# ---- Single source of truth for classification + order ----
PERFORMANCE_TABLE = {
    "Featured GPUs": [
        "NVIDIA B200",
        "NVIDIA H200",
        "NVIDIA A40",
        "NVIDIA GeForce RTX 5090",
    ],
    "NVIDIA Latest Gen": [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 PCIe",
        "NVIDIA H100 NVL",
        "NVIDIA L40S",
        "NVIDIA L40",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX 4000 Ada Generation",
        "NVIDIA L4",
    ],
    "NVIDIA Previous Gen": [
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA RTX A6000",
        "NVIDIA RTX A5000",
        "NVIDIA RTX A4500",
        "NVIDIA RTX A4000",
        "NVIDIA GeForce RTX 3090",
    ],
    "AMD": [
        "AMD MI300X",
    ],
}

def classify_gpu(gpu_name):
    """Classify GPU into category defined in PERFORMANCE_TABLE."""
    for category, models in PERFORMANCE_TABLE.items():
        if gpu_name in models:
            return category
    if any(a in gpu_name for a in PERFORMANCE_TABLE.get("AMD", [])):
        return "AMD"
    return "Other"


def rent_compute():
    st.title("Rent Compute")
    st.caption("Select from available compute instances by GPU type and generation.")

    # Load data
    with st.spinner("Fetching available miners..."):
        data = get_data_from_server("rent/available")
        specs_raw = get_data_from_server("specs").get("specs", {})
        specs_map = build_specs_map(specs_raw)

    groups = data.get("groups", [])
    if not groups:
        st.info("No rentable miners at the moment. Try again later.")
        return

    # === FILTER BAR (inline, top of page) ===
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        min_vram = st.slider("Min VRAM (GB)", 0, 192, 0)
    with f2:
        min_ram = st.slider("Min RAM (GB)", 0, 512, 0)
    with f3:
        min_disk = st.slider("Min Disk (GB)", 0, 2000, 0)
    with f4:
        reset = st.button("Reset Filters")

    if reset:
        min_vram, min_ram, min_disk = 0, 0, 0

    # Category display order
    category_order = list(PERFORMANCE_TABLE.keys())

    # Collect miners into categories (flat, miner-centric)
    catalog = {c: [] for c in category_order}
    for grp in groups:
        for m in grp.get("miners", []):
            hotkey = m.get("hotkey")
            sp = specs_map.get(hotkey, {"vram": 0, "cpu": 0, "ram": 0, "disk": 0})

            # Apply filters
            if sp["vram"] < min_vram or sp["ram"] < min_ram or sp["disk"] < min_disk:
                continue

            category = classify_gpu(m.get("gpu_name", "Unknown"))
            if category in catalog:
                m["_specs"] = sp
                catalog[category].append(m)

    # Render categories
    any_displayed = False
    for category in category_order:
        miners = catalog.get(category, [])
        if not miners:
            continue

        # ---- Sort miners within the category by PERFORMANCE_TABLE order ----
        order_list = PERFORMANCE_TABLE.get(category, [])
        rank_map = {name: idx for idx, name in enumerate(order_list)}
        miners.sort(key=lambda mm: rank_map.get(mm.get("gpu_name", ""), len(order_list)))

        any_displayed = True
        st.subheader(f"{category} ({len(miners)} available)")

        cols_per_row = 4
        rows = (len(miners) + cols_per_row - 1) // cols_per_row
        idx = 0

        for _ in range(rows):
            cols = st.columns(cols_per_row)
            for col in cols:
                if idx >= len(miners):
                    break
                m = miners[idx]; idx += 1
                sp = m["_specs"]

                with col.container(border=True):
                    st.markdown(f"### {m['gpu_name']}")
                    st.markdown(
                        f"**VRAM**: {sp['vram']} GB  \n"
                        f"**CPU**: {sp['cpu']} cores  \n"
                        f"**RAM**: {sp['ram']} GB  \n"
                        f"**Disk**: {sp['disk']} GB"
                    )

                    with st.expander("Details"):
                        st.caption(f"UID: `{m['uid']}`")
                        st.caption(f"Hotkey: `{m['hotkey']}`")
                        st.caption(f"GPUs: `{m['gpu_count']}`")
                        st.caption(f"Axon: {m['axon_ip']}:{m['axon_port']}")
                        if m.get("reliability") is not None:
                            st.caption(f"Reliability: {m['reliability']}")
                        if m.get("score") is not None:
                            st.caption(f"Score: {m['score']}")

                    if st.button("Rent", key=f"rent_{m['hotkey']}", use_container_width=True):
                        with st.spinner("Requesting rental..."):
                            rsp = post_to_server("rent/allocate", {"hotkey": m["hotkey"]})
                        if rsp and rsp.get("status"):
                            st.success("✅ Rental successful.")
                        else:
                            st.error("❌ Rental failed.")

    if not any_displayed:
        st.warning("No miners matched your filters. Try lowering the requirements.")

# ---------------------- Rentals Management Page ----------------------
def my_rentals():
    st.title("My Active Rentals")
    st.caption("View and manage your currently allocated compute nodes.")

    with st.spinner("Loading rentals..."):
        rentals = get_data_from_server("rent/rentals").get("rentals", [])

    if not rentals:
        st.info("You have no active rentals.")
        return

    cols_per_row = 4
    rows = (len(rentals) + cols_per_row - 1) // cols_per_row
    idx = 0

    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for col in cols:
            if idx >= len(rentals):
                break
            r = rentals[idx]; idx += 1
            d = r.get("details", {})
            gpu_name = d.get("gpu_name", "Unknown")
            gpu_count = d.get("gpu_count", "?")

            ssh = d.get("ssh", {})
            ssh_host = ssh.get("host")
            ssh_port = ssh.get("port", 22)
            ssh_user = ssh.get("username", "user")
            ssh_pass = ssh.get("password")

            ssh_cmd = f"ssh -p {ssh_port} {ssh_user}@{ssh_host}"

            with col.container(border=True):
                # Card header
                st.markdown(f"### {gpu_name}")
                st.caption(f"{gpu_count} GPU(s) | UID {r['uid']}")

                # SSH command
                st.markdown("**SSH Command**")
                st.code(ssh_cmd, language="bash")

                # Password (if provided)
                if ssh_pass:
                    st.markdown("**Password**")
                    st.code(ssh_pass, language="bash")

                # Deallocate button
                if st.button("Deallocate", key=f"dealloc_{r['hotkey']}", use_container_width=True):
                    with st.spinner("Requesting deallocation..."):
                        dealloc_rsp = post_to_server("rent/deallocate", {
                            "hotkey": r["hotkey"],
                            "public_key": r["public_key"]
                        })
                    if dealloc_rsp and dealloc_rsp.get("status"):
                        st.success("✅ Deallocated successfully.")
                    else:
                        st.error("❌ Failed to deallocate.")

# ───────────────────── NAVIGATION ─────────────────────────────
network_pages = [
    st.Page(dashboard,   title="Dashboard", icon=":material/dashboard:"),
    st.Page(metagraph,   title="Metagraph", icon=":material/hub:"),
    st.Page(stats,       title="Stats",     icon=":material/insights:")
]

tool_pages = [
    st.Page(search,      title="Search",    icon=":material/search:"),
    st.Page(benchmark,   title="Benchmark", icon=":material/speed:")
]

help_pages = [
    st.Page(help_links,  title="Links & Docs", icon=":material/help:"),
    st.Page(chat,        title="Chat",         icon=":material/chat:")
]

rental_pages = []
# Admin or Master can see rentals
if st.session_state.role in ("admin", "master"):
    rental_pages = [
        st.Page(rent_compute, title="Rent Compute", icon=":material/deployed_code:"),
        st.Page(my_rentals, title="My Rentals", icon=":material/list_alt:")
    ]
# Only Master can see Config
if st.session_state.role == "master":
    help_pages.append(st.Page(config_page, title="Config", icon=":material/settings:"))

pages = {
    "Network Overview": network_pages,
    "Tools": tool_pages,
    "Resources": help_pages,
}

# Only show Compute Rentals if role allows
if rental_pages:
    pages["Compute Rentals"] = rental_pages

pg = st.navigation(pages, position="sidebar")
pg.run()
