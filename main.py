import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import matplotlib.pyplot as plt
import base64
import plotly.graph_objects as go  # type: ignore
import plotly.express as px        # type: ignore

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
ADMIN_PASSWORD = get_config("ADMIN_PASSWORD", "")
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
        r = requests.get(f"{SERVER_URL}/{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        st.error(f"Error fetching {endpoint}: {e}")
        return {}

# ─────────────────────── SIDEBAR LOGIN (no button) ────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def _login_cb():
    """Called when the user presses Enter in the password box."""
    if st.session_state.get("admin_pwd", "") == ADMIN_PASSWORD:
        st.session_state.authenticated = True
    else:
        st.sidebar.error("❌ Invalid password")

with st.sidebar:
    if not st.session_state.authenticated:
        # on_change fires when they hit Enter
        st.text_input(
            "Admin Login",
            type="password",
            key="admin_pwd",
            on_change=_login_cb
        )
    else:
        if st.button("Logout"):
            # 1) log them out
            st.session_state.authenticated = False
            # 2) clear the saved password so the box is empty on next show
            if "admin_pwd" in st.session_state:
                del st.session_state["admin_pwd"]
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
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA GeForce RTX 3090": "RTX 3090",
    "NVIDIA L4": "L4"
}

OTHER_GPU_LABEL = "Other GPUs"

def stats():
    st.title("Network Statistics")
    with st.spinner("Loading..."):
        alloc = get_data_from_server("allocated_keys").get("allocated_keys", [])
        specs = get_data_from_server("specs").get("specs", {})
    if not specs:
        st.error("No data available."); return
    gpu_counts = {}
    rented_gpu = {}
    total = 0
    for itm in specs.values():
        sd = itm.get("stats",{}) or {}
        name = sd.get("gpu_name","Unknown GPU")
        raw_cnt = sd.get("gpu_num")
        cnt = int(raw_cnt) if isinstance(raw_cnt, (int, float)) else 0
        status = "Res." if itm.get("hotkey","") in alloc else "Avail."
        lbl = SUPPORTED_GPUS.get(name, OTHER_GPU_LABEL)
        gpu_counts[lbl] = gpu_counts.get(lbl,0)+cnt
        total += cnt
        if status=="Res.":
            rented_gpu[lbl] = rented_gpu.get(lbl,0)+cnt
    gpu_data = pd.DataFrame([
        {"GPU Model":g, "Count":c,
         "Rented":rented_gpu.get(g,0),
         "Percentage":(c/total*100) if total else 0}
        for g,c in gpu_counts.items()
    ]).sort_values("Count",ascending=False)
    c1, c2 = st.columns([1.5,2])
    with c1:
        st.markdown("#### GPU Distribution")
        fig = px.pie(gpu_data,names="GPU Model",values="Count",
                     color_discrete_sequence=px.colors.sequential.Blues)
        fig.update_layout(width=800,height=600,showlegend=True,
                          legend=dict(orientation="h",yanchor="bottom",y=-0.2,
                                      xanchor="center",x=0.5))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(gpu_data.style.format({"Percentage":"{:.2f}%"}),use_container_width=True)
        st.markdown("#### GPU Rentals")
        for g in gpu_counts:
            if g==OTHER_GPU_LABEL: continue
            tot = gpu_counts[g]; ren = rented_gpu.get(g,0)
            pct = ren/tot if tot else 0
            st.text(f"{g}: {ren}/{tot} rented")
            st.progress(pct)

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
    st.title("Benchmark Tool")
    st.write("Download PoG benchmark files:")
    try:
        data = open("Data/PoG_Benchmark.zip","rb").read()
        st.download_button("Download ZIP",data,"PoG_Benchmark.zip","application/zip")
    except Exception:
        st.error("ZIP not found.")

    benchmark_output = """\
[Step 1] Establishing SSH connection to the miner...
[Step 1] Sending miner script and requesting remote hash.
[Integrity Check] SUCCESS: Local and remote hashes match.

[Step 2] Executing benchmarking mode on the miner...
[Step 2] Benchmarking completed.
[Benchmark Results] Detected 4 GPU(s) with 34.36 GB unfractured VRAM.
                    FP16 - Matrix Size: 92672, Execution Time: 3.099146 s
                    FP32 - Matrix Size: 46336, Execution Time: 3.778708 s
[Performance Metrics] Calculated TFLOPS:
                    FP16: 513.61 TFLOPS
                    FP32: 52.66 TFLOPS
[GPU Identification] Based on performance: NVIDIA H100 80GB HBM3

[Step 3] Initiating Merkle Proof Mode.
[Step 3] Compute mode executed on miner - Matrix Size: 20704
         Compute mode execution time: 9.99 seconds.
[Merkle Proof] Root hashes received from GPUs:
        GPU 0: d8a28e819e5225d4fe1e6313b361edb32d96309ed069ca0bef78afe40faea236
        GPU 1: 89dd17f81441cf595219afbb80681945ee12aed2ecc13130380bdd75892d056d
        GPU 3: e267818f0186721ed2f23458d8414420a6f1c08862381563b58ffe0c6a70e1fe
        GPU 2: 4e0037c4bd23b36772015c50a47e21c95b6a60b5cc38666598fc4f6750c0c6c5
Average Matrix Multiplication Time: 0.7565 seconds
Average Merkle Tree Time: 1.2923 seconds
[Merkle Proof] Proof mode executed on miner.
[Merkle Proof] Responses received from miner.
[Verification] GPU 0 passed verification.
[Verification] GPU 1 passed verification.
[Verification] GPU 3 passed verification.
[Verification] GPU 2 passed verification.
[Verification] SUCCESS: 4 out of 4 GPUs passed verification.

==================================================
[Verification] SUCCESS
  Merkle Proof: PASSED
  GPU Identification: Detected 4 x NVIDIA H100 80GB HBM3 GPU(s)
==================================================
Score: 37.5/100
Total time: 30.88 seconds.
    """

    st.markdown("---")

    # Display the benchmark output as a code block
    st.markdown("#### Expected Validator Output")
    st.code(benchmark_output, language="bash")  # Syntax highlighting for readability

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
    # Only show to logged-in admins
    if not st.session_state.authenticated:
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

            # Right: general subnet parameters + sybil list
            with col2:
                st.write("**General Subnet Parameters**")
                total_em     = st.number_input(
                    "Total Miner Emission",
                    min_value=0.0, max_value=1.0,
                    value=float(sc.get("total_miner_emission", 0.05)),
                    step=0.01
                )
                blocks_epoch = st.number_input(
                    "Blocks per Epoch",
                    min_value=1, max_value=100_000,
                    value=int(sc.get("blocks_per_epoch", 360)),
                    step=1
                )
                max_chg      = st.number_input(
                    "Max Challenge Blocks",
                    min_value=1, max_value=1000,
                    value=int(sc.get("max_challenge_blocks", 11)),
                    step=1
                )
                rand_delay   = st.number_input(
                    "Rand Delay Blocks Max",
                    min_value=0, max_value=1000,
                    value=int(sc.get("rand_delay_blocks_max", 5)),
                    step=1
                )
                allow_sybil  = st.checkbox(
                    "Allow Fake Sybil Slot",
                    value=bool(sc.get("allow_fake_sybil_slot", True))
                )

                st.markdown("**Sybil-check Eligible Hotkeys**")
                existing_sybil = sc.get("sybil_check_eligible_hotkeys", [])
                new_sybil = st.multiselect(
                    "Select hotkeys eligible for Sybil check",
                    options=all_keys,
                    default=existing_sybil
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
                    "gpu_weights":                new_gpu_weights
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
                cfg["gpu_performance"] = {
                    "GPU_TFLOPS_FP16": { g: new_perf[g]["fp16"] for g in new_perf },
                    "GPU_TFLOPS_FP32": { g: new_perf[g]["fp32"] for g in new_perf },
                    "GPU_AVRAM":        { g: new_perf[g]["vram"] for g in new_perf },
                    "gpu_scores":       { g: new_perf[g]["score"] for g in new_perf }
                }
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
if st.session_state.authenticated:
    help_pages.append(st.Page(config_page, title="Config", icon=":material/settings:"))

pages = {
    "Network Overview": network_pages,
    "Tools": tool_pages,
    "Resources": help_pages
}

pg = st.navigation(pages, position="sidebar")
#st.sidebar.info("**Neural Internet Compute**  \nDecentralized, high-performance GPU sharing.")
pg.run()
