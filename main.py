import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import matplotlib.pyplot as plt
import base64
import plotly.graph_objects as go # type: ignore
import plotly.express as px  # type: ignore # Interactive chart

# Configure Streamlit page
logo_path = "Icon_White_crop.png"  # Update this with the correct path to your logo
st.set_page_config(page_title="NI-Compute", layout="wide", page_icon=logo_path)

# Set Sidebar Logo
logo_path2 = "Neural_Internet_White_crop.png"
st.logo(logo_path2, size="large", link=None, icon_image=None)

# Inject Custom CSS to Resize the Logo
st.markdown(
    """
    <style>
        img[data-testid="stLogo"] {
            height: 8rem !important;
            width: auto !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Server details
SERVER_IP = "65.108.33.88"
SERVER_PORT = "8000"
SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

# Function to fetch data
def get_data_from_server(endpoint):
    try:
        response = requests.get(f"{SERVER_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json() or {}  # Ensure a valid dictionary is returned
        else:
            st.error(f"Failed to fetch {endpoint}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {endpoint}: {e}")
    
    return {}  # Always return an empty dictionary on failure

# Function to display hardware specs
def display_hardware_specs(specs_details, allocated_keys, penalized_keys, metagraph_data):
    """Displays a professional, interactive hardware specs table in Streamlit with correct sorting and incentive data."""

    column_headers = [
        "UID", "Hotkey", "Incentive", "GPU Name", "GPU Capacity (GiB)", "GPU Count",
        "CPU Count", "RAM (GiB)", "Disk Space (GiB)", "Status", "PoG Status", "Penalized"
    ]

    # Fetch incentives from metagraph data (default to 0 if missing)
    incentives_dict = dict(zip(metagraph_data.get("uids", []), metagraph_data.get("incentive", [])))

    table_data = []

    for index in sorted(specs_details.keys(), key=lambda x: int(x)):  # ✅ Sort UIDs numerically
        item = specs_details.get(index, {})

        if not item or not isinstance(item, dict):
            table_data.append([int(index), "No item", 0, "N/A", 0, 0, 0, 0, 0, "N/A", "Unverified", "No"])  # ✅ Updated PoG label
            continue

        stats_data = item.get('stats', {}) or {}
        hotkey = item.get('hotkey', 'unknown')  # ✅ Full hotkey (no truncation)
        details = item.get('details', {}) or {}

        # Extract relevant data (ensuring correct numeric values)
        gpu_specs = stats_data.get('gpu_specs', {}) or {}
        gpu_name = gpu_specs.get('gpu_name', "N/A")
        gpu_count = gpu_specs.get('num_gpus', 0)

        gpu_miner = details.get('gpu', {}) or {}
        gpu_capacity = gpu_miner.get('capacity', 0) / 1024 if isinstance(gpu_miner, dict) else 0  # ✅ Convert to GiB, ensure numeric

        cpu_miner = details.get('cpu', {}) or {}
        cpu_count = cpu_miner.get('count', 0) if isinstance(cpu_miner, dict) else 0  # ✅ Ensure numeric

        ram_miner = details.get('ram', {}) or {}
        ram_gib = ram_miner.get('available', 0) / (1024.0**3) if isinstance(ram_miner, dict) else 0  # ✅ Convert to GiB, ensure numeric

        hard_disk_miner = details.get('hard_disk', {}) or {}
        disk_gib = hard_disk_miner.get('free', 0) / (1024.0**3) if isinstance(hard_disk_miner, dict) else 0  # ✅ Convert to GiB, ensure numeric

        # Allocated & Penalized
        status = "Res." if hotkey in allocated_keys else "Avail."
        pog_status = "Pass" if gpu_specs else "Fail"  # ✅ Updated PoG Status label
        penalized = "Yes" if hotkey in penalized_keys else "No"  # ✅ Penalization status

        # Get incentive for this UID (default to 0 if missing)
        incentive = incentives_dict.get(int(index), 0)

        table_data.append([
            int(index),  # ✅ UID stored as an integer for correct sorting
            hotkey,  # ✅ Full hotkey displayed (not truncated)
            round(incentive, 6),  # ✅ Moved Incentive column right after Hotkey
            gpu_name, round(gpu_capacity, 2), int(gpu_count),
            int(cpu_count), round(ram_gib, 2), round(disk_gib, 2), status, pog_status, penalized  # ✅ Penalized info is now last column
        ])

    # Convert to DataFrame
    df = pd.DataFrame(table_data, columns=column_headers)

    # ✅ Ensure no extra row by resetting index
    df = df.set_index("UID")  # ✅ Removes automatic index, UID becomes the first column

    # ✅ Remove automatic index column and enable proper sorting
    st.dataframe(df.style.format({
        "GPU Capacity (GiB)": "{:.2f}",
        "RAM (GiB)": "{:.2f}",
        "Disk Space (GiB)": "{:.2f}",
        "Incentive": "{:.6f}",  # ✅ Display incentives with 6 decimal places
    }), use_container_width=True, height=1000)  # ✅ Unlimited height (high enough to show all)


# Function to plot Incentive vs UID
def plot_incentive_vs_uid(metagraph_data):
    """Plots incentive vs UID number with a modern, dark theme."""
    if not metagraph_data or "uids" not in metagraph_data or "incentive" not in metagraph_data:
        st.error("Incomplete metagraph data. Try again later.")
        return

    uids = metagraph_data["uids"]
    incentives = metagraph_data["incentive"]

    # Ensure valid data
    if not uids or not incentives or len(uids) != len(incentives):
        st.error("Inconsistent data in metagraph response.")
        return

    # Convert lists to DataFrame
    df = pd.DataFrame({"UID": uids, "Incentive": incentives})

    # Sort incentives in descending order but keep UIDs unsorted
    df = df.sort_values(by="Incentive", ascending=True)  # Lowest incentive on the left

    # Create scatter plot with Plotly (high resolution & dark theme)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(df))),  # Just tick indices, not sorted UIDs
        y=df["Incentive"],
        mode='markers',  # Scatter plot (dots only)
        marker=dict(
            size=8, 
            color=df["Incentive"],  # Color based on incentive value
            colorscale="reds",  # Professional gradient color
            showscale=True
        ),
    ))

    # Custom styling
    fig.update_layout(
        title="Incentive Distribution",
        xaxis=dict(
            title="Miners",
            showticklabels=False,  # Hide numbers on x-axis (ticks only)
        ),
        yaxis=dict(
            title="Incentive",
            gridcolor="gray"
        ),
        template="plotly_dark",  # Dark mode to match Streamlit
        margin=dict(l=40, r=40, t=50, b=40),  # Clean margins
    )

    # Display interactive plot
    st.plotly_chart(fig, use_container_width=True)

# Define pages
def dashboard():
    st.title("Dashboard - NI Compute")
    load_dotenv()

    with st.spinner('Fetching data from server...'):
        try:
            hotkeys_response = get_data_from_server("keys")
            hotkeys = hotkeys_response.get("keys", [])

            specs_response = get_data_from_server("specs")
            specs_details = specs_response.get("specs", {})

            allocated_keys_response = get_data_from_server("allocated_keys")
            allocated_keys = allocated_keys_response.get("allocated_keys", [])

            penalized_keys_response = get_data_from_server("penalized_keys")
            penalized_keys = penalized_keys_response.get("penalized_keys", [])

            # Fetch metagraph data
            metagraph_response = get_data_from_server("metagraph")
            metagraph_data = metagraph_response.get("metagraph", {})

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            specs_details, allocated_keys, penalized_keys, metagraph_data = {}, [], [], {}

    # Plot incentive vs UID
    #st.subheader("Incentive Distribution by Miner")
    try:
        plot_incentive_vs_uid(metagraph_data)
    except Exception as e:
        st.error(f"Unable to generate incentive plot: {e}")

    # Display specs
    try:
        display_hardware_specs(specs_details, allocated_keys, penalized_keys, metagraph_data)
    except Exception as e:
        st.error(f"Unable to fetch or display data: {e}")

import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd

def metagraph():
    """Displays Metagraph information using fetched metagraph_data from the API."""

    st.title("Metagraph Data")

    # Fetch metagraph data from API
    with st.spinner("Fetching metagraph data..."):
        metagraph_response = get_data_from_server("metagraph")
        metagraph_data = metagraph_response.get("metagraph", {})

    if not metagraph_data or "uids" not in metagraph_data:
        st.error("Metagraph data not available. Please try again later.")
        return

    # Extract data from metagraph_data
    uids = metagraph_data.get("uids", [])
    hotkeys = metagraph_data.get("hotkeys", [])
    active = metagraph_data.get("active", [])
    stake = metagraph_data.get("stake", [])
    trust = metagraph_data.get("trust", [])
    v_trust = metagraph_data.get("validator_trust", [])
    v_permit = metagraph_data.get("validator_permit", [])
    axons = metagraph_data.get("axons", [])  # Contains IP, port, and version

    miner_version_summary = {}
    validator_version_summary = {}

    # Prepare DataFrame
    data = []

    for i, uid in enumerate(uids):
        axon = axons[i] if axons and len(axons) > i else {}
        ip = axon.get("ip", "N/A")
        port = axon.get("port", "N/A")
        version = axon.get("version", "N/A")  # ✅ Correctly fetching axon version

        # Track versions for miners and validators
        if v_trust[i] == 0:
            miner_version_summary[version] = miner_version_summary.get(version, 0) + 1
        else:
            validator_version_summary[version] = validator_version_summary.get(version, 0) + 1

        data.append([
            uid, hotkeys[i], active[i], round(stake[i], 6), round(trust[i], 6), 
            v_permit[i], round(v_trust[i], 6), ip, port, version
        ])

    # Create DataFrame
    columns = ['UID', 'Hotkey', 'Active', 'Stake', 'Trust', 'V_Permit', 'V_Trust', 'IP', 'Port', 'Version']
    df = pd.DataFrame(data, columns=columns)

    # Display Metagraph Nodes Data
    st.write("### Metagraph Nodes Data")
    st.dataframe(df.style.format({
        "Stake": "{:.6f}",
        "Trust": "{:.6f}",
        "V_Trust": "{:.6f}",
    }), use_container_width=True, height=800)

    col1, col2 = st.columns([1, 1])  # Use only the first column (50% width)

    # Validator Version Summary
    validator_count = sum(validator_version_summary.values())
    validator_summary_data = [
        {"Version": version, "Count": count, "Percentage": (count / validator_count * 100)}
        for version, count in validator_version_summary.items()
    ]
    validator_summary_df = pd.DataFrame(validator_summary_data)

    col2.write("### Validator Version Summary")
    col2.write(f"Total Validator Count: {validator_count}")
    col2.dataframe(validator_summary_df.style.format({"Percentage": "{:.2f}%"}), use_container_width=True)

    # Miner Version Summary
    miner_count = sum(miner_version_summary.values())
    miner_summary_data = [
        {"Version": version, "Count": count, "Percentage": (count / miner_count * 100)}
        for version, count in miner_version_summary.items()
    ]
    miner_summary_df = pd.DataFrame(miner_summary_data)

    col1.write("### Miner Version Summary")
    col1.write(f"Total Miner Count: {miner_count}")
    col1.dataframe(miner_summary_df.style.format({"Percentage": "{:.2f}%"}), use_container_width=True)

# Define supported GPU models
SUPPORTED_GPUS = {
    "NVIDIA B200:": "B200",
    "NVIDIA H200": "H200",
    "NVIDIA H100 80GB HBM3": "H100 80GB HBM3",
    "NVIDIA H100": "H100 80GB PCIE",
    "NVIDIA A100-SXM4-80GB": "A100 80GB SXM4",
    "NVIDIA A100-SXM4-40GB": "A100 40GB SXM4",
    "NVIDIA GeForce RTX 5090": "RTX 5090",
    "NVIDIA L40s": "L40s",
    "NVIDIA RTX 6000 Ada Generation": "RTXA6000 Ada",
    "NVIDIA L40": "L40",
    "NVIDIA RTX A6000": "RTX A6000",
    "NVIDIA RTX 4090": "RTX 4090",
    "NVIDIA RTX A5000": "RTX A5000"
}

OTHER_GPU_LABEL = "Other GPUs"

def stats():
    """Network Statistics Page"""
    st.title("Network Statistics")

    with st.spinner('Fetching data from server...'):
        try:
            allocated_keys_response = get_data_from_server("allocated_keys")
            allocated_keys = allocated_keys_response.get("allocated_keys", [])

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            allocated_keys = []

    # Fetch data from the server
    with st.spinner("Fetching data..."):
        specs_response = get_data_from_server("specs")
        specs_details = specs_response.get("specs", {})

    if not specs_details:
        st.error("No data available. Try again later.")
        return

    # Extract GPU information
    gpu_counts = {}  # Total GPUs per type
    rented_gpu_counts = {}  # Allocated GPUs per type
    total_gpus = 0

    for item in specs_details.values():
        stats_data = item.get('stats', {}) or {}
        gpu_specs = stats_data.get('gpu_specs', {}) or {}

        gpu_name = gpu_specs.get('gpu_name', "Unknown GPU")
        num_gpus = gpu_specs.get('num_gpus', 0)
        status = "Res." if item.get('hotkey', '') in allocated_keys else "Avail."  # Check if allocated

        # Categorize GPUs
        if gpu_name in SUPPORTED_GPUS:
            gpu_label = SUPPORTED_GPUS[gpu_name]
        else:
            gpu_label = OTHER_GPU_LABEL  # Group all non-supported GPUs

        # Total GPU count
        if gpu_label not in gpu_counts:
            gpu_counts[gpu_label] = 0
        gpu_counts[gpu_label] += num_gpus
        total_gpus += num_gpus

        # Count rented GPUs
        if status == "Res.":
            if gpu_label not in rented_gpu_counts:
                rented_gpu_counts[gpu_label] = 0
            rented_gpu_counts[gpu_label] += num_gpus

    # Convert GPU data into a DataFrame
    gpu_data = pd.DataFrame([
        {
            "GPU Model": gpu, 
            "Count": count, 
            "Rented": rented_gpu_counts.get(gpu, 0), 
            "Percentage": (count / total_gpus) * 100
        }
        for gpu, count in gpu_counts.items()
    ])

    # Sort the table by GPU count
    gpu_data = gpu_data.sort_values(by="Count", ascending=False)

    # Create a 2-column layout
    col1, col2 = st.columns([1.5, 2])  # Adjust width ratio (table slightly narrower than chart)

    # 📊 Column 1: Pie Chart + Table
    with col1:
        # 🎯 **Pie Chart: GPU Distribution** 
        st.markdown("#### GPU Distribution")
        fig = px.pie(
            gpu_data, 
            names="GPU Model", 
            values="Count", 
            title="", 
            color_discrete_sequence=px.colors.sequential.Reds
        )

        # Adjust layout for better visual balance
        fig.update_layout(
            width=800,  # Bigger chart
            height=600,  # Bigger height
            showlegend=True, 
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)  # ✅ Move legend below chart
        )

        # Display the pie chart
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(gpu_data.style.format({"Percentage": "{:.2f}%"}), use_container_width=True)

        st.markdown("#### GPU Rentals")

        for gpu in gpu_counts:
            if gpu == OTHER_GPU_LABEL:
                continue  # 🚫 Skip "Other GPUs" in progress bars

            total_count = gpu_counts[gpu]
            rented_count = rented_gpu_counts.get(gpu, 0)
            rented_percentage = rented_count / total_count if total_count > 0 else 0

            st.text(f"{gpu}: {rented_count}/{total_count} GPUs rented")
            st.progress(rented_percentage)  # Progress bar showing allocation status

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
            if not hardware_info or "stats" not in hardware_info or not hardware_info.get("stats", {}).get("gpu_specs"):
                col.error("⚠️ This hotkey is registered but inactive. It did not pass Proof-of-GPU (PoG) verification.")
                return  # Stop execution to avoid further errors

            # ✅ **Display Hardware Information**
            col.subheader("Hardware Information")
            stats = hardware_info.get("stats", {})
            details = hardware_info.get("details", {})

            gpu_specs = stats.get("gpu_specs", {})
            cpu_specs = details.get("cpu", {})
            gpu_miner = details.get("gpu", {})
            ram_specs = details.get("ram", {})
            disk_specs = details.get("hard_disk", {})

            hardware_table = {
                "GPU Name": gpu_specs.get("gpu_name", "N/A"),
                "GPU Count": gpu_specs.get("num_gpus", 0),
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
                "PoG Status": "Pass" if gpu_specs else "Fail",
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
    st.write("Run PoG benchmarking tests for nodes and compute resources:")

    # Path to your ZIP file (make sure it's in the same directory or adjust the path)
    zip_file_path = "Data/PoG_Benchmark.zip"

    # Read the ZIP file as bytes
    try:
        with open(zip_file_path, "rb") as zip_file:
            zip_data = zip_file.read()

        # Provide a download button
        st.download_button(
            label="Download the PoG Benachmark Files (Validator & Miner)",
            data=zip_data,
            file_name="PoG_Benchmark.zip",
            mime="application/zip"
        )
    except FileNotFoundError:
        st.error(f"Could not find the file '{zip_file_path}'. Please ensure it exists.")

    # Expected benchmark result (formatted as code)
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

# Load Material Icons & Apply Red Styling to Icons and Links
st.markdown(
    """
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <style>
        .icon {
            font-family: 'Material Icons';
            font-size: 18px;
            vertical-align: middle;
            margin-right: 5px;
            color: #D32F2F;  /* Red color for icons */
        }
        a {
            color: #D32F2F !important;  /* Red color for links */
            text-decoration: none;
            font-weight: light;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True
)

def help_links():
    """Help & Useful Links Page"""
    st.title("Help & Resources")
    st.write("Find useful resources, documentation, and support links below.")

    # Neural Dashboard & Access
    st.subheader("Neural Dashboard & Access")
    st.markdown(
        """
        - <span class="icon">dashboard</span> <a href="https://app.neuralinternet.ai/" target="_blank">Neural Dashboard App</a>
        """, unsafe_allow_html=True
    )

    # Documentation & Support
    st.subheader("Documentation & Support")
    st.markdown(
        """
        - <span class="icon">menu_book</span> <a href="https://docs.neuralinternet.ai/ni-ecosystem/ni-compute-sn27" target="_blank">User Guide</a>
        - <span class="icon">article</span> <a href="https://github.com/neuralinternet/ni-compute/blob/main/README.md" target="_blank">Readme</a>
        """, unsafe_allow_html=True
    )

    # Community & External Resources
    st.subheader("Community & External Resources")
    st.markdown(
        """
        - <span class="icon">public</span> <a href="https://neuralinternet.ai/" target="_blank">Official Website</a>
        - <span class="icon">code</span> <a href="https://github.com/neuralinternet/ni-compute" target="_blank">GitHub Repository</a>
        - <span class="icon">chat</span> <a href="https://discord.com/channels/799672011265015819/1174835090539433994" target="_blank">Discord Community</a>
        - <span class="icon">campaign</span> <a href="https://x.com/neural_internet" target="_blank">X/Twitter Updates</a>
        """, unsafe_allow_html=True
    )

    # Performance & Monitoring Tools
    st.subheader("Performance & Monitoring Tools")
    st.markdown(
        """
        - <span class="icon">bar_chart</span> <a href="https://wandb.ai/neuralinternet/opencompute/runs/0djlnjjs/overview" target="_blank">WandB - Validator</a>
        - <span class="icon">insights</span> <a href="https://taostats.io/subnets/27/chart" target="_blank">Taostats (SN27)</a>
        """, unsafe_allow_html=True
    )

def chat():
    """Neural Internet Chat Support (Now Powered by ChatGPT)"""
    st.title("Chat Support")
    
    st.markdown("---")

    st.write("The Neural Internet ChatGPT assistant can help answer common questions like:")

    st.markdown(
        """
        - <span class="icon">help</span> **What is Neural Internet?**
        - <span class="icon">computer</span> **How do I mine on NI Compute?**
        - <span class="icon">warning</span> **Why is my miner not receiving emissions?**
        - <span class="icon">check_circle</span> **How do I check if my miner is running properly?**
        """,
        unsafe_allow_html=True,
    )

    # Button to ChatGPT-powered Neural Internet chat
    chat_link = "https://chatgpt.com/g/g-67cf2506b6f88191a4400819f6963378-neural-internet"
    st.link_button("Open Neural Internet Chat", chat_link, type="secondary", icon=":material/chat:")

    st.markdown("---")

# Organizing Pages into Sections
network_pages = [
    st.Page(dashboard, title="Dashboard", icon=":material/dashboard:"),
    st.Page(metagraph, title="Metagraph", icon=":material/hub:"),
    st.Page(stats, title="Stats", icon=":material/insights:")
]

tool_pages = [
    st.Page(search, title="Search", icon=":material/search:"),
    st.Page(benchmark, title="Benchmark", icon=":material/speed:"),  
]

help_pages = [
    st.Page(help_links, title="Links & Docs", icon=":material/help:"),
    st.Page(chat, title="Chat", icon=":material/chat:") 
        ]

# Apply navigation
pg = st.navigation(
    {
        "Network Overview": network_pages,
        "Tools": tool_pages,
        "Resources": help_pages
    },
    position="sidebar"
)

# --- About Section (Minimalist & Professional) ---
# Apply navigation
pg = st.navigation(
    {
        "Network Overview": network_pages,
        "Tools": tool_pages,
        "Resources": help_pages
    },
    position="sidebar"
)

# --- About Section (Minimalist & Informative) ---
#with st.sidebar.expander("About", expanded=False):
st.sidebar.info(
    """ 
    **Neural Internet Compute**  
    A decentralized AI computing network for high-performance compute resource sharing.  

    **Key Features:**  
    - AI & ML optimized GPU computing  
    - Secure, decentralized validation  
    - Blockchain-based incentives  
    - Open-source and community-driven   
    """,
)

# Run the selected page
pg.run()
