import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import time

# Configure the page to use wide layout
st.set_page_config(page_title="Opencompute", layout="wide", page_icon="icon.ico")

# Server details
SERVER_IP = "65.108.33.88"
SERVER_PORT = "8000"
SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

def get_data_from_server(endpoint):
    response = requests.get(f"{SERVER_URL}/{endpoint}")
    if response.status_code == 200:
        return response.json()
    else:
        return {}

def display_hardware_specs(specs_details, allocated_keys, penalized_keys):
    column_headers = [
        "UID", "Hotkey", 
        "GPU Name", 
        "GPU Capacity (GiB)", 
        "GPU Count", 
        "CPU Count", 
        "RAM (GiB)", 
        "Disk Space (GiB)", 
        "Status", 
        "Conformity"
    ]

    table_data = []
    gpu_instances = {}
    total_gpu_counts = {}

    for index in sorted(specs_details.keys()):
        item = specs_details[index]  # e.g. { "hotkey":..., "details":..., "stats": ... }
        if not item:
            # If item itself is None or empty, skip or handle
            row = [str(index), "No item", "N/A"] + ["N/A"] * 7
            table_data.append(row)
            continue
        
        # Handle stats = null -> None
        stats_data = item.get('stats') or {}

        hotkey = item.get('hotkey', 'unknown')
        details = item.get('details', {})
        
        # Now stats_data is guaranteed a dict
        gpu_specs = stats_data.get('gpu_specs')  # might be None

        # GPU Name & Count from stats
        if gpu_specs is not None:
            gpu_name = gpu_specs.get('gpu_name', "N/A")
            gpu_count = gpu_specs.get('num_gpus', 0)
        else:
            gpu_name = "No GPU Stats"
            gpu_count = 0

        # GPU capacity from the miner’s `details`
        try:
            gpu_miner = details.get('gpu', {})
            capacity_mib = gpu_miner.get('capacity', 0)
            gpu_capacity = "{:.2f}".format(capacity_mib / 1024)
        except (KeyError, TypeError):
            gpu_capacity = "N/A"

        # CPU
        try:
            cpu_miner = details.get('cpu', {})
            cpu_count = cpu_miner.get('count', 0)
        except:
            cpu_count = "N/A"

        # RAM
        try:
            ram_miner = details.get('ram', {})
            ram_bytes = ram_miner.get('available', 0)
            ram_gib = "{:.2f}".format(ram_bytes / (1024.0**3))
        except:
            ram_gib = "N/A"

        # Disk
        try:
            hard_disk_miner = details.get('hard_disk', {})
            disk_bytes = hard_disk_miner.get('free', 0)
            disk_gib = "{:.2f}".format(disk_bytes / (1024.0**3))
        except:
            disk_gib = "N/A"

        # Allocated & Penalized
        status = "Res." if hotkey in allocated_keys else "Avail."
        # If penalized OR stats is missing GPU specs, mark as "No"
        if hotkey in penalized_keys or gpu_specs is None:
            conform = "No"
        else:
            conform = "Yes"

        # Final row
        row = [
            str(index),
            hotkey[:6] + "...",
            gpu_name,
            gpu_capacity,
            str(gpu_count),
            str(cpu_count),
            ram_gib,
            disk_gib,
            status,
            conform
        ]
        table_data.append(row)

        # Summaries for GPU
        if gpu_specs is not None and gpu_name != "N/A" and gpu_name != "No GPU Stats" and gpu_count > 0:
            gpu_key = (gpu_name, gpu_count)
            gpu_instances[gpu_key] = gpu_instances.get(gpu_key, 0) + 1
            total_gpu_counts[gpu_name] = total_gpu_counts.get(gpu_name, 0) + gpu_count

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Hardware Overview", "Instances Summary", "Total GPU Counts"])

    with tab1:
        df = pd.DataFrame(table_data, columns=column_headers)
        st.table(df)

    with tab2:
        summary_data = [
            [gpu_key[0], str(gpu_key[1]), str(instances)]
            for gpu_key, instances in gpu_instances.items()
        ]
        if summary_data:
            st.table(pd.DataFrame(summary_data, columns=["GPU Name", "GPU Count", "Instances Count"]))
        else:
            st.write("No GPU instance data to summarize.")

    with tab3:
        summary_data = [[name, str(count)] for name, count in total_gpu_counts.items()]
        if summary_data:
            st.table(pd.DataFrame(summary_data, columns=["GPU Name", "Total GPU Count"]))
        else:
            st.write("No total GPU count data to display.")

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title('NI Compute (SN27) - Hardware Specifications')

# Load environment vars (optional)
load_dotenv()

# Fetching data from external server
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

    except:
        st.error("Error fetching data from server.")
        specs_details = {}
        allocated_keys = []
        penalized_keys = []

# Display fetched hardware specs
try:
    display_hardware_specs(specs_details, allocated_keys, penalized_keys)
except Exception as e:
    st.write("Unable to connect to the server or parse data. Please try again later.")
    st.write(f"Exception: {e}")
