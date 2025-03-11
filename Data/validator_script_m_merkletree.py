import paramiko
import hashlib
import numpy as np
import os
import time
import blake3
import secrets  # For secure random seed generation
import json
import tempfile
import yaml
import torch
from paramiko import SSHException, AuthenticationException

def load_yaml_config(file_path):
    """
    Load GPU performance data from a YAML file.
    """
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error decoding YAML file {file_path}: {e}")

def identify_gpu(fp16_tflops, fp32_tflops, estimated_avram, reported_name=None, tolerance_pairs=None):
    """
    Identify GPU based on TFLOPS and AVRAM with a tolerance check for GPUs with similar fingerprints.

    Parameters:
        fp16_tflops (float): Measured FP16 TFLOPS.
        fp32_tflops (float): Measured FP32 TFLOPS.
        estimated_avram (float): Estimated available VRAM in GB.
        reported_name (str): GPU name reported by the system (optional).
        tolerance_pairs (dict): Dictionary of GPUs with similar performance to apply tolerance adjustments.

    Returns:
        str: Identified GPU name with tolerance handling.
    """
    tolerance_pairs = tolerance_pairs or {}  # Default to empty dict if not provided

    combined_scores = []
    for gpu in GPU_TFLOPS_FP16.keys():
        fp16_theoretical = GPU_TFLOPS_FP16[gpu]
        fp32_theoretical = GPU_TFLOPS_FP32[gpu]
        avram_theoretical = GPU_AVRAM[gpu]

        fp16_deviation = abs(fp16_tflops - fp16_theoretical) / fp16_theoretical
        fp32_deviation = abs(fp32_tflops - fp32_theoretical) / fp32_theoretical
        avram_deviation = abs(estimated_avram - avram_theoretical) / avram_theoretical

        combined_score = (fp16_deviation + fp32_deviation + avram_deviation) / 3
        combined_scores.append((gpu, combined_score))
    
    # Sort by the lowest deviation
    identified_gpu = sorted(combined_scores, key=lambda x: x[1])[0][0]

    # Tolerance handling for nearly identical GPUs
    if reported_name:
        # Check if identified GPU matches the tolerance pair
        if identified_gpu in tolerance_pairs and reported_name == tolerance_pairs.get(identified_gpu):
            print(f"[Tolerance Adjustment] Detected GPU {identified_gpu} matches reported GPU {reported_name}.")
            identified_gpu = reported_name
        # Check if reported GPU matches the tolerance pair in reverse
        elif reported_name in tolerance_pairs and identified_gpu == tolerance_pairs.get(reported_name):
            print(f"[Tolerance Adjustment] Reported GPU {reported_name} matches detected GPU {identified_gpu}.")
            identified_gpu = reported_name

    return identified_gpu

def compute_script_hash(script_path):
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def send_script_and_request_hash(ssh_client, script_path):
    sftp = ssh_client.open_sftp()
    sftp.put(script_path, "/tmp/miner_script.py")
    sftp.close()

    # Command to compute the hash on the remote server
    hash_command = """
    python3 -c "
import hashlib
with open('/tmp/miner_script.py', 'rb') as f:
    computed_hash = hashlib.sha256(f.read()).hexdigest()
print(computed_hash)
"
    """
    stdin, stdout, stderr = ssh_client.exec_command(hash_command)
    computed_hash = stdout.read().decode().strip()
    hash_error = stderr.read().decode().strip()

    if hash_error:
        raise RuntimeError(f"Hash computation failed: {hash_error}")
    return computed_hash

def execute_script_on_miner(ssh_client, mode):
    execution_command = f"python3 /tmp/miner_script.py --mode {mode}"
    stdin, stdout, stderr = ssh_client.exec_command(execution_command)
    execution_output = stdout.read().decode().strip()
    execution_error = stderr.read().decode().strip()

    if execution_error:
        raise RuntimeError(f"Script execution failed: {execution_error}")
    return execution_output

def parse_benchmark_output(output):
    try:
        parts = output.strip().split()
        num_gpus = int(parts[0])  # First value is the number of GPUs
        vram = float(parts[1])
        size_fp16 = int(parts[2])
        time_fp16 = float(parts[3])
        size_fp32 = int(parts[4])
        time_fp32 = float(parts[5])
        return num_gpus, vram, size_fp16, time_fp16, size_fp32, time_fp32
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse execution output: {output}") from e

def parse_merkle_output(output):
    try:
        lines = output.strip().split('\n')
        root_hashes_line = None
        timings_line = None
        for line in lines:
            if line.startswith('Root hashes:'):
                root_hashes_line = line
            elif line.startswith('Timings:'):
                timings_line = line
        if root_hashes_line is None or timings_line is None:
            raise ValueError("Output does not contain root hashes or timings")
        # Parse root hashes
        root_hashes_str = root_hashes_line.split(': ', 1)[1]
        root_hashes = json.loads(root_hashes_str)  # List of tuples (gpu_id, root_hash)

        # Parse timings
        timings_str = timings_line.split(': ', 1)[1]
        gpu_timings = json.loads(timings_str)  # List of tuples (gpu_id, timings_dict)

        return root_hashes, gpu_timings
    except (ValueError, IndexError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse execution output: {output}") from e

def get_random_seeds(num_gpus):
    seeds = {}
    for gpu_id in range(num_gpus):
        s_A = secrets.randbits(64)
        s_B = secrets.randbits(64)
        seeds[gpu_id] = (s_A, s_B)
    return seeds

def send_seeds(ssh_client, seeds, n):
    lines = [str(n)]  # First line is n
    for gpu_id in seeds.keys():
        s_A, s_B = seeds[gpu_id]
        line = f"{gpu_id} {s_A} {s_B}"
        lines.append(line)
    content = '\n'.join(lines)
    command = f"echo '{content}' > /tmp/seeds.txt"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    stdout.channel.recv_exit_status()

def send_challenge_indices(ssh_client, indices):
    lines = []
    for gpu_id in indices.keys():
        idx_list = indices[gpu_id]
        indices_str = ';'.join([f"{i},{j}" for i, j in idx_list])
        line = f"{gpu_id} {indices_str}"
        lines.append(line)
    content = '\n'.join(lines)
    command = f"echo '{content}' > /tmp/challenge_indices.txt"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    stdout.channel.recv_exit_status()

def receive_responses(ssh_client, num_gpus):
    responses = {}
    try:
        with ssh_client.open_sftp() as sftp, tempfile.TemporaryDirectory() as temp_dir:
            for gpu_id in range(num_gpus):
                remote_path = f'/dev/shm/responses_gpu_{gpu_id}.npy'
                local_path = f'{temp_dir}/responses_gpu_{gpu_id}.npy'
                
                try:
                    sftp.get(remote_path, local_path)
                    response = np.load(local_path, allow_pickle=True)
                    responses[gpu_id] = response.item()
                except Exception as e:
                    print(f"Error processing GPU {gpu_id}: {e}")
                    responses[gpu_id] = None
    except Exception as e:
        print(f"SFTP connection error: {e}")
    
    return responses

def xorshift32_numpy(state):
    state = np.uint64(state)
    x = state & np.uint64(0xFFFFFFFF)
    x ^= (np.uint64((x << np.uint64(13)) & np.uint64(0xFFFFFFFF)))
    x ^= (np.uint64((x >> np.uint64(17)) & np.uint64(0xFFFFFFFF)))
    x ^= (np.uint64((x << np.uint64(5)) & np.uint64(0xFFFFFFFF)))
    x = x & np.uint64(0xFFFFFFFF)
    return x

def generate_prng_value(s, i, j):
    s = np.uint64(s)
    i = np.uint64(i % np.uint64(2**32))
    j = np.uint64(j)
    state = (s + i + j) & np.uint64(0xFFFFFFFF)

    for _ in range(10):
        state = xorshift32_numpy(state)

    return state / float(0xFFFFFFFF)

def verify_responses(seeds, root_hashes, responses, indices, n):
    """
    Verifies the responses from GPUs by checking computed values and Merkle proofs.

    Parameters:
        seeds (dict): Seeds used for generating PRNG values for each GPU.
        root_hashes (dict): Merkle root hashes for each GPU.
        responses (dict): Responses from each GPU containing computed rows and proofs.
        indices (dict): Challenge indices for each GPU.
        n (int): Total number of leaves in the Merkle tree.

    Returns:
        bool: True if verification passes within the allowed failure threshold, False otherwise.
    """
    verification_passed = True
    failed_gpus = []
    num_gpus = len(root_hashes.keys())

    # Define the minimum number of GPUs that must pass verification
    if num_gpus == 4:
        required_passes = 3
    elif num_gpus > 4:
        # For systems with more than 4 GPUs, adjust the required_passes as needed
        # Example: Require at least 75% to pass
        required_passes = int(np.ceil(0.75 * num_gpus))
    else:
        # For systems with 2 or fewer GPUs, require all to pass
        required_passes = num_gpus

    for gpu_id in root_hashes.keys():
        s_A, s_B = seeds[gpu_id]
        gpu_indices = indices[gpu_id]
        response = responses[gpu_id]
        root_hash = root_hashes[gpu_id]
        total_leaves = n

        gpu_failed = False  # Flag to track if the current GPU has failed

        for idx, (i, j) in enumerate(gpu_indices):
            # Generate only the necessary row and column entries using PRNG
            A_row = np.array([generate_prng_value(s_A, i, col) for col in range(n)], dtype=np.float32)
            B_col = np.array([generate_prng_value(s_B, row, j) for row in range(n)], dtype=np.float32)

            # Compute C_{i,j} as the dot product of A_row and B_col
            value_validator = np.dot(A_row, B_col)

            # Retrieve miner's computed value and corresponding Merkle proof
            row_miner = response['rows'][idx]
            proof = response['proofs'][idx]
            value_miner = row_miner[j]

            # Check if the miner's value matches the expected value
            if not np.isclose(value_miner, value_validator, atol=1e-5):
                print(f"[Verification] GPU {gpu_id}: Value mismatch at index ({i}, {j}).")
                gpu_failed = True
                break  # Exit the loop for this GPU as it has already failed

            # Verify the Merkle proof for the row
            if not verify_merkle_proof_row(row_miner, proof, bytes.fromhex(root_hash), i, total_leaves):
                print(f"[Verification] GPU {gpu_id}: Invalid Merkle proof at index ({i}).")
                gpu_failed = True
                break  # Exit the loop for this GPU as it has already failed

        if gpu_failed:
            failed_gpus.append(gpu_id)
            print(f"[Verification] GPU {gpu_id} failed verification.")
        else:
            print(f"[Verification] GPU {gpu_id} passed verification.")

    # Calculate the number of GPUs that passed verification
    passed_gpus = num_gpus - len(failed_gpus)

    # Determine if verification passes based on the required_passes
    if passed_gpus >= required_passes:
        verification_passed = True
        print(f"[Verification] SUCCESS: {passed_gpus} out of {num_gpus} GPUs passed verification.")
        if len(failed_gpus) > 0:
            print(f"            Note: {len(failed_gpus)} GPU(s) failed verification but within allowed threshold.")
    else:
        verification_passed = False
        print(f"[Verification] FAILURE: Only {passed_gpus} out of {num_gpus} GPUs passed verification.")
        if len(failed_gpus) > 0:
            print(f"            {len(failed_gpus)} GPU(s) failed verification which exceeds the allowed threshold.")

    return verification_passed

def verify_merkle_proof_row(row, proof, root_hash, index, total_leaves, hash_func=hashlib.sha256):
    """
    Verifies a Merkle proof for a given row.

    Parameters:
    - row (np.ndarray): The data row to verify.
    - proof (list of bytes): The list of sibling hashes required for verification.
    - root_hash (bytes): The root hash of the Merkle tree.
    - index (int): The index of the row in the tree.
    - total_leaves (int): The total number of leaves in the Merkle tree.
    - hash_func (callable): The hash function to use (default: hashlib.sha256).

    Returns:
    - bool: True if the proof is valid, False otherwise.
    """
    # Initialize the computed hash with the hash of the row using the specified hash function
    computed_hash = hash_func(row.tobytes()).digest()
    idx = index
    num_leaves = total_leaves
    
    # Iterate through each sibling hash in the proof
    for sibling_hash in proof:
        if idx % 2 == 0:
            # If the current index is even, concatenate computed_hash + sibling_hash
            combined = computed_hash + sibling_hash
        else:
            # If the current index is odd, concatenate sibling_hash + computed_hash
            combined = sibling_hash + computed_hash
        # Compute the new hash using the specified hash function
        computed_hash = hash_func(combined).digest()
        # Move up to the next level
        idx = idx // 2
    
    # Compare the computed hash with the provided root hash
    return computed_hash == root_hash

def adjust_matrix_size(vram, element_size=2, buffer_factor=0.8):
    usable_vram = vram * buffer_factor * 1e9  # Usable VRAM in bytes
    max_size = int((usable_vram / (2 * element_size)) ** 0.5)  # Max size fitting in VRAM
    aligned_size = (max_size // 32) * 32  # Ensure alignment to multiple of 32
    return aligned_size

def get_remote_gpu_info(ssh_client):
    """
    Execute the miner script in gpu_info mode to get GPU information from the remote miner.

    Args:
        ssh_client (paramiko.SSHClient): SSH client connected to the miner.

    Returns:
        dict: Dictionary containing GPU information (number and names).
    """
    command = "python3 /tmp/miner_script.py --mode gpu_info"
    stdin, stdout, stderr = ssh_client.exec_command(command)

    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()

    if error:
        raise RuntimeError(f"Failed to get GPU info: {error}")

    return json.loads(output)

if __name__ == "__main__":
    # Load configuration from YAML
    config_file = "config.yaml"
    config_data = load_yaml_config(config_file)

    # Extract GPU performance data
    gpu_data = config_data["gpu_performance"]
    GPU_TFLOPS_FP16 = gpu_data["GPU_TFLOPS_FP16"]
    GPU_TFLOPS_FP32 = gpu_data["GPU_TFLOPS_FP32"]
    GPU_AVRAM = gpu_data["GPU_AVRAM"]
    gpu_tolerance_pairs = gpu_data.get("gpu_tolerance_pairs", {})
    gpu_scores = gpu_data.get("gpu_scores", {})

    # Get the GPU with the maximum score
    max_gpu = max(gpu_scores, key=gpu_scores.get)
    max_score = gpu_scores[max_gpu]

    # Extract Merkle Proof Settings
    merkle_proof = config_data["merkle_proof"]
    time_tol = merkle_proof.get("time_tolerance",5)
    
    # Extract SSH configuration
    ssh_config = config_data["ssh_config"]
    ssh_host = ssh_config["host"]
    ssh_user = ssh_config["user"]
    ssh_port = ssh_config.get("port", 22)
    ssh_pw = ssh_config.get("password")
    miner_script_path = ssh_config["miner_script_path"]

    # Initialize
    num_gpus = 0
    try:
        start_time_total = time.time()
        print("=== GPU Verification Process Started ===\n")

        print("[Step 1] Starting Proof of GPU")
        # Compute the local hash
        local_hash = compute_script_hash(miner_script_path)
        print("[Step 1] Local script hash computed successfully.")
        print(f"         Local Hash: {local_hash}")

        print("[Step 2] Establishing SSH connection to the miner...")
        # Connect to the remote server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Attempt to connect with a 30-second timeout
            if ssh_pw:  # Check if a password is provided
                ssh_client.connect(
                    hostname=ssh_host, 
                    port=ssh_port, 
                    username=ssh_user, 
                    password=ssh_pw, 
                    timeout=30
                )
            else:
                ssh_client.connect(
                    hostname=ssh_host, 
                    port=ssh_port, 
                    username=ssh_user, 
                    timeout=30
                )
            print("[Step 2] SSH connection established.")

        except (SSHException, AuthenticationException) as e:
            print(f"[Error] SSH connection failed: {e}")
        except Exception as e:
            print(f"[Error] SSH connection failed, port is closed: {e}")
        finally:
            ssh_client.close()

        print("[Step 3] Sending miner script and requesting remote hash.")
        # Send the script and request the computed hash from the miner
        remote_hash = send_script_and_request_hash(ssh_client, miner_script_path)
        print("[Step 3] Remote script hash received.")
        print(f"         Remote Hash: {remote_hash}")

        if local_hash == remote_hash:
            print("[Integrity Check] SUCCESS: Local and remote hashes match.\n")
        else:
            print("[Integrity Check] FAILURE: Hash mismatch detected.")
            raise ValueError("Script integrity verification failed.")
        
        # Get GPU info from the remote miner
        # Get GPU info from the remote miner
        print("[Step 4] Retrieving GPU information (NVIDIA driver) from miner...")
        gpu_info = get_remote_gpu_info(ssh_client)
        num_gpus_reported = gpu_info["num_gpus"]
        gpu_name_reported = gpu_info["gpu_names"][0] if num_gpus_reported > 0 else None  # Take the first GPU name

        # Print GPU information
        print("[Step 4] Reported GPU Information:")
        if num_gpus_reported > 0:
            print(f"  Number of GPUs: {num_gpus_reported}")
            print(f"  GPU Type: {gpu_name_reported}")
            print()
        else:
            print("  No GPUs detected.")
            raise ValueError("No GPUs detected.")

        print("[Step 5] Executing benchmarking mode on the miner...")
        # Run the benchmarking mode
        execution_output = execute_script_on_miner(ssh_client, mode='benchmark')
        print("[Step 5] Benchmarking completed.")

        # Parse the execution output
        num_gpus, vram, size_fp16, time_fp16, size_fp32, time_fp32 = parse_benchmark_output(execution_output)
        
        print(f"[Benchmark Results] Detected {num_gpus} GPU(s) with {vram} GB unfractured VRAM.")
        print(f"                    FP16 - Matrix Size: {size_fp16}, Execution Time: {time_fp16} s")
        print(f"                    FP32 - Matrix Size: {size_fp32}, Execution Time: {time_fp32} s\n")

        # Calculate performance metrics
        fp16_tflops = (2 * size_fp16 ** 3) / time_fp16 / 1e12
        fp32_tflops = (2 * size_fp32 ** 3) / time_fp32 / 1e12

        print("[Performance Metrics] Calculated TFLOPS:")
        print(f"                    FP16: {fp16_tflops:.2f} TFLOPS")
        print(f"                    FP32: {fp32_tflops:.2f} TFLOPS\n")

        # Perform GPU identification based on TFLOPS
        gpu_name = identify_gpu(fp16_tflops, fp32_tflops, vram, gpu_name_reported, gpu_tolerance_pairs)
        print(f"[GPU Identification] Based on performance: {gpu_name}\n")

        print("[Step 6] Initiating Merkle Proof Mode.")
        # Run the Merkle proof mode
        # Step 1: Send seeds and execute compute mode
        n = adjust_matrix_size(vram, element_size=4, buffer_factor=0.10)
        seeds = get_random_seeds(num_gpus)
        send_seeds(ssh_client, seeds, n)
        print(f"[Step 6] Compute mode executed on miner - Matrix Size: {n}")
        start_time = time.time()
        execution_output = execute_script_on_miner(ssh_client, mode='compute')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"         Compute mode execution time: {elapsed_time:.2f} seconds.")

        # Parse the execution output
        root_hashes_list, gpu_timings_list = parse_merkle_output(execution_output)
        print("[Merkle Proof] Root hashes received from GPUs:")
        for gpu_id, root_hash in root_hashes_list:
            print(f"        GPU {gpu_id}: {root_hash}")

        # Calculate total times
        total_multiplication_time = 0.0
        total_merkle_tree_time = 0.0
        num_gpus = len(gpu_timings_list)

        # Sum up the times from each GPU
        for _, timing in gpu_timings_list:
            total_multiplication_time += timing.get('multiplication_time', 0.0)
            total_merkle_tree_time += timing.get('merkle_tree_time', 0.0)

        # Compute averages
        average_multiplication_time = total_multiplication_time / num_gpus if num_gpus > 0 else 0.0
        average_merkle_tree_time = total_merkle_tree_time / num_gpus if num_gpus > 0 else 0.0

        timing_passed = False
        if elapsed_time < time_tol + num_gpus * time_fp32 and average_multiplication_time < time_fp32:
            timing_passed = True

        # Print the average times
        print(f"Average Matrix Multiplication Time: {average_multiplication_time:.4f} seconds")
        print(f"Average Merkle Tree Time: {average_merkle_tree_time:.4f} seconds")
        print()

        # Convert root_hashes and timings to dictionaries
        root_hashes = {gpu_id: root_hash for gpu_id, root_hash in root_hashes_list}
        gpu_timings = {gpu_id: timing for gpu_id, timing in gpu_timings_list}
        n = gpu_timings[0]['n']  # Assuming same n for all GPUs

        # Step 2: Select challenge indices and send to miner
        indices = {}
        num_indices = 1  # Adjust for security level
        for gpu_id in range(num_gpus):
            indices[gpu_id] = [(np.random.randint(0, n), np.random.randint(0, n)) for _ in range(num_indices)]
        send_challenge_indices(ssh_client, indices)

        # Step 3: Execute proof mode
        execution_output = execute_script_on_miner(ssh_client, mode='proof')
        print("[Merkle Proof] Proof mode executed on miner.")

        # Step 4: Receive responses and verify
        responses = receive_responses(ssh_client, num_gpus)
        print("[Merkle Proof] Responses received from miner.")

        # Verify responses
        verification_passed = verify_responses(seeds, root_hashes, responses, indices, n)
        print()

        if verification_passed and timing_passed:
            print("=" * 50)
            print("[Verification] SUCCESS")
            print("  Merkle Proof: PASSED")
            print(f"  GPU Identification: Detected {num_gpus} x {gpu_name} GPU(s)")
            print("=" * 50)
        else:
            print("=" * 50)
            print("[Verification] FAILURE")
            print("  Merkle Proof: FAILED")
            print("  GPU Identification: Aborted due to verification failure")
            print("=" * 50)
        
        # Scoring
        # Get the GPU with the maximum score
        max_gpu = max(gpu_scores, key=gpu_scores.get)
        max_score = gpu_scores[max_gpu]*8
        score_factor = 100/max_score

        # Get GPU score
        score = gpu_scores.get(gpu_name) * num_gpus * score_factor

        print(f"Score: {score:.1f}/{100}")

        total_time = time.time() - start_time_total
        print(f"Total time: {total_time:.2f} seconds.\n")

        print("=== GPU Verification Process Completed ===\n")

    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
    finally:
        ssh_client.close()
        # Clean up local temporary files
        for gpu_id in range(num_gpus):
            response_file = f"responses_gpu_{gpu_id}.npy"
            if os.path.exists(response_file):
                os.remove(response_file)