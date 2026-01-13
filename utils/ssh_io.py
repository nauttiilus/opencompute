# utils/ssh_io.py
from __future__ import annotations

from typing import Tuple
import paramiko  # pip install paramiko

def ssh_connect(host: str, port: int, username: str, password: str, timeout: int = 25):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=username, password=password, timeout=timeout)
    return c


def ssh_connect_key(host: str, port: int, username: str, private_key: paramiko.PKey, timeout: int = 25):
    """Connect to SSH using a private key (Ed25519/RSA) instead of password."""
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=username, pkey=private_key, timeout=timeout)
    return c


def generate_ssh_keypair():
    """Generate an RSA SSH keypair. Returns (private_key_obj, public_key_string)."""
    from paramiko import RSAKey

    # Generate new RSA key (2048 bits is standard)
    key = RSAKey.generate(2048)

    # Get public key in OpenSSH format
    public_key_str = f"ssh-rsa {key.get_base64()} hwcheck@nodexo"

    return key, public_key_str

def ssh_close(c) -> None:
    c.close()

def scp_put(c, local_path: str, remote_path: str):
    s = c.open_sftp()
    try:
        s.put(local_path, remote_path)
    finally:
        s.close()

def scp_put_text(c, remote_path: str, text: str):
    s = c.open_sftp()
    try:
        with s.file(remote_path, "w") as f:
            f.write(text)
    finally:
        s.close()

def scp_get(c, remote_path: str, local_path: str):
    s = c.open_sftp()
    try:
        s.get(remote_path, local_path)
    finally:
        s.close()

def ssh_exec(c, cmd: str, timeout: int = 120) -> Tuple[int, str, str]:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err
