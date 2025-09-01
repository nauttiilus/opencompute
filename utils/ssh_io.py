# utils/ssh_io.py
from __future__ import annotations

from typing import Tuple
import paramiko  # pip install paramiko

def ssh_connect(host: str, port: int, username: str, password: str, timeout: int = 25):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=username, password=password, timeout=timeout)
    return c

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
