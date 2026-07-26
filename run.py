#!/usr/bin/env python3
"""Dev launcher: starts all MCP servers + backend as subprocesses."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Windows console encoding fix
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent
SERVERS = [
    ("file-server", 8001),
    ("weather-server", 8002),
    ("calculator-server", 8003),
    ("db-server", 8004),
    ("memory-server", 8005),
]
processes = []


def load_env(path: Path) -> None:
    """Load the simple KEY=VALUE format used by backend/.env."""
    if not path.is_file():
        return
    # ponytail: this covers the checked-in template; use python-dotenv only if
    # multiline values or variable interpolation become necessary.
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def wait_for_port(port: int, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_open(port):
            return
        time.sleep(0.1)
    raise TimeoutError(f"端口 {port} 未在 {timeout} 秒内就绪")


def stop_processes() -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main():
    load_env(BASE / "backend" / ".env")
    backend_port = int(os.getenv("BACKEND_PORT", "8000"))
    processes.clear()
    print("MCP 工具市场 — 开发模式启动")
    print("-" * 40)

    try:
        for name, port in SERVERS:
            if port_is_open(port):
                raise RuntimeError(f"端口 {port} 已被占用")
            server_dir = BASE / "servers" / name
            env = os.environ.copy()
            env["PORT"] = str(port)
            env["HOST"] = "127.0.0.1"
            proc = subprocess.Popen(
                [sys.executable, "server.py"], cwd=server_dir, env=env
            )
            processes.append(proc)
            wait_for_port(port)
            if proc.poll() is not None:
                raise RuntimeError(f"{name} 启动失败")
            print(f"  OK {name} (:{port}) — PID {proc.pid}")

        if port_is_open(backend_port):
            raise RuntimeError(f"端口 {backend_port} 已被占用")
        backend_dir = BASE / "backend"
        backend = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app:app",
                "--host", "127.0.0.1", "--port", str(backend_port),
            ],
            cwd=backend_dir,
        )
        processes.append(backend)
        wait_for_port(backend_port)
        if backend.poll() is not None:
            raise RuntimeError("backend 启动失败")
        print(f"  OK backend (:{backend_port}) — PID {backend.pid}")
        print("-" * 40)
        print(f"Open browser: http://localhost:{backend_port}")
        print("Press Ctrl+C to stop all services")
        backend.wait()
    except KeyboardInterrupt:
        print("\nStopping all services...")
    finally:
        stop_processes()
        print("All services stopped")

if __name__ == "__main__":
    main()
