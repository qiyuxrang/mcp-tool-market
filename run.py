#!/usr/bin/env python3
"""Dev launcher: starts all MCP servers + backend as subprocesses."""
import subprocess, sys, os, time
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

def main():
    print("MCP 工具市场 — 开发模式启动")
    print("-" * 40)

    # Start all MCP servers
    for name, port in SERVERS:
        server_dir = BASE / "servers" / name
        env = os.environ.copy()
        env["PORT"] = str(port)
        p = subprocess.Popen([sys.executable, "server.py"], cwd=server_dir, env=env)
        processes.append(p)
        print(f"  OK {name} (:{port}) — PID {p.pid}")
        time.sleep(1.5)

    # Start backend
    backend_dir = BASE / "backend"
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=backend_dir,
    )
    processes.append(p)
    print(f"  OK backend (:8000) — PID {p.pid}")
    print("-" * 40)
    print("Open browser: http://localhost:8000")
    print("Press Ctrl+C to stop all services")

    try:
        p.wait()
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    proc.kill()
                except Exception:
                    pass
        print("All services stopped")

if __name__ == "__main__":
    main()
