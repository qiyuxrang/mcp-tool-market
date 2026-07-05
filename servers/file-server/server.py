import json
import os

from mcp.server.fastmcp import FastMCP

SANDBOX = os.path.abspath("./workspace")
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB


def safe_path(path: str) -> str:
    """Resolve the full path and ensure it is within the sandbox directory."""
    if ".." in path.split(os.sep) or ".." in path.split("/"):
        raise ValueError(f"Path traversal detected in: {path}")
    resolved = os.path.abspath(os.path.join(SANDBOX, path))
    if not resolved.startswith(SANDBOX + os.sep) and resolved != SANDBOX:
        raise ValueError(f"Access denied: path is outside the sandbox: {resolved}")
    return resolved


# Create sandbox directory on startup
os.makedirs(SANDBOX, exist_ok=True)

mcp = FastMCP("file-server")


@mcp.tool()
def read_file(path: str) -> str:
    """Read the content of a file within the sandbox."""
    try:
        full_path = safe_path(path)
        if not os.path.isfile(full_path):
            return f"File not found: {path}"
        file_size = os.path.getsize(full_path)
        if file_size > MAX_FILE_SIZE:
            return f"File too large: {file_size} bytes (max {MAX_FILE_SIZE} bytes)"
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file within the sandbox."""
    try:
        full_path = safe_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        data = content.encode("utf-8")
        with open(full_path, "wb") as f:
            f.write(data)
        return f"Written {len(data)} bytes to {path}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def list_dir(path: str = "") -> str:
    """List directory contents within the sandbox."""
    try:
        full_path = safe_path(path) if path else SANDBOX
        if not os.path.isdir(full_path):
            return f"Not a directory: {path or '.'}"
        entries = []
        for entry in sorted(os.listdir(full_path)):
            entry_path = os.path.join(full_path, entry)
            entry_type = "file" if os.path.isfile(entry_path) else "dir"
            try:
                size = os.path.getsize(entry_path) if os.path.isfile(entry_path) else 0
            except OSError:
                size = 0
            entries.append({"name": entry, "type": entry_type, "size": size})
        return json.dumps(entries, ensure_ascii=False)
    except ValueError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    import uvicorn
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
