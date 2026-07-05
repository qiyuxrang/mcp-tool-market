import os

from mcp import ClientSession
from mcp.client.sse import sse_client

# 服务器地址支持环境变量覆盖：Docker Compose 中需指向服务名而非 localhost
SERVER_REGISTRY = {
    "file":       {"url": os.getenv("FILE_SERVER_URL", "http://localhost:8001"), "port": 8001},
    "weather":    {"url": os.getenv("WEATHER_SERVER_URL", "http://localhost:8002"), "port": 8002},
    "calculator": {"url": os.getenv("CALCULATOR_SERVER_URL", "http://localhost:8003"), "port": 8003},
    "database":   {"url": os.getenv("DATABASE_SERVER_URL", "http://localhost:8004"), "port": 8004},
    "memory":     {"url": os.getenv("MEMORY_SERVER_URL", "http://localhost:8005"), "port": 8005},
}

_STATUS_LABELS = {
    "file": "文件系统",
    "weather": "天气查询",
    "calculator": "计算器",
    "database": "数据库查询",
    "memory": "记忆系统",
}


class MCPClientManager:
    """Manages connections to multiple MCP Servers."""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.streams: dict[str, tuple] = {}
        self.context_managers: dict[str, object] = {}
        self.status: dict[str, str] = {}
        self.tools_cache: dict[str, list] = {}
        for name in SERVER_REGISTRY:
            self.status[name] = "disconnected"
            self.tools_cache[name] = []

    async def connect(self, name: str) -> bool:
        """Connect to an MCP Server by name using SSE transport."""
        if name not in SERVER_REGISTRY:
            return False
        try:
            url = f"{SERVER_REGISTRY[name]['url']}/sse"
            cm = sse_client(url)
            streams = await cm.__aenter__()
            session = ClientSession(streams[0], streams[1])
            await session.__aenter__()
            await session.initialize()
            tools_result = await session.list_tools()
            self.tools_cache[name] = tools_result.tools
            self.sessions[name] = session
            self.streams[name] = streams
            self.context_managers[name] = cm
            self.status[name] = "connected"
            return True
        except Exception as e:
            self.status[name] = f"error: {e}"
            return False

    async def disconnect(self, name: str) -> bool:
        """Disconnect from an MCP Server and clean up resources."""
        was_connected = name in self.sessions
        if name in self.sessions:
            try:
                session = self.sessions[name]
                await session.__aexit__(None, None, None)
            except Exception:
                pass
            finally:
                del self.sessions[name]
            try:
                cm = self.context_managers.pop(name, None)
                if cm:
                    await cm.__aexit__(None, None, None)
            except Exception:
                pass
            self.streams.pop(name, None)
        self.status[name] = "disconnected"
        self.tools_cache[name] = []
        return was_connected

    async def call_tool(self, name: str, tool_name: str, arguments: dict) -> str:
        """Call a tool on a connected MCP Server and return the text output."""
        if name not in self.sessions:
            return f"Error: Server '{name}' is not connected"
        try:
            session = self.sessions[name]
            result = await session.call_tool(tool_name, arguments)
            texts = []
            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
            return "".join(texts)
        except Exception as e:
            return f"Error calling {name}/{tool_name}: {e}"

    def get_all_tools_spec(self) -> list[dict]:
        """Return tool definitions from all connected servers as a list of dicts,
        each augmented with a 'server' key identifying the source server name."""
        tools = []
        for name, tool_list in self.tools_cache.items():
            if self.status.get(name) != "connected":
                continue
            for tool in tool_list:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                    "server": name,
                })
        return tools

    def get_status(self) -> dict:
        """Return the connection status of all servers with Chinese labels."""
        return {
            name: {
                "name": name,
                "label": _STATUS_LABELS.get(name, name),
                "status": self.status.get(name, "disconnected"),
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                    for t in self.tools_cache.get(name, [])
                ],
            }
            for name in SERVER_REGISTRY
        }
