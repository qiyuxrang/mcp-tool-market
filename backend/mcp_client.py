import os
from contextlib import asynccontextmanager

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

_MARKET_METADATA = {
    "file": {
        "category": "Workspace",
        "risk": "受限写入",
        "summary": "在沙箱目录中读写文件，阻止目录穿越和超大文件。",
        "permissions": ["读写 workspace", "1MB 文件上限", "禁止沙箱外路径"],
        "examples": ["读取 notes.md", "写入 analysis/result.txt"],
    },
    "weather": {
        "category": "Web API",
        "risk": "只读网络",
        "summary": "通过 wttr.in 查询实时天气和短期预报。",
        "permissions": ["外部 HTTP 请求", "不写本地文件"],
        "examples": ["查询北京天气", "查询上海三日预报"],
    },
    "calculator": {
        "category": "Local Compute",
        "risk": "本地纯计算",
        "summary": "AST 白名单求值，支持数学计算和常见单位换算。",
        "permissions": ["无网络权限", "无文件权限", "限制表达式复杂度"],
        "examples": ["计算 (2+3)*4", "换算 2.5kg 为磅"],
    },
    "database": {
        "category": "Data",
        "risk": "只读查询",
        "summary": "查询示例 SQLite 数据库，只允许 SELECT 并限制返回行数。",
        "permissions": ["SELECT-only", "最多返回 100 行", "禁止写入语句"],
        "examples": ["列出数据表", "查询库存最低产品"],
    },
    "memory": {
        "category": "RAG",
        "risk": "用户级记忆",
        "summary": "保存长期记忆，索引知识库，并按来源检索片段。",
        "permissions": ["按 user_id 逻辑分区", "ChromaDB 持久化", "检索片段带来源"],
        "examples": ["存入员工手册", "按来源回答报销标准"],
    },
}


class MCPClientManager:
    """Manages connections to multiple MCP Servers."""

    def __init__(self):
        self.status: dict[str, str] = {}
        self.tools_cache: dict[str, list] = {}
        for name in SERVER_REGISTRY:
            self.status[name] = "disconnected"
            self.tools_cache[name] = []

    @asynccontextmanager
    async def _session(self, name: str):
        """Open one task-local MCP session and always close it in that task."""
        url = f"{SERVER_REGISTRY[name]['url']}/sse"
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session

    async def connect(self, name: str) -> bool:
        """Connect to an MCP Server by name using SSE transport."""
        if name not in SERVER_REGISTRY:
            return False
        if self.status.get(name) == "connected":
            return True
        try:
            async with self._session(name) as session:
                tools_result = await session.list_tools()
            self.tools_cache[name] = tools_result.tools
            self.status[name] = "connected"
            return True
        except Exception as e:
            self.status[name] = f"error: {e}"
            return False

    async def disconnect(self, name: str) -> bool:
        """Mark a server disconnected and clear its cached tools."""
        if name not in SERVER_REGISTRY:
            return False
        was_connected = self.status.get(name) == "connected"
        self.status[name] = "disconnected"
        self.tools_cache[name] = []
        return was_connected

    async def call_tool(self, name: str, tool_name: str, arguments: dict) -> str:
        """Call a tool on a connected MCP Server and return the text output."""
        if self.status.get(name) != "connected":
            return f"Error: Server '{name}' is not connected"
        try:
            # ponytail: per-call sessions avoid cross-task SSE cleanup bugs;
            # use dedicated connection workers only if handshake latency matters.
            async with self._session(name) as session:
                result = await session.call_tool(tool_name, arguments)
            texts = []
            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
            return "".join(texts)
        except Exception as e:
            self.status[name] = f"error: {e}"
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
                **_MARKET_METADATA.get(name, {}),
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                    for t in self.tools_cache.get(name, [])
                ],
            }
            for name in SERVER_REGISTRY
        }
