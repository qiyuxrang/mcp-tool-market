import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mcp_client import MCPClientManager, SERVER_REGISTRY
from agent_engine import AgentEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    messages: list[dict]
    user_id: str = "default"


class ConnectRequest(BaseModel):
    name: str


class TestToolRequest(BaseModel):
    server: str
    tool: str
    arguments: dict = {}


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup: nothing to do — servers connect on demand
    yield
    # Shutdown: disconnect every known server
    for name in SERVER_REGISTRY:
        try:
            await mcp_client.disconnect(name)
        except Exception:
            logger.exception("Error disconnecting %s during shutdown", name)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="MCP Tool Market", lifespan=lifespan)

mcp_client = MCPClientManager()
agent = AgentEngine(mcp_client)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/tools")
async def list_tools():
    """Return connection status of all known MCP servers."""
    return mcp_client.get_status()


@app.post("/api/tools/connect")
async def connect_server(req: ConnectRequest):
    """Connect to an MCP server by name."""
    try:
        ok = await mcp_client.connect(req.name)
        return {"success": ok, "status": mcp_client.status.get(req.name, "unknown")}
    except Exception as e:
        return {"success": False, "status": f"error: {e}"}


@app.post("/api/tools/disconnect")
async def disconnect_server(req: ConnectRequest):
    """Disconnect from an MCP server by name."""
    try:
        ok = await mcp_client.disconnect(req.name)
        return {"success": ok}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/tools/test")
async def test_tool(req: TestToolRequest):
    """Call a tool on a connected MCP server and return the result."""
    try:
        result = await mcp_client.call_tool(req.server, req.tool, req.arguments)
        return {"result": result}
    except Exception as e:
        return {"result": "", "error": str(e)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream a conversation with the AI agent via SSE."""

    async def event_stream():
        async for event in agent.chat(req.messages, req.user_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


import httpx as _httpx

MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8005")


@app.get("/api/memories")
async def api_list_memories(user_id: str = "default"):
    """代理 memory-server：列出该用户全部记忆。"""
    try:
        async with _httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f"{MEMORY_SERVER_URL}/memories",
                                  params={"user_id": user_id})
            return resp.json()
    except Exception as e:
        return {"memories": [], "error": str(e)}


@app.get("/api/memories/search")
async def api_search_memories(q: str = "", user_id: str = "default"):
    """代理 memory-server：搜索记忆。"""
    try:
        async with _httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f"{MEMORY_SERVER_URL}/memories/search",
                                  params={"q": q, "user_id": user_id})
            return resp.json()
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.delete("/api/memories/{mem_id}")
async def api_delete_memory(mem_id: str):
    """代理 memory-server：删除单条记忆。"""
    try:
        async with _httpx.AsyncClient(timeout=10) as http:
            resp = await http.delete(f"{MEMORY_SERVER_URL}/memories/{mem_id}")
            return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
