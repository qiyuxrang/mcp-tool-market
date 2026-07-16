import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mcp_client import MCPClientManager, SERVER_REGISTRY
from agent_engine import AgentEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    user_id: str = Field(default="default", min_length=1, max_length=100)


class ConnectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=50)


class TestToolRequest(BaseModel):
    server: str = Field(min_length=1, max_length=50)
    tool: str = Field(min_length=1, max_length=100)
    arguments: dict = Field(default_factory=dict)
    user_id: str = Field(default="default", min_length=1, max_length=100)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup: nothing to do — servers connect on demand
    yield
    await agent.close()
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
        arguments = agent.bind_user_scope(req.server, req.tool, req.arguments, req.user_id)
        result = await mcp_client.call_tool(req.server, req.tool, arguments)
        return {"result": result}
    except Exception as e:
        return {"result": "", "error": str(e)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream a conversation with the AI agent via SSE."""

    async def event_stream():
        try:
            messages = [message.model_dump() for message in req.messages]
            async for event in agent.chat(messages, req.user_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Unhandled chat stream error")
            event = {"type": "final", "content": "服务暂时不可用，请稍后重试。"}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8005")


@app.get("/api/memories")
async def api_list_memories(
    user_id: str = Query("default", min_length=1, max_length=100),
):
    """代理 memory-server：列出该用户全部记忆。"""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f"{MEMORY_SERVER_URL}/memories",
                                  params={"user_id": user_id})
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Memory list proxy failed: %s", e)
        raise HTTPException(status_code=502, detail="记忆服务不可用")


@app.get("/api/memories/search")
async def api_search_memories(
    q: str = Query("", max_length=500),
    user_id: str = Query("default", min_length=1, max_length=100),
):
    """代理 memory-server：搜索记忆。"""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f"{MEMORY_SERVER_URL}/memories/search",
                                  params={"q": q, "user_id": user_id})
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Memory search proxy failed: %s", e)
        raise HTTPException(status_code=502, detail="记忆服务不可用")


@app.delete("/api/memories/{mem_id}")
async def api_delete_memory(
    mem_id: UUID,
    user_id: str = Query("default", min_length=1, max_length=100),
):
    """代理 memory-server：删除单条记忆。"""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.delete(
                f"{MEMORY_SERVER_URL}/memories/{mem_id}", params={"user_id": user_id}
            )
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Memory delete proxy failed: %s", e)
        raise HTTPException(status_code=502, detail="记忆服务不可用")


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
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=port)
