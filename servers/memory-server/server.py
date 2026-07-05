"""记忆系统 MCP Server：MCP 工具（AI主动模式）+ REST API（系统自动模式/前端CRUD）。"""
import json
import os

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

import embeddings
import memory_store
import pipeline

mcp = FastMCP("Memory Server")


# ------------------------- MCP 工具（AI 主动调用） -------------------------

@mcp.tool()
async def save_memory(content: str, user_id: str = "default") -> str:
    """保存一条重要信息到长期记忆。当用户明确要求记住某事时调用。

    Args:
        content: 要记住的内容（一句话事实）
        user_id: 用户标识
    """
    try:
        emb = await embeddings.embed_text(content)
        mem = memory_store.add_memory(content, emb, user_id,
                                      importance=0.8, source="manual")
        return f"已记住: {content} (id={mem['id']})"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def search_memory(query: str, user_id: str = "default", top_k: int = 5) -> str:
    """按语义搜索用户的长期记忆。需要回忆用户相关信息时调用。

    Args:
        query: 检索内容
        user_id: 用户标识
        top_k: 返回条数
    """
    try:
        emb = await embeddings.embed_text(query)
        results = memory_store.search_memories(emb, user_id, top_k=top_k)
        if not results:
            return "没有找到相关记忆"
        return json.dumps(
            [{"content": m["content"], "score": m["score"]} for m in results],
            ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def list_memories(user_id: str = "default") -> str:
    """列出用户的全部长期记忆。"""
    try:
        mems = memory_store.list_memories(user_id)
        if not mems:
            return "该用户还没有任何记忆"
        return json.dumps(
            [{"content": m["content"], "importance": m["importance"]}
             for m in mems],
            ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


# ------------------------- REST API（后端/前端直连） -------------------------

@mcp.custom_route("/memories", methods=["GET"])
async def http_list(request: Request) -> JSONResponse:
    user_id = request.query_params.get("user_id", "default")
    try:
        return JSONResponse({"memories": memory_store.list_memories(user_id)})
    except Exception as e:
        return JSONResponse({"memories": [], "error": str(e)}, status_code=500)


@mcp.custom_route("/memories/search", methods=["GET"])
async def http_search(request: Request) -> JSONResponse:
    user_id = request.query_params.get("user_id", "default")
    query = request.query_params.get("q", "")
    top_k = int(request.query_params.get("top_k", 5))
    if not query:
        return JSONResponse({"results": []})
    try:
        emb = await embeddings.embed_text(query)
        results = memory_store.search_memories(emb, user_id, top_k=top_k)
        return JSONResponse({"results": results})
    except Exception as e:
        return JSONResponse({"results": [], "error": str(e)}, status_code=500)


@mcp.custom_route("/memories/extract", methods=["POST"])
async def http_extract(request: Request) -> JSONResponse:
    """接收对话，运行双阶段流水线。由后端 Agent Engine 对话结束后调用。"""
    try:
        body = await request.json()
        conversation = body.get("conversation", [])
        user_id = body.get("user_id", "default")
        operations = await pipeline.run_pipeline(conversation, user_id)
        return JSONResponse({"operations": operations})
    except Exception as e:
        return JSONResponse({"operations": [], "error": str(e)}, status_code=500)


@mcp.custom_route("/memories/{mem_id}", methods=["DELETE"])
async def http_delete(request: Request) -> JSONResponse:
    mem_id = request.path_params.get("mem_id", "")
    try:
        ok = memory_store.delete_memory(mem_id)
        return JSONResponse({"success": ok})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8005))
    import uvicorn
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
