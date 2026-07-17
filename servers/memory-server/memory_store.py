"""ChromaDB 记忆存储封装。所有方法接收调用方算好的 embedding。"""
import os
import uuid
import math
from datetime import datetime, timezone

import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")
MAX_MEMORIES_PER_USER = 200  # 每用户容量上限，超出时惰性清理

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name="memories", metadata={"hnsw:space": "cosine"}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_user_id(user_id: str) -> None:
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 100:
        raise ValueError("user_id 必须是 1 到 100 个字符")


def _validate_content(content: str) -> None:
    if not isinstance(content, str) or not content.strip() or len(content) > 2000:
        raise ValueError("记忆内容必须是 1 到 2000 个字符")


def add_memory(content: str, embedding: list[float], user_id: str,
               importance: float = 0.5, source: str = "auto") -> dict:
    """新增一条记忆，返回记忆 dict。写入后触发容量清理。"""
    _validate_content(content)
    _validate_user_id(user_id)
    if not math.isfinite(importance) or not 0 <= importance <= 1:
        raise ValueError("importance 必须在 0 到 1 之间")
    mem_id = str(uuid.uuid4())
    meta = {
        "user_id": user_id,
        "created_at": _now(),
        "updated_at": _now(),
        "importance": float(importance),
        "source": source,
    }
    _collection.add(ids=[mem_id], embeddings=[embedding],
                    documents=[content], metadatas=[meta])
    _evict_if_needed(user_id)
    return {"id": mem_id, "content": content, **meta}


def update_memory(mem_id: str, content: str, embedding: list[float], user_id: str,
                  importance: float | None = None) -> bool:
    """更新已有记忆的内容与向量。"""
    _validate_content(content)
    _validate_user_id(user_id)
    if importance is not None and (
        not math.isfinite(importance) or not 0 <= importance <= 1
    ):
        raise ValueError("importance 必须在 0 到 1 之间")
    existing = _collection.get(ids=[mem_id])
    if not existing["ids"]:
        return False
    meta = existing["metadatas"][0]
    if meta.get("user_id") != user_id:
        return False
    meta["updated_at"] = _now()
    if importance is not None:
        meta["importance"] = float(importance)
    _collection.update(ids=[mem_id], embeddings=[embedding],
                       documents=[content], metadatas=[meta])
    return True


def delete_memory(mem_id: str, user_id: str) -> bool:
    _validate_user_id(user_id)
    existing = _collection.get(ids=[mem_id])
    if not existing["ids"]:
        return False
    if existing["metadatas"][0].get("user_id") != user_id:
        return False
    _collection.delete(ids=[mem_id])
    return True


def search_memories(query_embedding: list[float], user_id: str,
                    top_k: int = 5) -> list[dict]:
    """按向量相似度检索该用户的记忆，返回带 score 的列表。"""
    _validate_user_id(user_id)
    top_k = min(max(int(top_k), 1), 20)
    count = _collection.count()
    if count == 0:
        return []
    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        where={"user_id": user_id},
    )
    memories = []
    for i, mem_id in enumerate(result["ids"][0]):
        meta = result["metadatas"][0][i]
        # cosine 距离转相似度
        score = 1.0 - result["distances"][0][i]
        memories.append({
            "id": mem_id,
            "content": result["documents"][0][i],
            "score": round(score, 4),
            **meta,
        })
    return memories


def list_memories(user_id: str) -> list[dict]:
    """列出该用户全部记忆，按 importance 降序。"""
    _validate_user_id(user_id)
    result = _collection.get(where={"user_id": user_id})
    memories = []
    for i, mem_id in enumerate(result["ids"]):
        meta = result["metadatas"][i]
        memories.append({
            "id": mem_id,
            "content": result["documents"][i],
            **meta,
        })
    memories.sort(key=lambda m: m.get("importance", 0), reverse=True)
    return memories


def _evict_if_needed(user_id: str) -> None:
    """容量超限时删除 importance 最低的记忆（遗忘策略）。"""
    mems = list_memories(user_id)
    if len(mems) <= MAX_MEMORIES_PER_USER:
        return
    for m in mems[MAX_MEMORIES_PER_USER:]:
        _collection.delete(ids=[m["id"]])
