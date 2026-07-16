"""ChromaDB 知识库：文档切块、覆盖式写入和向量检索。"""
import os
import uuid
from datetime import datetime, timezone

import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_data")

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name="knowledge", metadata={"hnsw:space": "cosine"}
)


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """按字符滑窗切块；中文无需额外分词依赖。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 overlap，且 overlap 不能为负数")
    if len(text) > 100_000:
        raise ValueError("单篇文档不能超过 10 万字符")
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    step = chunk_size - overlap
    return [cleaned[start:start + chunk_size] for start in range(0, len(cleaned), step)]


def replace_document(source: str, chunks: list[str], vectors: list[list[float]]) -> int:
    """用新切块覆盖同名来源，返回写入数量。"""
    if not source.strip():
        raise ValueError("source 不能为空")
    if len(source) > 200:
        raise ValueError("source 不能超过 200 个字符")
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("chunks 与 vectors 必须非空且数量一致")

    _collection.delete(where={"source": source})
    now = datetime.now(timezone.utc).isoformat()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": source, "chunk_index": i, "created_at": now}
        for i in range(len(chunks))
    ]
    _collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def search(
    query_vector: list[float], top_k: int = 5, min_score: float | None = None
) -> list[dict]:
    """返回最相关的知识片段及来源。"""
    count = _collection.count()
    if count == 0:
        return []
    result = _collection.query(
        query_embeddings=[query_vector],
        n_results=min(max(top_k, 1), count),
    )
    matches = [
        {
            "content": result["documents"][0][i],
            "source": result["metadatas"][0][i]["source"],
            "chunk_index": result["metadatas"][0][i]["chunk_index"],
            "score": round(1.0 - result["distances"][0][i], 4),
        }
        for i in range(len(result["ids"][0]))
    ]
    if min_score is not None:
        matches = [item for item in matches if item["score"] >= min_score]
    return matches
