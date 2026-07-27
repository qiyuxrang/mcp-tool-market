"""Embedding 客户端（OpenAI 兼容协议）。

默认走本地 Ollama 的 bge-m3，无外部 API 依赖；如需切回中转站，
设置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL 即可。
"""
import os

from openai import AsyncOpenAI

# 与 chat 用的 OPENAI_BASE_URL 解耦：embedding 走本地 Ollama
BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("EMBEDDING_API_KEY", "ollama")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


async def embed_text(text: str) -> list[float]:
    """将单条文本向量化。"""
    resp = await _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化，保持输入顺序。"""
    if not texts:
        return []
    resp = await _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]
