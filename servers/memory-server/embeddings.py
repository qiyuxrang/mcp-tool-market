"""中转站 Embedding API 客户端（OpenAI 兼容协议）。"""
import os

from openai import AsyncOpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:3000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-not-set")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

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
