"""记忆系统验证脚本。
用法:
  python verify_memory.py store     # 只验证存储层（无需API）
  python verify_memory.py pipeline  # 验证完整流水线（需要中转站API环境变量）
"""
import asyncio
import sys

import memory_store


def fake_vec(seed: float) -> list[float]:
    """确定性假向量（384维），用于无 API 环境验证存储层。"""
    return [(seed * (i + 1)) % 1.0 for i in range(384)]


def verify_store():
    uid = "verify-user"
    # 清理旧数据
    for m in memory_store.list_memories(uid):
        memory_store.delete_memory(m["id"], uid)

    m1 = memory_store.add_memory("用户在西安", fake_vec(0.1), uid, importance=0.7)
    m2 = memory_store.add_memory("用户喜欢摄影", fake_vec(0.5), uid, importance=0.9)
    assert len(memory_store.list_memories(uid)) == 2, "add failed"

    results = memory_store.search_memories(fake_vec(0.1), uid, top_k=1)
    assert results and results[0]["id"] == m1["id"], "search failed"

    ok = memory_store.update_memory(m1["id"], "用户搬到杭州了", fake_vec(0.2), uid)
    assert ok, "update failed"
    listed = memory_store.list_memories(uid)
    contents = [m["content"] for m in listed]
    assert "用户搬到杭州了" in contents, "update content not applied"

    assert memory_store.delete_memory(m2["id"], uid), "delete failed"
    assert len(memory_store.list_memories(uid)) == 1, "delete count wrong"

    # 清理
    for m in memory_store.list_memories(uid):
        memory_store.delete_memory(m["id"], uid)
    print("[PASS] store layer OK")


async def verify_pipeline():
    import pipeline
    uid = "verify-pipeline-user"
    for m in memory_store.list_memories(uid):
        memory_store.delete_memory(m["id"], uid)

    convo = [
        {"role": "user", "content": "你好，我叫小张，我对花生过敏，帮我记住"},
        {"role": "assistant", "content": "好的小张，我记住了你对花生过敏。"},
    ]
    ops = await pipeline.run_pipeline(convo, uid)
    print("operations:", ops)
    assert any(op["action"] == "ADD" for op in ops), "no ADD op"

    print("[PASS] pipeline OK")
    print("final memories:", [m["content"] for m in memory_store.list_memories(uid)])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "store"
    if mode == "store":
        verify_store()
    else:
        asyncio.run(verify_pipeline())
