"""知识库最小评测。

python evaluate_knowledge.py       # 离线验证检索链路
python evaluate_knowledge.py live  # 使用配置的 Embedding API 评测语义检索
"""
import asyncio
import sys
from time import perf_counter

import chromadb

import embeddings
import knowledge_store

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOCUMENTS = {
    "差旅制度": "员工出差住宿标准为每晚 500 元，高铁可乘坐二等座，报销需在返程后 10 个工作日内提交。",
    "退款政策": "付费产品支持购买后 7 天内无理由退款，数字内容一经下载不支持退款。",
    "安全规范": "生产环境账号必须启用多因素认证，密码不得通过聊天工具发送，每 90 天轮换一次。",
    "请假制度": "年假需至少提前 3 个工作日申请，病假应在当天提交医院证明，直属主管负责审批。",
    "数据规范": "客户日志在线保留 30 天，归档数据保留 1 年，到期后由系统自动删除。",
}
CASES = [
    ("出差住酒店每晚最多报销多少？", "差旅制度"),
    ("返程后多久要提交报销？", "差旅制度"),
    ("购买产品几天内可以退款？", "退款政策"),
    ("下载过的数字内容还能退吗？", "退款政策"),
    ("生产账号需要开启什么认证？", "安全规范"),
    ("生产密码多久轮换一次？", "安全规范"),
    ("年假要提前几天申请？", "请假制度"),
    ("病假需要提供什么材料？", "请假制度"),
    ("客户日志在线保存多久？", "数据规范"),
    ("归档数据保留多长时间？", "数据规范"),
    ("今天天气怎么样？", None),
]


def fake_embed(text: str) -> list[float]:
    """离线确定性向量，只验证切块、存储、召回和指标计算。"""
    groups = [
        ("出差", "住宿", "酒店", "报销", "高铁"),
        ("购买", "退款", "产品", "数字内容"),
        ("账号", "认证", "密码", "安全", "生产"),
        ("请假", "年假", "病假", "申请", "医院", "主管"),
        ("数据", "日志", "归档", "保留", "保存", "删除"),
    ]
    return [float(sum(word in text for word in words)) for words in groups] + [0.01]


async def evaluate(live: bool = False) -> None:
    # ponytail: 临时内存库足够做回归；需要规模指标时再接专用评测平台。
    knowledge_store._collection = chromadb.EphemeralClient().get_or_create_collection(
        "knowledge_eval", metadata={"hnsw:space": "cosine"}
    )

    embed = embeddings.embed_text if live else fake_embed
    for source, content in DOCUMENTS.items():
        chunks = knowledge_store.split_text(content)
        vectors = await embeddings.embed_texts(chunks) if live else [fake_embed(c) for c in chunks]
        knowledge_store.replace_document(source, chunks, vectors)

    hits = 0
    started = perf_counter()
    for query, expected in CASES:
        results = knowledge_store.search(
            await embed(query) if live else embed(query), top_k=1, min_score=0.30
        )
        actual = results[0]["source"] if results else None
        hits += actual == expected
        print(f"{'PASS' if actual == expected else 'FAIL'} {query} -> {actual or '<none>'}")

    elapsed_ms = (perf_counter() - started) * 1000
    accuracy = hits / len(CASES)
    print(f"hit@1={accuracy:.0%}, avg_latency={elapsed_ms / len(CASES):.1f}ms")
    assert accuracy == 1.0, "知识库检索回归失败"


if __name__ == "__main__":
    asyncio.run(evaluate(len(sys.argv) > 1 and sys.argv[1] == "live"))
