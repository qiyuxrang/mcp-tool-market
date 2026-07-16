"""Mem0 式双阶段记忆流水线：抽取候选事实 -> 对比已有记忆做更新决策。"""
import json
import math
import os

from openai import AsyncOpenAI

import embeddings
import memory_store

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:3000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-not-set")
MODEL = os.getenv("MODEL_NAME", "gpt-4o")

_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

EXTRACT_PROMPT = """你是记忆抽取器。阅读下面的对话，提取值得长期记住的关于用户的事实。

只提取：用户身份、偏好、目标、约束、重要经历（如"用户对花生过敏"、"用户在找AI岗位工作"）。
不要提取：临时性问题（如"用户问了天气"）、闲聊、AI自己说的话。

输出 JSON 数组，每项格式 {"content": "事实描述", "importance": 0到1的重要性分数}。
没有值得记的就输出 []。只输出 JSON，不要其他文字。

对话：
{conversation}"""

RESOLVE_PROMPT = """你是记忆管理器。判断"候选事实"与"已有记忆"的关系，决定如何操作。

规则：
- ADD：候选是全新信息，已有记忆中没有 -> {"action": "ADD"}
- UPDATE：候选与某条已有记忆是同一主题但信息更新/冲突（如搬家、换工作）-> {"action": "UPDATE", "target_id": "那条记忆的id", "content": "合并后的新表述"}
- DELETE：候选表明用户明确否定了某条已有记忆 -> {"action": "DELETE", "target_id": "那条记忆的id"}
- NONE：候选与已有记忆重复，无新信息 -> {"action": "NONE"}

只输出一个 JSON 对象，不要其他文字。

候选事实：{fact}

已有记忆：
{existing}"""


async def _chat_json(prompt: str):
    """调 LLM 并解析 JSON 输出，失败返回 None。"""
    try:
        resp = await _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        # 剥掉可能的 markdown 代码块
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return None


async def extract_facts(conversation: list[dict]) -> list[dict]:
    """阶段1：从对话中抽取候选事实。返回 [{"content", "importance"}]。"""
    convo_text = "\n".join(
        f"{m.get('role', '?')}: {m.get('content', '')}"
        for m in conversation if m.get("content")
    )
    if not convo_text.strip():
        return []
    result = await _chat_json(EXTRACT_PROMPT.replace("{conversation}", convo_text))
    if not isinstance(result, list):
        return []
    facts = []
    for item in result:
        if isinstance(item, dict) and item.get("content"):
            content = str(item["content"]).strip()
            if not content or len(content) > 2000:
                continue
            try:
                importance = float(item.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            if not math.isfinite(importance):
                importance = 0.5
            facts.append({
                "content": content,
                "importance": min(max(importance, 0), 1),
            })
    return facts


async def resolve_memory(fact: dict, user_id: str) -> dict:
    """阶段2：对单条候选事实做 ADD/UPDATE/DELETE/NONE 决策并执行。

    返回操作记录 {"action", "content", "target_id"?}。
    """
    fact_emb = await embeddings.embed_text(fact["content"])
    similar = memory_store.search_memories(fact_emb, user_id, top_k=5)

    if not similar:
        mem = memory_store.add_memory(
            fact["content"], fact_emb, user_id,
            importance=fact["importance"], source="auto")
        return {"action": "ADD", "content": fact["content"], "id": mem["id"]}

    existing_text = "\n".join(
        f'- id={m["id"]} content="{m["content"]}"' for m in similar
    )
    decision = await _chat_json(
        RESOLVE_PROMPT.replace("{fact}", fact["content"])
                      .replace("{existing}", existing_text)
    )
    if not isinstance(decision, dict):
        return {
            "action": "ERROR",
            "content": fact["content"],
            "error": "记忆决策返回无法解析，未修改存储",
        }

    action = str(decision.get("action", "")).upper()
    if action not in {"ADD", "UPDATE", "DELETE", "NONE"}:
        return {
            "action": "ERROR",
            "content": fact["content"],
            "error": f"不支持的记忆操作: {action or '<empty>'}",
        }
    if action in {"UPDATE", "DELETE"} and not decision.get("target_id"):
        return {
            "action": "ERROR",
            "content": fact["content"],
            "error": f"{action} 缺少 target_id，未修改存储",
        }

    if action == "UPDATE" and decision.get("target_id"):
        new_content = decision.get("content") or fact["content"]
        new_emb = await embeddings.embed_text(new_content)
        ok = memory_store.update_memory(
            decision["target_id"], new_content, new_emb, user_id,
            importance=fact["importance"])
        if ok:
            return {"action": "UPDATE", "content": new_content,
                    "target_id": decision["target_id"]}
        return {
            "action": "ERROR",
            "content": fact["content"],
            "target_id": decision["target_id"],
            "error": "目标记忆不存在或不属于当前用户",
        }

    if action == "DELETE" and decision.get("target_id"):
        if not memory_store.delete_memory(decision["target_id"], user_id):
            return {
                "action": "ERROR",
                "target_id": decision["target_id"],
                "content": fact["content"],
                "error": "目标记忆不存在或不属于当前用户",
            }
        return {"action": "DELETE", "target_id": decision["target_id"],
                "content": fact["content"]}

    if action == "NONE":
        return {"action": "NONE", "content": fact["content"]}

    mem = memory_store.add_memory(
        fact["content"], fact_emb, user_id,
        importance=fact["importance"], source="auto")
    return {"action": "ADD", "content": fact["content"], "id": mem["id"]}


async def run_pipeline(conversation: list[dict], user_id: str) -> list[dict]:
    """完整流水线：抽取 -> 逐条决策。返回操作记录列表。"""
    facts = await extract_facts(conversation)
    operations = []
    for fact in facts:
        try:
            op = await resolve_memory(fact, user_id)
            operations.append(op)
        except Exception as e:
            operations.append({"action": "ERROR", "content": str(e)})
    return operations
