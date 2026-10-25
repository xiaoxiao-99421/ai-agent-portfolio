"""generator.py - 拼上下文 -> LLM 生成带来源的回答。

设计要点：
    - 使用 OpenAI 兼容接口（base_url + api_key 来自 .env），不硬编码
    - 提示词约束：必须基于检索资料回答、资料不足时拒答、不提供诊断/个体化处置
    - 输出带来源（命中片段的科室 + 问题），便于溯源与人工复核
    - 失败处理：API 故障 / 空响应 -> 返回降级提示，不崩溃
"""
from __future__ import annotations

import time
from typing import List, Optional

from openai import OpenAI

from .config import Settings
from .retriever import Hit
from .trace import Trace


MEDICAL_SYSTEM_PROMPT = """你是健康信息问答助手。请严格遵守：
1. 只基于下面提供的【检索资料】回答，不得编造资料外的内容。
2. 若检索资料不足以回答问题，明确说明"资料不足，建议咨询专业医生"，不得猜测。
3. 不提供诊断结论、不开处方、不给出个体化用药/手术建议。
4. 回答末尾用「来源」列出你参考的资料条目（科室 + 问题）。"""


def _build_user_prompt(query: str, hits: List[Hit]) -> str:
    if not hits:
        return f"用户问题：{query}\n\n（无检索资料）"
    ctx = "\n\n".join(
        f"[{i+1}]（{h.department}）{h.question}\n{h.text}"
        for i, h in enumerate(hits)
    )
    return f"用户问题：{query}\n\n【检索资料】\n{ctx}"


def _extract_sources(hits: List[Hit]) -> str:
    return "\n".join(f"- （{h.department}）{h.question}" for h in hits)


def generate(query: str, hits: List[Hit],
             model: Optional[str] = None, temperature: float = 0.2,
             trace: Optional[Trace] = None) -> dict:
    """返回 {answer, sources, grounded, trace_id, steps}。

    健壮性（C4）：
        - 显式超时（LLM_TIMEOUT_SECONDS）
        - 指数退避重试（LLM_MAX_RETRIES 次，上限 8s）
        - 全部失败 -> 降级提示，绝不抛异常打断主流程
    """
    trace = trace or Trace(enabled=Settings.trace_enabled)
    prompt = _build_user_prompt(query, hits)
    sources = _extract_sources(hits)
    api_key, base_url, mdl = Settings.require_llm()
    trace.step("generate_start", grounded=bool(hits), context_chars=len(prompt))

    # 自管重试（SDK max_retries 关闭，避免双重退避）
    client = OpenAI(api_key=api_key, base_url=base_url,
                    timeout=Settings.llm_timeout_seconds, max_retries=0)

    last_err: Optional[Exception] = None
    for attempt in range(1, Settings.llm_max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model or mdl,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = resp.choices[0].message.content.strip() if resp.choices else ""
            if not answer:
                answer = "资料不足，建议咨询专业医生。"
            trace.step("generate_ok", attempt=attempt, answer_len=len(answer))
            return {"answer": answer, "sources": sources, "grounded": bool(hits),
                    "trace_id": trace.trace_id, "steps": trace.steps}
        except Exception as e:  # 网络/API 故障 -> 重试或降级
            last_err = e
            trace.step("generate_retry", attempt=attempt, error=type(e).__name__)
            if attempt < Settings.llm_max_retries:
                time.sleep(min(2 ** attempt, 8))  # 指数退避，封顶 8s

    # 全部重试仍失败 -> 降级（不崩溃）
    return {"answer": f"抱歉，模型服务暂不可用（已重试 {Settings.llm_max_retries} 次），"
                      f"请稍后重试或检查 API 配置。",
            "sources": sources, "grounded": bool(hits), "error": str(last_err),
            "trace_id": trace.trace_id, "steps": trace.steps}
