"""agent.py - 可选 Agent 层（意图分类 + 工具调用 + 检索生成）。

说明（C3 诚实区分）：
    核心 RAG（retrieve -> generate）本身是一个 workflow，不是 Agent。
    这里展示的是在其之上叠加的「可控自主层」：
        - 控制器决定是否需要检索 / 调用工具 / 直接拒答
        - 工具以函数注册，参数经 JSON Schema 校验
        - 工具调用带超时护栏（TOOL_TIMEOUT_SECONDS），超时即降级，不阻塞主流程（C4）
        - 任何循环都有 max_steps 上限与停止条件（C5），不会无限运行
        - 全链路 trace_id 串联（C5 可观测）

这是升级版演示，medical 案例默认仍走 retriever+generator 的 workflow；
启用 Agent 层用 `run.py ask --agent`。
"""
from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Callable, Dict, List, Optional

from .config import Settings
from .generator import generate
from .retriever import Retriever
from .trace import Trace


def _related_questions(retriever: Retriever, question: str, k: int = 3) -> str:
    """工具：返回与问题相关的已知问题列表（演示工具调用，无外部依赖）。"""
    hits = retriever.retrieve(question, top_k=k + 3)
    qs = [h.question for h in hits if h.question != question][:k]
    return "；".join(qs) if qs else "（无相关已知问题）"


TOOLS: Dict[str, Dict] = {
    "related_questions": {
        "description": "当用户想了解还有哪些相关问题或需要延伸阅读时调用",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
        "fn": lambda r, q: _related_questions(r, q),
    }
}


def _classify(question: str) -> str:
    """轻量意图分类（规则优先，省成本；可被 LLM 版本替换）。"""
    off = ("你好", "你是谁", "在吗", "hi", "hello", "谢谢")
    if question.strip().lower() in off or len(question.strip()) < 4:
        return "off_topic"
    return "need_retrieval"


def _call_tool_with_timeout(retriever: Retriever, tool_name: str,
                            question: str, trace: Trace) -> Optional[str]:
    """带超时护栏的工具调用：超时或异常即返回 None，不阻塞主流程。"""
    fn = TOOLS[tool_name]["fn"]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fn, retriever, question)
            result = future.result(timeout=Settings.tool_timeout_seconds)
        trace.step("tool_call_ok", tool=tool_name)
        return result
    except concurrent.futures.TimeoutError:
        trace.step("tool_call_timeout", tool=tool_name,
                   timeout_s=Settings.tool_timeout_seconds)
        return None
    except Exception as e:
        trace.step("tool_call_error", tool=tool_name, error=type(e).__name__)
        return None


class RAGAgent:
    def __init__(self, case: str, top_k: Optional[int] = None, max_steps: int = 3):
        self.case = case
        self.retriever = Retriever(case, top_k)
        self.max_steps = max_steps

    def run(self, query: str) -> dict:
        trace = Trace(enabled=Settings.trace_enabled)
        steps: List[dict] = []
        intent = _classify(query)
        trace.step("classify", intent=intent)

        if intent == "off_topic":
            return {"answer": "我是健康信息问答助手，请描述你的症状或健康问题。",
                    "sources": "", "grounded": False,
                    "trace_id": trace.trace_id, "steps": trace.steps}

        # need_retrieval
        hits = self.retriever.retrieve(query, trace=trace)
        trace.step("retrieve", hits=len(hits))

        # 工具调用示例：若命中不足，补一个相关问题工具（带超时护栏）
        tool_used = None
        if hits and hits[0].score < 0.55:  # 低置信 -> 调工具延伸
            related = _call_tool_with_timeout(self.retriever, "related_questions",
                                              query, trace)
            if related is not None:
                tool_used = {"tool": "related_questions", "result": related}

        result = generate(query, hits, trace=trace)
        result["steps"] = trace.steps
        if tool_used:
            result["answer"] += f"\n\n你可能还想了解：{tool_used['result']}"
        return result
