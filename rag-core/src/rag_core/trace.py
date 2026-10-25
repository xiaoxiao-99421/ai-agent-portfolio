"""trace.py - 轻量可观测层。

设计目标（C5 可观测）：
    - 每次「问答 / 评测」一个 trace_id，串联整条链路
    - 记录每个阶段：名称、相对耗时(ms)、关键指标（命中数、是否拒答等）
    - 结构化 JSON 日志，trace_id 始终带出，便于串联排查；绝不打印密钥
    - 不引入外部依赖，纯标准库，开关由 Settings.trace_enabled 控制

用法：
    from .trace import Trace
    t = Trace()
    t.step("retrieve", hits=5)
    t.step("generate", grounded=True, latency_ms=820)
    return {"trace_id": t.trace_id, "steps": t.steps}
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional


class Trace:
    def __init__(self, enabled: bool = True, trace_id: Optional[str] = None):
        self.enabled = enabled
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.steps: List[Dict[str, Any]] = []
        self._t0 = time.time()

    def step(self, name: str, **kw) -> Dict[str, Any]:
        """记录一个阶段。返回该步记录，便于调用方引用。"""
        rec = {"step": name, "ts_ms": round((time.time() - self._t0) * 1000)}
        rec.update(kw)
        self.steps.append(rec)
        if self.enabled:
            self._log(rec)
        return rec

    def _log(self, rec: Dict[str, Any]) -> None:
        payload = {"trace_id": self.trace_id, **rec}
        print(f"[trace] {json.dumps(payload, ensure_ascii=False)}", flush=True)

    def to_dict(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "steps": self.steps}
