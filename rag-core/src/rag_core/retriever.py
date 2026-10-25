"""retriever.py - query -> embed -> top-k 检索（纯 workflow，非 Agent）。

职责：把用户问题转向量，在 FAISS 中召回 top-k 片段，返回可解释结果。
失败处理：
    - 空结果：返回空列表（由上层决定拒答）
    - 检索/编码异常：记录 trace 后返回空列表，不抛异常打断主流程（C4）
    - 索引缺失/损坏：自动全量重建后再返回（兜底，宁可首次慢一点也不要演示时打不开）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import Settings, vector_dir_for
from .trace import Trace
from .vector_store import FaissStore


@dataclass
class Hit:
    id: str
    score: float
    question: str
    department: str
    text: str


class Retriever:
    def __init__(self, case: str, top_k: Optional[int] = None):
        self.case = case
        self.top_k = top_k or Settings.top_k
        from .ingest import load_embedding_model  # 函数内导入，避免模块级循环依赖

        self.model = load_embedding_model()
        self.store = self._load_store(case)

    def _load_store(self, case: str) -> FaissStore:
        from .ingest import ingest  # 函数内导入，避免模块级循环依赖

        d = vector_dir_for(case)
        try:
            return FaissStore.load(d)
        except Exception as e:
            print(f"[自检] 向量索引读取失败（{type(e).__name__}: {e}），开始自动重建…", flush=True)
            ingest(case, force_rebuild=True)
            return FaissStore.load(d)

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 trace: Optional[Trace] = None) -> List[Hit]:
        k = top_k or self.top_k
        trace = trace or Trace(enabled=Settings.trace_enabled)
        try:
            q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
            distances, positions = self.store.search(q_emb.reshape(1, -1), k)
        except Exception as e:  # 编码/向量库故障 -> 兜底空结果
            trace.step("retrieve_error", error=type(e).__name__)
            return []

        hits: List[Hit] = []
        if positions is None or len(positions[0]) == 0:
            trace.step("retrieve", hits=0)
            return hits
        for pos, dist in zip(positions[0], distances[0]):
            pos = int(pos)
            if pos < 0 or pos >= len(self.store.ids):
                continue
            hits.append(Hit(
                id=self.store.ids[pos],
                score=round(float(dist), 4),  # 归一化内积 = 余弦相似度
                question=self.store.metas[pos].get("question", ""),
                department=self.store.metas[pos].get("department", ""),
                text=self.store.docs[pos] or "",
            ))
        trace.step("retrieve", hits=len(hits), top_score=hits[0].score if hits else None)
        return hits
