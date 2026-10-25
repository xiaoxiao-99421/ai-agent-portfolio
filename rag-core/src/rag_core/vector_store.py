"""vector_store.py - 基于 FAISS 的向量索引封装（替代 chroma 的向量层）。

为什么换掉 chroma：chroma 0.4.x / 1.5.x 在 Windows 上的 HNSW 索引持久化不可靠
（metadata pickle 写了、HNSW 数据文件却没写，重开即 "Cannot open header file"；
已实测：即便设 hnsw:sync_threshold=64，真实 4266 条入库后 segment 目录仍只有
index_metadata.pickle，无 header.bin/data_level0.bin）。

FAISS 用显式 faiss.write_index / read_index，落盘时机 100% 由我们代码控制，
不依赖任何外部析构 / 后台线程 / 阈值触发——这正是生产级系统需要的「可控持久化」。

约定：embedding 由调用方在 ingest 时归一化（bge-m3 normalize_embeddings=True），
因此用 IndexFlatIP（归一化向量的内积 = 余弦相似度），检索分数即余弦相似度。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# faiss 采用【延迟导入】：faiss import 本身会占用大量提交内存（Windows 上实测会与
# bge-m3 模型加载撞车，报 "页面文件太小 (os error 1455)" / MemoryError）。
# 先加载模型、用到 FaissStore 时再 import faiss，两者错峰即可在 16GB 内存机器上共存。
faiss = None


def _ensure_faiss():
    global faiss
    if faiss is None:
        import faiss as _faiss

        faiss = _faiss
    return faiss


class FaissStore:
    """一个案例一个目录：faiss.index（向量）+ store.pkl（id/文档/元数据）+ _version.txt。"""

    def __init__(self, dim: int):
        _ensure_faiss()
        self.dim = dim
        # 精确内积索引；数据量（≤ 数十万）下暴力检索足够快，且无 HNSW 召回损失。
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.metas: List[Dict] = []

    def ntotal(self) -> int:
        return len(self.ids)

    def add(self, embeddings: np.ndarray, ids: List[str], docs: List[str],
            metas: List[Dict]) -> None:
        """追加向量与对应的 id/文档/元数据（四项顺序严格对应）。"""
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"维度不匹配：{embeddings.shape[1]} != {self.dim}")
        if not (embeddings.shape[0] == len(ids) == len(docs) == len(metas)):
            raise ValueError("embeddings / ids / docs / metas 长度不一致")
        self.index.add(embeddings)
        self.ids.extend(ids)
        self.docs.extend(docs)
        self.metas.extend(metas)

    def search(self, query: np.ndarray, k: int):
        """query 形状 (1, dim) float32。返回 (distances, positions)。"""
        k = min(k, self.ntotal())
        if k == 0:
            return np.empty((1, 0), dtype="float32"), np.empty((1, 0), dtype="int64")
        distances, positions = self.index.search(query, k)
        return distances, positions

    # ---- 持久化（显式，完全可控；不依赖任何析构/后台线程）----
    # 注意：不用 faiss.write_index / read_index 直写路径——其 C++ FileIO 在 Windows
    # 上用 ANSI 代码页解释路径，含中文的项目路径会 fopen 失败（No such file or
    # directory）。改用 serialize_index / deserialize_index + Python open()
    # （完整支持 Unicode 路径），并先写临时文件再原子替换，避免半写文件。
    def save(self, directory: Path) -> None:
        f = _ensure_faiss()
        directory.mkdir(parents=True, exist_ok=True)
        arr = f.serialize_index(self.index)  # uint8 numpy 数组
        tmp = directory / "faiss.index.tmp"
        with open(tmp, "wb") as fp:
            fp.write(arr.tobytes())
        tmp.replace(directory / "faiss.index")
        tmp_pkl = directory / "store.pkl.tmp"
        with open(tmp_pkl, "wb") as fp:
            pickle.dump(
                {"ids": self.ids, "docs": self.docs, "metas": self.metas, "dim": self.dim},
                fp,
            )
        tmp_pkl.replace(directory / "store.pkl")

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        f = _ensure_faiss()
        if not (directory / "faiss.index").exists():
            raise FileNotFoundError(f"未找到向量索引：{directory / 'faiss.index'}")
        with open(directory / "faiss.index", "rb") as fp:
            arr = np.frombuffer(fp.read(), dtype="uint8")
        index = f.deserialize_index(arr)
        with open(directory / "store.pkl", "rb") as fp:
            data = pickle.load(fp)
        store = cls(data["dim"])
        store.index = index
        store.ids = data["ids"]
        store.docs = data["docs"]
        store.metas = data["metas"]
        if index.ntotal != len(store.ids):
            raise RuntimeError(
                f"向量索引与元数据条数不一致（{index.ntotal} != {len(store.ids)}），索引可能损坏"
            )
        return store
