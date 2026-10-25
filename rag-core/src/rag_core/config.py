"""config.py - 集中管理路径与运行时配置。

所有路径相对「项目根目录」解析，因此 vscode 打开项目后无论从哪运行都一致。
密钥只从 .env 读取，禁止硬编码、禁止写进日志。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 项目根：src/rag_core/config.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 加载 .env（若存在）；不报错
_load_path = PROJECT_ROOT / ".env"
if _load_path.exists():
    load_dotenv(_load_path)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


class Settings:
    """运行时配置，从环境变量读取，提供合理默认值。"""

    # ---- 模型 / LLM ----
    # 注：本项目内置模型即 BAAI/bge-m3（1024 维，xlm-roberta，含 dense/colbert/sparse 多头），
    # 已复制到 models/bge-m3。默认指向它。
    # ⚠️ 不同 embedding 模型维度不同；一旦为某案例建了向量库，就别换模型，否则维度不匹配报错。
    #    要换模型，需先删除该案例的 data/vectors 再 ingest 重建。
    embedding_model_path: str = _env("EMBEDDING_MODEL_PATH", "models/bge-m3")
    llm_api_key: str = _env("LLM_API_KEY", "")
    llm_api_url: str = _env("LLM_API_URL", "")
    llm_model: str = _env("LLM_MODEL", "")
    top_k: int = int(_env("TOP_K", "5") or 5)

    # ---- 健壮性（超时 / 重试 / 可观测）----
    llm_timeout_seconds: int = int(_env("LLM_TIMEOUT_SECONDS", "30") or 30)
    llm_max_retries: int = int(_env("LLM_MAX_RETRIES", "3") or 3)
    tool_timeout_seconds: int = int(_env("TOOL_TIMEOUT_SECONDS", "15") or 15)
    trace_enabled: bool = _env("TRACE_ENABLED", "true").lower() in ("1", "true", "yes", "y")

    # ---- 评测 ----
    eval_use_llm: bool = _env("EVAL_USE_LLM", "true").lower() in ("1", "true", "yes", "y")
    human_review_ratio: float = float(_env("HUMAN_REVIEW_RATIO", "0.1") or 0.1)

    @staticmethod
    def resolve_model_path() -> str:
        """把 EMBEDDING_MODEL_PATH 解析为绝对路径（相对路径锚定到项目根，与 cwd 无关）。"""
        p = Path(Settings.embedding_model_path)
        if p.is_absolute():
            return str(p)
        return str((PROJECT_ROOT / p).resolve())

    @staticmethod
    def require_llm() -> tuple[str, str, str]:
        """返回 (api_key, base_url, model)；缺失任一项抛错，不静默。"""
        missing = [n for n, v in (
            ("LLM_API_KEY", Settings.llm_api_key),
            ("LLM_API_URL", Settings.llm_api_url),
            ("LLM_MODEL", Settings.llm_model),
        ) if not v]
        if missing:
            raise RuntimeError(
                f"缺少 LLM 配置: {missing}。请在 .env 中填写（参考 .env.example）。"
            )
        return Settings.llm_api_key, Settings.llm_api_url, Settings.llm_model


def case_dir(name: str) -> Path:
    """返回某个案例目录：cases/<name>/。"""
    d = PROJECT_ROOT / "cases" / name
    if not d.exists():
        raise FileNotFoundError(f"案例目录不存在: {d}")
    return d


def vector_dir_for(name: str) -> Path:
    """每个案例独立的向量库目录：cases/<name>/data/vectors。

    内含 FAISS 索引（faiss.index）+ 元数据（store.pkl）+ 版本标记（_version.txt）。
    """
    return case_dir(name) / "data" / "vectors"
