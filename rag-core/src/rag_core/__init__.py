"""rag_core: 领域无关的 RAG 内核 + 自动评测 harness。

包结构：
    config.py      —— 配置与路径（读 .env）
    pipeline.py    —— 数据接入层（清洗/去重/脱敏/科室归一/版本指纹）
    ingest.py      —— 校验 + 分批向量化 + FAISS 入库（版本跳过重建）
    retriever.py   —— query -> embed -> top-k 检索
    generator.py   —— 拼上下文 -> LLM 生成带来源的回答
    agent.py       —— 可选 Agent 层（意图分类 + 工具调用）
"""

__version__ = "0.1.0"
