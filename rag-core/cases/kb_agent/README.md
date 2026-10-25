# 用例① 企业知识库问答 Agent

对应《AI Agent 用例适配分析》中匹配度最高的场景：把已有知识（制度/文档/FAQ）升级为
「可检索 + 可对话 + 可选工具调用」的 Agent。

## 为什么用本用例
你的竞争力在 RAG 全链路落地，而企业知识库问答正是把简历里 92% 召回 / 96.7% Recall@
直接复用的最佳演示件。它和医疗问答共用 `rag_core` 引擎，**只换数据、不换代码**。

## 怎么跑（先小样本验证）
```bash
uv run python run.py data   --case kb_agent
uv run python run.py ingest --case kb_agent
uv run python run.py ask "年假怎么算" --case kb_agent
uv run python run.py ask "忘记密码怎么办" --case kb_agent --agent   # 走 Agent 层
uv run python run.py eval   --case kb_agent
```

## 接入真实企业数据
1. 把企业制度/FAQ 整理成 CSV（列：`ask,answer,title`，可选 `department`）。
2. 改 `data_sources.json` 指向你的文件（多个 source 可并列）。
3. `run.py data` → `run.py ingest` → `run.py ask`。
4. 若要"工具调用"（如查 HR 系统实时余额），在 `rag_core/agent.py` 的 `TOOLS` 注册新函数。

> 无科室字段时，pipeline 自动标 `未分类`，不影响检索，仅不参与按科室评估——这正是
> 领域无关设计的价值。
