# rag-core · 领域无关的 RAG 内核 + 自动评测 harness

把原 `rag-minimal`（医疗问答 RAG 原型）中**与领域无关**的能力抽出来，做成可复用工程，
并把"评估需大量人力标注"的痛点用**自动评测 harness** 解决。对应《AI Agent 用例适配分析》用例①。

## 结构
```
rag-core/
├── pyproject.toml          # uv 项目（python 3.11）
├── .env.example            # 复制为 .env 并填 LLM / 模型路径
├── src/rag_core/
│   ├── config.py           # 路径与配置（读 .env，密钥不外泄）
│   ├── pipeline.py         # 数据接入层（清洗/去重/脱敏/科室归一/版本指纹）
│   ├── ingest.py           # 校验 + 分批向量化 + FAISS 入库（增量 add，支持分块续跑）
│   ├── vector_store.py     # FAISS 向量索引封装（显式 write/read_index 落盘）
│   ├── retriever.py        # query -> embed -> top-k
│   ├── generator.py        # 拼上下文 -> LLM 带来源回答（医疗安全约束）
│   └── agent.py            # 用例① 可选 Agent 层（意图分类 + 工具调用）
├── eval/harness.py         # 自动 gold 候选 + LLM 判相关 + 抽样人工复核
├── cases/
│   ├── medical_qa/         # 医疗用例（已纳入真实 ~247k 数据，gitignore；用 limit 分块续跑）
│   └── kb_agent/           # 用例① 企业知识库问答（演示领域无关）
└── run.py                  # 统一入口
```

## 快速开始
```bash
# 1. 环境
uv venv && uv pip install -e .

# 2. 密钥与模型（二选一填法）
cp .env.example .env
#   - EMBEDDING_MODEL_PATH 指向本地 bge 模型（如原 rag-minimal/models/bge-m3）
#     或填 HuggingFace 名（如 BAAI/bge-small-zh-v1.5）自动下载
#   - LLM_API_KEY / LLM_API_URL / LLM_MODEL 填 OpenAI 兼容接口

# 3. 跑通一条链路（医疗小样本）
uv run python run.py data   --case medical_qa
uv run python run.py ingest --case medical_qa
uv run python run.py ask "拉肚子咋整" --case medical_qa

# 4. 自动评测（不再人工填 gold_ids）
uv run python run.py eval   --case medical_qa
#   -> data/eval/eval_results.jsonl（全量自动判定）
#   -> data/eval/eval_review.jsonl（仅 10% 低置信样本交你复核）
```

## 大文件分块增量运行（续跑）
原始 `内科.csv`/`外科.csv` 各数十万行，一次向量化要数小时。用 `data_sources.json` 的 `limit` 把"本次纳入行数"分块，配合 `ingest` 的**增量入库**（只向量化新 id），可把长任务拆成多块，电脑空闲时逐块累加进同一向量库。

### 范围限制（limit）落在哪里
- 配置：`cases/medical_qa/data_sources.json` 每个数据源的 `"limit"`（如 `"limit": 30000`）表示本次只读取该 CSV 的**前 N 行**。
- 代码：`pipeline.py` 的 `stage1_raw_to_interim()` 用 `pd.read_csv(path, encoding=enc, nrows=limit)` 截断读取；并且该 `limit` 被写进数据源指纹（`_source_sig`），调大 limit 会让该源缓存失效、重新清洗。
- 不设 `limit`（或值为 `null`）→ 读取全文件，即"跑全量"。

### 怎么分块续跑（以 1/5 起步）
1. 把 `limit` 设成第一批大小（例：内科 30000 / 外科 15000），然后：
   ```bash
   uv run python run.py data   --case medical_qa
   uv run python run.py ingest --case medical_qa
   ```
   集合为空 → 这批全部向量化入库。
2. 空闲时把 `limit` 调大（例：内科 60000），再跑 `data` + `ingest`：
   - `data`：因 limit 变化重新清洗**前 6 万行**（前一批重复，但清洗很快，瓶颈在向量化）。
   - `ingest`：**增量 upsert**——按稳定 `record_id` 比对集合已有 id，只向量化新增的 3 万行，累加进同一集合。
3. 重复调大 `limit`（90000、120000…）直到覆盖全量；或删掉 `limit` 字段跑全量，`ingest` 仍只补未入库的 id。

### 为什么能累加且幂等
- `record_id` 由 `(source_name, ask, answer)` 稳定哈希生成（`pipeline.py`），同一行在不同 limit 下 id 不变。
- 入库前取集合已有 id 集合，仅 `upsert` 缺失项；中途中断重跑会补齐，不重复、不丢数据。
- 仅当 embedding 模型变更时才加 `--force-rebuild` 清空重建；日常续跑不加该参数：
  ```bash
  uv run python run.py ingest --case medical_qa --force-rebuild
  ```

### 注意
- 分块是"从前往后"累积（读前 N 行），不能跳段处理中间区间。
- `data` 每次重读前 limit 行，前几批会被重复清洗（轻量）；真正省时的是 `ingest` 只做新增向量化。

## 评测为什么不再"大量人力"
原项目 `eval_set_annotated.jsonl` 每条都是 `query_type: TODO` + "待人工核对 gold_ids"。
本 harness：① 用向量近邻 top-N **自动生成候选 gold**；② **LLM 判相关 + 自动打 query_type**；
③ 仅把低置信/边界样本（默认 10%）写入 `eval_review.jsonl` 交你抽样确认。
换数据/换方向时这套逻辑不变。

## 设计取舍（面试可讲）
- 核心检索→生成是 **workflow，不是 Agent**；Agent 层只是可选的"可控自主"叠加（有 max_steps 上限）。
- 版本指纹贯穿 pipeline→ingest，数据变更才重建向量库，避免"查到旧数据"。
- 外部调用（LLM/向量）均有超时与降级；凭据只走环境变量，不进日志。
- 大体积模型/数据不进仓库（`.gitignore`），由 `.env` / setup 指向，工作空间保持轻量、vscode 直接打开可跑 demo。

## Web 演示件（用例①，面试现场演示）
把已有的 `Retriever` + `Generator` 包成网页，面试时输入医疗问题即可现场展示「向量检索 → 带来源回答」全链路：

```bash
cd rag-core
python web_demo.py --case medical_qa --port 8001
# 浏览器打开 http://localhost:8001
```

- 界面：输入问题（含示例问题一键填入）→ 展示 top-k 检索片段（相似度分数 + 科室 + 原文摘要）+ LLM 带来源回答 + trace_id
- 依赖：必须在 rag-core 的 Python 环境运行（含 faiss-cpu / sentence_transformers / openai + `models/bge-m3` + 已 `ingest` 的向量库）
- 零 Web 框架（标准库 `http.server`）；bge-m3 模型与 FAISS 索引在首次提问时加载
- 演示数据：已全量向量化的真实医疗问答（约 24.8 万条，`data_version.json` 记录 247,680 条），覆盖内科 / 外科两源，经清洗、去重、脱敏与科室归一
- 路由：`GET /` 演示界面；`POST /api/ask` 入参 `{question, top_k?}`，返回 `{answer, sources, hits[], trace_id}`

> 项目定位：健康信息 / 企业知识问答**技术原型**，非诊断或临床决策系统。
