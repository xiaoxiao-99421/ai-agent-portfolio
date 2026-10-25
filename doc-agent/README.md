# 文档自动化处理

粘贴采购单 / 票据 / 合同文本 → Agent 自动解析 → 结构化抽取 → 规则校验 → 写入数据池（幂等 + 版本指纹）。

面向「单据 / 文档批量入库」场景：把非结构化文本转成可校验、可对账的结构化记录，减少人工逐条录入与错漏。

## 特性
- **端到端自动化编排**：解析 → LLM 抽取 → 规则校验 → 入库 → 报告，每步可见
- **工程机制**：金额一致性校验、重复检测（幂等跳过）、数据版本指纹（`data/version.json`）
- **零额外依赖**：仅 Python 标准库（`http.server` + `urllib`），不装 FastAPI/requests
- **无 CORS、不暴露密钥**：浏览器同源 localhost，key 只在服务端读 `.env`
- **复用 LLM 配置**：优先 `doc-agent/.env`，缺失则回退 `../rag-core/.env`

## 运行
```bash
cd 求职作品集/doc-agent
cp .env.example .env        # rag-core/.env 已填则可跳过
python server.py --port 8002
# 浏览器打开 http://localhost:8002
```
打开后粘贴采购单/票据文本（或点示例），点「运行 Agent」，即可看到：执行步骤时间线、结构化字段表、校验报告、入库记录（含 record_id 与数据版本）。

## 接口
- `GET  /`                演示界面
- `POST /api/process`      入参 `{text}` → 返回 `{steps[], extracted, validation, store, summary}`

## 数据落地
- 抽取并校验通过的记录追加到 `doc-agent/data/records.jsonl`
- `doc-agent/data/version.json` 记录数据版本指纹与总条数（每次写入重算，便于增量对账）
- 同一文档（按原文哈希）重复提交会被幂等跳过，不会重复入库

## 自定义
- 改抽取字段/格式：编辑 `server.py` 的 `SYSTEM_PROMPT`
- 改校验规则：编辑 `validate()`（如增加税号校验、币种白名单）
- 换端口：`python server.py --port 9002` 或 `DOC_PORT=9002 python server.py`

> 演示数据均为仿真示例；生产环境可在此基础上接入文件夹监听、PDF/OCR 解析与真实业务库。
