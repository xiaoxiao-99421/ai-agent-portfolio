# 会议纪要与任务工单

粘贴会议转录 → Agent 用 LLM 抽取 action items → 逐个调用 `create_ticket` 工具 → 生成纪要摘要。

面向「开完会没人跟」的场景：自动把讨论结论拆成可指派、可跟踪的任务工单，避免事项遗漏。

## 特性
- **Agent 编排 + 工具调用**：明确展示「抽取 → 调用 create_ticket 工具」的编排链路
- **工程机制**：工单幂等（同任务只建一次）、本地工单池（`data/tickets.jsonl`）
- **零额外依赖**：仅 Python 标准库（`http.server` + `urllib`），不装 FastAPI/requests
- **无 CORS、不暴露密钥**：浏览器同源 localhost，key 只在服务端读 `.env`
- **复用 LLM 配置**：优先 `meeting-agent/.env`，缺失则回退 `../rag-core/.env`

## 运行
```bash
cd 求职作品集/meeting-agent
cp .env.example .env        # rag-core/.env 已填则可跳过
python server.py --port 8003
# 浏览器打开 http://localhost:8003
```
打开后粘贴会议转录（或点示例），点「运行 Agent」，即可看到：执行步骤时间线、抽取的 action items 表、创建的工单列表、纪要摘要。

## 接口
- `GET  /`                演示界面
- `POST /api/process`      入参 `{text}` → 返回 `{steps[], extracted, tickets[], summary}`

## 换真实工单系统
当前 `create_ticket()` 为本地 mock（写 `data/tickets.jsonl`）。要接飞书/Notion/Linear 等，只需把该函数体替换为对应 API 调用（保持返回 `{ticket_id, status}` 即可），上层编排无需改动。

## 自定义
- 改抽取字段/格式：编辑 `server.py` 的 `SYSTEM_PROMPT`
- 换端口：`python server.py --port 9003` 或 `MEETING_PORT=9003 python server.py`

> 演示数据均为仿真示例；生产环境可在此基础上接入实时转录、工单系统 webhook 与权限校验。
