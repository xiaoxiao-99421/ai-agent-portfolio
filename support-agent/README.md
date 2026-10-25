# 智能客服助手

一个多轮对话式客服 Agent：**RAG 知识库 + 工具调用 + 风险约束 + 人在回路（human-in-the-loop）**。

面向电商客服场景：用户咨询政策/订单时，Agent 先检索知识库、按需调用订单工具作答；
一旦涉及退款或高风险诉求，**不自动执行**，而是生成待人工审核工单并引导转人工。

## 一、能力构成

| 能力 | 在本项目中如何体现 |
|---|---|
| **RAG 知识库** | `knowledge.json` 预置客服政策（退换货/物流/发票/账户/支付/会员等）；`retrieve()` 用字符 bigram 重叠 + 关键词加权做稀疏检索，取 top-3 片段喂给 LLM，严格"只依据片段作答"。 |
| **工具调用** | Agent 决策后调用三类工具：`query_order`（查订单状态/物流）、`request_refund`（退款）、`escalate_to_human`（转人工）。工具由 Planner 决策、后端真实执行。 |
| **风险约束** | 代码层硬规则优先级高于 LLM 软判断：敏感词黑名单拦截、用户强风险词升级、**答复越界承诺词检测**，越界即拦截改写。 |
| **人在回路** | 退款 / 高风险诉求**不直接执行**，生成 `pending_human` 工单落盘并话术转接人工；前端展示"已转人工 + 工单号"。这正是生产级客服 Agent 的审批节点。 |

## 二、运行

```bash
cd 求职作品集/support-agent
cp .env.example .env        # rag-core/.env 已填则可跳过
python server.py --port 8005
# 浏览器打开 http://localhost:8005
```

- 多轮对话：界面维护上下文，可直接连续追问。
- 示例问题已内置若干，点 chip 即可（含退款/高风险等转人工场景）。

## 三、演示路径

1. **RAG 检索**：点退换货政策类问题 → 看 Agent 思考过程里的"知识库检索"命中条目，回答依据片段生成。
2. **工具调用**：点查订单物流 → Planner 调用 `query_order` → 返回运单号与预计到达。
3. **人在回路（退款）**：点申请退款 → 调用 `request_refund` → **强制转人工**，生成退款工单号。
4. **风险约束 + 转人工**：点高风险索赔 → 命中强风险词 → 风险升为 high → 转人工工单。

## 四、文件结构

```
support-agent/
├── server.py          # 零依赖服务端：Planner→检索+工具→Answerer→风险后处理
├── index.html         # 多轮对话界面（可折叠 Agent 思考过程）
├── knowledge.json     # RAG 知识库（客服政策条目）
├── orders.json        # 订单池（供 query_order）
├── .env.example       # LLM 配置示例（回退 ../rag-core/.env）
├── data/              # 运行时生成：refund_tickets.jsonl / escalations.jsonl（已 gitignore）
└── README.md
```

## 五、与其他作品的关系

本作品与 RAG 问答、文档自动化、会议纪要等作品共用同一套「零依赖 + 复用 LLM 配置」工程范式。

> 演示数据（知识库条目、订单）均为仿真示例；生产环境可在此基础上接入真实知识库与工单/订单系统。
