#!/usr/bin/env python3
"""support-agent · 智能客服/支持 Agent

把 RAG 知识库 + 工具调用 + 风险约束 + 人在回路（human-in-the-loop）
做成端到端客服 Agent 的演示件，对应「对话系统/智能客服开发工程师」岗。

零额外依赖（Python 标准库 http.server + urllib）：
  GET  /               演示界面（index.html）
  POST /api/chat       {messages:[{role,content}...]} -> {reply, steps, sources, tools, risk, needs_human, ticket}

能力拆解（即该 Agent 的四大要素）：
  1) RAG 知识库   -> 本地稀疏检索（字符 bigram 重叠 + 关键词加权）取 top-k 片段，喂给 LLM
  2) 工具调用     -> query_order（查订单）/ request_refund（退款）/ escalate_to_human（转人工）
  3) 风险约束     -> 代码层硬规则：敏感词黑名单拦截、绝对化承诺检测、高风险动作强制转人工
  4) 人在回路     -> 退款/高风险不直接执行，生成 pending_human 工单，话术转接人工

LLM 配置：优先 support-agent/.env，回退 ../rag-core/.env（复用已填配置）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def _resolve_port(env_key: str, default: str) -> int:
    """端口优先级：命令行 --port > 环境变量 > 默认值。"""
    argv = sys.argv[1:]
    if "--port" in argv:
        i = argv.index("--port")
        if i + 1 < len(argv):
            return int(argv[i + 1])
    return int(os.environ.get(env_key, default))


PORT = _resolve_port("SUPPORT_PORT", "8005")

# ---------------------------------------------------------------------------
# 风险硬约束（代码层，优先级高于 LLM 软判断）
# ---------------------------------------------------------------------------
# 输入黑名单：命中即合规拦截并强制转人工（演示用，可扩展）
BLOCK_WORDS = [
    "政治", "色情", "赌博", "诈骗", "私下交易", "加微信私聊", "转账到私人",
    "银行卡号发给我", "把密码告诉我", "操你", "傻逼", "去死",
]
# 用户消息中的强风险词：命中则风险升为 high 并转人工
HIGH_RISK_USER_WORDS = ["赔偿", "赔款", "起诉", "法院", "曝光", "媒体", "维权", "投诉到", "消协"]
# 答复中绝对化/越界承诺词：命中则说明 LLM 越界，强制转人工复核
ANSWER_BLOCK_WORDS = ["保证退款", "100%退款", "一定退款", "必定退款", "赔偿你", "赔你", "双倍赔偿", "全额赔付"]


def load_env() -> dict:
    for cand in (ROOT / ".env", ROOT.parent / "rag-core" / ".env"):
        if cand.exists():
            for line in cand.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    return {
        "key": os.environ.get("LLM_API_KEY", ""),
        "url": os.environ.get("LLM_API_URL", "").rstrip("/"),
        "model": os.environ.get("LLM_MODEL", ""),
    }


CFG = load_env()


def call_llm(system: str, user: str, temperature: float = 0.3, timeout: int = 60) -> str:
    if not (CFG["key"] and CFG["url"] and CFG["model"]):
        raise RuntimeError("LLM 配置缺失：请填 .env（或确保 ../rag-core/.env 已填 LLM_API_KEY/URL/MODEL）")
    payload = {
        "model": CFG["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        CFG["url"] + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {CFG['key']}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s : e + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# RAG 知识库：本地稀疏检索（字符 bigram 重叠 + 关键词加权）
# ---------------------------------------------------------------------------
def _bigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s)
    return set(s[i : i + 2] for i in range(len(s) - 1))


KNOWLEDGE_RAW = json.loads((ROOT / "knowledge.json").read_text(encoding="utf-8"))["items"]
for _it in KNOWLEDGE_RAW:
    _blob = _it["title"] + " " + _it["content"] + " " + " ".join(_it.get("keywords", []))
    _it["_bg"] = _bigrams(_blob)

ORDERS_RAW = json.loads((ROOT / "orders.json").read_text(encoding="utf-8"))["orders"]


def retrieve(query: str, top_k: int = 3) -> list:
    q_bg = _bigrams(query)
    hits = []
    for it in KNOWLEDGE_RAW:
        score = len(q_bg & it["_bg"])
        for kw in it.get("keywords", []):
            if kw and kw in query:
                score += 5
        if it["title"] in query:
            score += 3
        if score > 0:
            hits.append((score, it))
    hits.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, it in hits[:top_k]:
        out.append({
            "id": it["id"],
            "title": it["title"],
            "category": it["category"],
            "snippet": it["content"][:90],
            "score": score,
        })
    return out


# ---------------------------------------------------------------------------
# 工具（由 Agent Planner 决策调用；request_refund 为高风险，强制转人工）
# ---------------------------------------------------------------------------
def query_order(args: dict) -> dict:
    oid = (args.get("order_id") or "").strip().upper()
    if not oid:
        return {"ok": False, "message": "未提供订单号，无法查询。"}
    for o in ORDERS_RAW:
        if o["order_id"].upper() == oid:
            return {"ok": True, "order": o}
    return {"ok": False, "message": f"未找到订单 {oid}，请核对订单号（可在「我的订单」中查看）。"}


def _write_ticket(path_name: str, record: dict) -> str:
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    p = d / path_name
    tid = hashlib.sha256((path_name + json.dumps(record, ensure_ascii=False)).encode("utf-8")).hexdigest()[:12]
    record = {"ticket_id": tid, **record, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return tid


def request_refund(args: dict) -> dict:
    """高风险动作：不直接执行，生成 pending_human 工单并强制转人工。"""
    oid = (args.get("order_id") or "").strip().upper()
    reason = (args.get("reason") or "").strip()
    tid = _write_ticket("refund_tickets.jsonl", {
        "order_id": oid, "reason": reason,
        "status": "pending_human", "type": "refund",
    })
    return {"ok": True, "forced_human": True, "ticket_id": tid,
            "message": f"退款申请已提交人工审核（工单 {tid}），审核结果将以短信/站内信通知。"}


def escalate_to_human(args: dict) -> dict:
    topic = (args.get("topic") or "人工转接").strip()
    tid = _write_ticket("escalations.jsonl", {
        "topic": topic, "status": "pending_human", "type": "escalation",
    })
    return {"ok": True, "ticket_id": tid,
            "message": f"已为您转接人工客服（工单 {tid}），专员将在 30 分钟内联系您。"}


TOOLS = {"query_order": query_order, "request_refund": request_refund, "escalate_to_human": escalate_to_human}


# ---------------------------------------------------------------------------
# Agent 主流程：Planner（决策）-> 执行检索+工具 -> Answerer（综合生成）-> 风险后处理
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """你是智能客服 Agent 的「决策器」。给定一段对话（含历史），判断下一步动作，只输出一个 JSON 对象（不要 markdown 代码块）：
{
  "thought": "简短分析用户意图",
  "retrieve_keywords": ["用于检索知识库的关键词，最多 3 个"],
  "tools": [{"name": "query_order|request_refund|escalate_to_human", "args": {...}}],
  "risk": "low 或 medium 或 high",
  "needs_human": false,
  "human_reason": "若需转人工，简述原因"
}
规则：
- 用户询问政策/流程（退换货、物流、发票、账户、支付、会员）时，retrieve_keywords 填相关词，tools 可为空。
- 用户报出订单号并问「到哪了/状态/物流」时，调用 query_order，args={"order_id":"..."}。
- 用户明确要求「退款/退货」时，调用 request_refund，args={"order_id":"...","reason":"..."}（系统会强制转人工审核）。
- 用户情绪激动、投诉、威胁曝光/起诉，或你判断需人工介入时，needs_human=true 并调用 escalate_to_human。
- 不要编造订单号；用户未提供订单号时不要臆造。"""


def _summarize_history(messages: list, current: str) -> str:
    lines = []
    for m in messages[:-1]:
        role = "用户" if m["role"] == "user" else "客服"
        lines.append(f"{role}：{m['content']}")
    lines.append("用户（当前）：" + current)
    return "\n".join(lines)


def do_turn(messages: list) -> dict:
    steps = []
    if not messages or messages[-1]["role"] != "user":
        return {"error": "messages 需以用户消息结尾", "steps": steps}
    current = messages[-1]["content"].strip()
    history = _summarize_history(messages, current)

    # ---- 风险硬约束 1：输入黑名单拦截 ----
    block_hit = [w for w in BLOCK_WORDS if w in current]
    if block_hit:
        steps.append({"name": "输入风险拦截", "detail": f"命中敏感词：{block_hit[0]}", "status": "warn"})
        tid = escalate_to_human({"topic": "敏感内容合规转接"})["ticket_id"]
        steps.append({"name": "人工转接", "detail": f"工单 {tid}", "status": "warn"})
        return {
            "reply": "您的问题涉及合规审核，已为您转接人工客服（工单 "
                     f"{tid}），专员将尽快与您联系。请勿在对话中透露密码、验证码等敏感信息。",
            "steps": steps, "sources": [], "tools": [], "risk": "high",
            "needs_human": True, "ticket": tid,
        }

    # ---- Planner 决策 ----
    try:
        plan_text = call_llm(PLANNER_PROMPT, history, temperature=0.0, timeout=60)
        plan = extract_json(plan_text)
    except Exception as e:
        steps.append({"name": "Agent 决策（Planner）", "detail": f"失败，降级为纯检索：{e}", "status": "error"})
        plan = {"thought": "降级", "retrieve_keywords": [], "tools": [], "risk": "low", "needs_human": False, "human_reason": ""}

    # ---- RAG 检索 ----
    q = current
    if plan.get("retrieve_keywords"):
        q = current + " " + " ".join(plan["retrieve_keywords"])
    sources = retrieve(q, top_k=3)
    steps.append({
        "name": "知识库检索（RAG）",
        "detail": f"命中 {len(sources)} 条（{', '.join(s['id'] for s in sources) or '无'}）",
        "status": "ok" if sources else "skip",
    })

    # ---- 工具执行 ----
    tool_results = []
    needs_human = bool(plan.get("needs_human"))
    risk = str(plan.get("risk", "low")).lower()
    ticket = None
    for t in plan.get("tools", []) or []:
        name = t.get("name")
        if name not in TOOLS:
            continue
        try:
            res = TOOLS[name](t.get("args", {}))
        except Exception as e:
            res = {"ok": False, "message": f"工具执行失败：{e}"}
        tool_results.append({"name": name, "result": res})
        steps.append({"name": f"工具调用：{name}", "detail": str(res.get("message", res))[:120], "status": "ok"})
        if res.get("forced_human"):
            needs_human = True
            risk = "high"
            ticket = res.get("ticket_id")
        if res.get("ticket_id") and not ticket:
            ticket = res.get("ticket_id")

    # ---- 风险硬约束 2：用户强风险词 -> high + 转人工 ----
    high_hit = [w for w in HIGH_RISK_USER_WORDS if w in current]
    if high_hit and not needs_human:
        needs_human = True
        risk = "high"
        r = escalate_to_human({"topic": f"高风险诉求：{high_hit[0]}"})
        ticket = r["ticket_id"]
        tool_results.append({"name": "escalate_to_human", "result": r})
        steps.append({"name": "风险升级转人工", "detail": f"命中高风险词：{high_hit[0]}（工单 {ticket}）", "status": "warn"})

    # ---- Answerer 综合生成 ----
    kb_block = "\n".join(f"[{s['id']}|{s['category']}] {s['title']}：{s['snippet']}" for s in sources)
    tool_block = "\n".join(f"- {t['name']} -> {json.dumps(t['result'], ensure_ascii=False)}" for t in tool_results)
    policy = (
        "你只能依据【知识片段】与【工具结果】回答；信息不足时诚实说不知道并建议联系人工，严禁编造政策或承诺具体金额/时效。"
        "语气专业、简洁、有同理心。若 needs_human=true，明确告知用户已进入人工流程，不要替人工承诺结果。"
    )
    answerer_sys = "你是电商智能客服助手。" + policy
    answerer_user = (
        f"对话上下文：\n{history}\n\n"
        f"【知识片段】\n{kb_block or '（无）'}\n\n"
        f"【工具结果】\n{tool_block or '（无）'}\n\n"
        f"needs_human={needs_human}（若为 true，话术须引导用户等待人工，不承诺结果）\n\n"
        f"请仅针对用户当前问题给出最终回复。"
    )
    try:
        reply = call_llm(answerer_sys, answerer_user, temperature=0.3, timeout=60).strip()
    except Exception as e:
        reply = "抱歉，当前客服系统繁忙，已为您记录（稍后人工跟进）。" if needs_human else "抱歉，暂时无法回答，请稍后再试或联系人工客服。"
        steps.append({"name": "生成回复（Answerer）", "detail": f"失败：{e}", "status": "error"})

    # ---- 风险硬约束 3：答复越界承诺检测 ----
    answer_block = [w for w in ANSWER_BLOCK_WORDS if w in reply]
    if answer_block and not needs_human:
        needs_human = True
        risk = "high"
        r = escalate_to_human({"topic": f"答复含越界承诺：{answer_block[0]}"})
        ticket = r["ticket_id"]
        reply = "关于您的诉求，涉及具体赔付/退款处理，已为您转接人工专员核实（工单 " + ticket + "），将由专员与您确认方案，我不便直接承诺。"
        steps.append({"name": "答复风险复核", "detail": f"检测到越界承诺词：{answer_block[0]}，已转人工", "status": "warn"})

    steps.append({"name": "生成回复", "detail": f"{len(reply)} 字 · 风险 {risk}" + (" · 转人工" if needs_human else ""), "status": "ok"})
    return {
        "reply": reply, "steps": steps, "sources": sources,
        "tools": tool_results, "risk": risk, "needs_human": needs_human, "ticket": ticket,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path == "/api/chat":
            try:
                n = int(self.headers.get("Content-Length", 0))
                inp = json.loads(self.rfile.read(n).decode("utf-8"))
                res = do_turn(inp.get("messages", []))
                code = 200 if "error" not in res else 400
                self._send(code, json.dumps(res, ensure_ascii=False))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            self._send(404, b"not found")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"智能客服 Agent 已启动 -> http://localhost:{PORT}")
    print(f"LLM 配置: model={CFG['model'] or '(未配置)'}  url={CFG['url'] or '(未配置)'}")
    if not (CFG["key"] and CFG["url"] and CFG["model"]):
        print("[警告] LLM 未配置，请填 .env 后再使用 /api/chat")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
