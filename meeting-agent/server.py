#!/usr/bin/env python3
"""meeting-agent · 会议纪要→任务工单 Agent

把会议纪要→任务工单 Agent 落地为可演示成品：
  会议转录文本 -> LLM 抽取 action items -> 逐个调用 create_ticket 工具（mock 可换真实 API）-> 生成纪要摘要
展示 Agent 编排 + 工具调用，把"了解 Agent 任务拆解"变成可演示作品。

零额外依赖（Python 标准库 http.server + urllib）：
  GET  /               演示界面（index.html）
  POST /api/process    {text} -> {steps, extracted, tickets, summary}

LLM 配置：优先 meeting-agent/.env，回退 ../rag-core/.env（复用已填配置）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
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


PORT = _resolve_port("MEETING_PORT", "8003")

SYSTEM_PROMPT = """你是会议纪要 Agent，负责把会议转录解析为可执行的 action items，并准备创建工单。
只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字）：
{
  "summary": "一段话概括本次会议结论与重点",
  "tasks": [
    {"title": "任务标题", "owner": "负责人", "due": "截止日期 YYYY-MM-DD（无则空字符串）",
     "priority": "高" 或 "中" 或 "低", "detail": "任务细节"}
  ]
}
如果没有明确任务则 tasks 为空数组。负责人必须从转录中推断，不要臆造不存在的人名。
输出应稳定可复现：相对日期（如"下周五"）必须基于会议日期转换为具体的 YYYY-MM-DD；字段措辞保持一致。"""


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


def call_llm(text: str) -> dict:
    if not (CFG["key"] and CFG["url"] and CFG["model"]):
        raise RuntimeError("LLM 配置缺失：请填 .env（或确保 ../rag-core/.env 已填 LLM_API_KEY/URL/MODEL）")
    payload = {
        "model": CFG["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "会议转录：\n" + text},
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(
        CFG["url"] + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {CFG['key']}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    s, e = content.find("{"), content.rfind("}")
    if s != -1 and e != -1 and e > s:
        content = content[s : e + 1]
    return json.loads(content)


def create_ticket(task: dict, raw: str, idx: int) -> dict:
    """工具：创建工单（当前为本地 mock；换真实 API 只需改此处）。

    幂等键以「转录内容 + 任务序号」为准，避免 LLM 表述/相对日期波动导致同会议重复建单。
    """
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    p = d / "tickets.jsonl"
    existing = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["ticket_id"]] = r
    tid = hashlib.sha256(f"{raw}|{idx}".encode("utf-8")).hexdigest()[:16]
    if tid in existing:
        return {"ticket_id": tid, "status": "exists", "action": "复用已存在工单"}
    rec = {"ticket_id": tid, **task}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ticket_id": tid, "status": "created", "action": "已创建工单"}


def do_process(text: str) -> dict:
    steps = []
    raw = text or ""
    steps.append({"name": "解析转录", "detail": f"原始转录 {len(raw)} 字符", "status": "ok"})
    try:
        ext = call_llm(raw)
    except Exception as e:
        steps.append({"name": "LLM 抽取 action items", "detail": f"失败：{e}", "status": "error"})
        return {"steps": steps, "error": str(e),
                "hint": "请在已装 Python 且 .env 已配置 LLM 的环境下运行本服务。"}
    tasks = ext.get("tasks") or []
    steps.append({"name": "LLM 抽取 action items", "detail": f"识别出 {len(tasks)} 个任务", "status": "ok"})
    tickets = []
    for i, t in enumerate(tasks, 1):
        res = create_ticket(t, raw, i)
        tickets.append(res)
        steps.append({
            "name": f"调用 create_ticket 工具（任务 {i}）",
            "detail": f"{t.get('title', '?')[:18]} → {res['status']}",
            "status": "ok" if res["status"] == "created" else "skip",
        })
    steps.append({"name": "生成纪要摘要", "detail": (ext.get("summary", "") or "")[:40], "status": "ok"})
    return {"steps": steps, "extracted": ext, "tickets": tickets, "summary": ext.get("summary", "")}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
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
        if self.path == "/api/process":
            try:
                n = int(self.headers.get("Content-Length", 0))
                inp = json.loads(self.rfile.read(n).decode("utf-8"))
                res = do_process(inp.get("text", ""))
                code = 200 if "error" not in res else 500
                self._send(code, json.dumps(res, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, b"not found")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"会议纪要→工单 Agent 已启动 -> http://localhost:{PORT}")
    print(f"LLM 配置: model={CFG['model'] or '(未配置)'}  url={CFG['url'] or '(未配置)'}")
    if not (CFG["key"] and CFG["url"] and CFG["model"]):
        print("[警告] LLM 未配置，请填 .env 后再使用 /api/process")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
