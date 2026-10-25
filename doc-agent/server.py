#!/usr/bin/env python3
"""doc-agent · 文档/采购单自动化处理 Agent

把大规模数据治理流水线升级为端到端自动化 Agent 的演示件：
  文档/采购单文本 -> LLM 结构化抽取 -> 规则校验 -> 写入数据池（幂等 + 版本指纹）

零额外依赖（Python 标准库 http.server + urllib）：
  GET  /               演示界面（index.html）
  POST /api/process    {text, doc_type?} -> {steps, extracted, validation, store, summary}

LLM 配置：优先 doc-agent/.env，回退 ../rag-core/.env（复用已填配置）。
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


PORT = _resolve_port("DOC_PORT", "8002")

SYSTEM_PROMPT = """你是文档自动化处理 Agent，专门从采购单/票据/合同等非结构化文本中抽取结构化字段。
只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字）：
{
  "doc_type": "采购单" 或 "票据" 或 "合同" 或 "其他",
  "supplier": "供应商/开票方名称（无则空字符串）",
  "date": "日期 YYYY-MM-DD（从文本推断，无则空）",
  "currency": "币种如 CNY/USD（无则空）",
  "items": [
    {"name": "物品名", "qty": 数量数字, "unit_price": 单价数字, "subtotal": 小计数字}
  ],
  "total": 合计数字（无则 0）,
  "summary": "一句话概括这份文档的业务内容"
}
金额必须基于文本中的数字，不要臆造。qty/unit_price/subtotal/total 都输出数值（不带单位与千分位逗号）。"""


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
            {"role": "user", "content": "待处理文档：\n" + text},
        ],
        "temperature": 0.1,
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


def validate(ext: dict) -> dict:
    issues = []
    if not (ext.get("supplier") or "").strip():
        issues.append("缺少供应商/开票方")
    if not (ext.get("date") or "").strip():
        issues.append("缺少日期")
    items = ext.get("items") or []
    if not items:
        issues.append("无物品明细")
    try:
        tf = float(ext.get("total") or 0)
        ss = sum(float(it.get("subtotal", 0)) for it in items)
        if abs(ss - tf) > 0.01:
            issues.append(f"金额不一致：明细小计合计 {ss:.2f} 与合计 {tf:.2f} 不符")
        for it in items:
            q = float(it.get("qty", 0))
            u = float(it.get("unit_price", 0))
            s = float(it.get("subtotal", 0))
            if abs(q * u - s) > 0.01:
                issues.append(f"物品「{it.get('name')}」小计 {s} 与 数量×单价 {q * u:.2f} 不符")
    except Exception:
        issues.append("金额字段非数值，无法校验")
    return {"status": "ok" if not issues else "review", "issues": issues}


def store(ext: dict, raw_text: str) -> dict:
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    p = d / "records.jsonl"
    existing = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(json.loads(line)["record_id"])
    rid = hashlib.sha256(raw_text.strip().encode("utf-8")).hexdigest()[:32]
    if rid in existing:
        return {"stored": False, "reason": "幂等跳过：相同文档已入库", "record_id": rid}
    rec = {
        "record_id": rid,
        "doc_type": ext.get("doc_type", ""),
        "supplier": ext.get("supplier", ""),
        "date": ext.get("date", ""),
        "currency": ext.get("currency", ""),
        "items": ext.get("items", []),
        "total": ext.get("total", 0),
        "summary": ext.get("summary", ""),
        "raw_len": len(raw_text),
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ids = sorted(existing | {rid})
    version = hashlib.sha256("".join(ids).encode("utf-8")).hexdigest()[:16]
    (d / "version.json").write_text(
        json.dumps({"version": version, "count": len(ids)}, ensure_ascii=False)
    )
    return {"stored": True, "record_id": rid, "version": version, "count": len(ids)}


def do_process(text: str) -> dict:
    steps = []
    raw = text or ""
    steps.append({"name": "解析文档", "detail": f"原始文本 {len(raw)} 字符", "status": "ok"})
    try:
        ext = call_llm(raw)
    except Exception as e:
        steps.append({"name": "LLM 结构化抽取", "detail": f"失败：{e}", "status": "error"})
        return {"steps": steps, "error": str(e),
                "hint": "请在已装 Python 且 .env 已配置 LLM 的环境下运行本服务。"}
    nitems = len(ext.get("items") or [])
    steps.append({
        "name": "LLM 结构化抽取",
        "detail": f"doc_type={ext.get('doc_type', '?')}，{nitems} 条物品，合计 {ext.get('total', 0)}",
        "status": "ok",
    })
    val = validate(ext)
    steps.append({
        "name": "规则校验",
        "detail": "通过" if val["status"] == "ok" else f"{len(val['issues'])} 项待复核",
        "status": val["status"],
    })
    st = store(ext, raw)
    steps.append({
        "name": "写入数据池",
        "detail": (st.get("reason") if not st.get("stored")
                   else f"record_id={st['record_id'][:8]}… 版本 {st.get('version')} 共 {st.get('count')} 条"),
        "status": "ok" if st.get("stored") else "skip",
    })
    summary = ext.get("summary", "")
    steps.append({"name": "生成处理报告", "detail": summary, "status": "ok"})
    return {"steps": steps, "extracted": ext, "validation": val, "store": st, "summary": summary}


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
    print(f"文档自动化 Agent 已启动 -> http://localhost:{PORT}")
    print(f"LLM 配置: model={CFG['model'] or '(未配置)'}  url={CFG['url'] or '(未配置)'}")
    if not (CFG["key"] and CFG["url"] and CFG["model"]):
        print("[警告] LLM 未配置，请填 .env 后再使用 /api/process")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
