#!/usr/bin/env python3
"""web_demo.py · rag-core 医疗 RAG 网页演示件

复用 rag-core 已有的 Retriever + Generator，做成可现场演示的网页：
  GET  /          演示界面（web_demo.html）
  POST /api/ask   {question, top_k?} -> {answer, sources, hits[], trace_id, error?}

- 零额外 Web 框架（Python 标准库 http.server）
- faiss-cpu / sentence_transformers / openai 等重依赖延迟到首次请求时 import，
  以便在未装包的沙箱里也能启动、验证静态界面（POST 会返回友好错误）
- 必须在 rag-core 的 Python 环境下运行（含依赖 + models/bge-m3 + 已 ingest 的 FAISS 向量库）

用法：
  cd rag-core
  python web_demo.py --case medical_qa --port 8001
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8001
DEFAULT_CASE = "medical_qa"

_state = {"case": DEFAULT_CASE, "port": DEFAULT_PORT, "retriever": None, "lock": threading.Lock()}


def _ensure_imports() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))


def get_retriever():
    if _state["retriever"] is None:
        with _state["lock"]:
            if _state["retriever"] is None:
                _ensure_imports()
                from rag_core.retriever import Retriever
                _state["retriever"] = Retriever(_state["case"])
    return _state["retriever"]


def do_ask(question: str, top_k):
    try:
        _ensure_imports()
        from rag_core.retriever import Retriever
        from rag_core.generator import generate

        retriever = get_retriever()
        hits = retriever.retrieve(question, top_k=top_k) if top_k else retriever.retrieve(question)
        gen = generate(question, hits)
        return {
            "answer": gen.get("answer", ""),
            "sources": gen.get("sources", ""),
            "grounded": gen.get("grounded", False),
            "trace_id": gen.get("trace_id", ""),
            "hit_count": len(hits),
            "hits": [
                {
                    "score": h.score,
                    "department": h.department,
                    "question": h.question,
                    "text": (h.text or "")[:240],
                }
                for h in hits
            ],
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "hint": "请在 rag-core 的 Python 环境下运行本服务，且已执行 `python run.py ingest` "
                    "建好 FAISS 向量库（models/bge-m3 与 .env 的 LLM 配置就绪）。",
        }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/web_demo.html"):
            html = (ROOT / "web_demo.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path == "/api/ask":
            try:
                length = int(self.headers.get("Content-Length", 0))
                inp = json.loads(self.rfile.read(length).decode("utf-8"))
                question = (inp.get("question") or "").strip()
                top_k = inp.get("top_k")
                if not question:
                    self._send(400, json.dumps({"error": "question 不能为空"}, ensure_ascii=False).encode("utf-8"))
                    return
                result = do_ask(question, top_k)
                code = 200 if "error" not in result else 500
                self._send(code, json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, b"not found")

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="rag-core 医疗 RAG 网页演示件")
    ap.add_argument("--case", default=DEFAULT_CASE, help="案例名，默认 medical_qa")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="端口，默认 8001")
    args = ap.parse_args()
    _state["case"] = args.case
    _state["port"] = args.port
    print(f"RAG 演示件已启动 -> http://localhost:{args.port}  (case={args.case})")
    print("提示：首次提问会加载 bge-m3 模型 + FAISS 向量库，请稍候数秒。")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
