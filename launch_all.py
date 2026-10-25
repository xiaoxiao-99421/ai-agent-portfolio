#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_all.py · 一键启动求职作品集的全部本地服务

用法（在 求职作品集/ 目录下）：
    python launch_all.py
    python launch_all.py --no-browser      # 不自动打开浏览器
    python launch_all.py --only rag,support  # 只启动指定服务（名称见 SERVICES）

说明：
- doc-agent / meeting-agent / support-agent 仅依赖 Python 标准库，用当前 python 即可启动。
- 求职投递助手（sdr-agent）与企业知识库问答 Agent（agent_kb_qa）均属个人工具，
  已移至 job/个人工具/，不通过本脚本启动；如需使用请进入对应目录自行运行。
- rag-core 的 web_demo 需要 faiss-cpu / sentence-transformers，
  脚本会自动探测并使用 rag-core/.venv 里的解释器（不存在则回退当前 python，
  此时 RAG 演示可能需自行在 rag-core 环境启动）。
- 启动后自动打开 portfolio/index.html（作品集入口），其「打开演示」按钮
  指向各服务端口，服务已在运行时即可直接访问。
- 按 Ctrl+C 可一次性停止全部服务。
"""
import argparse
import os
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))          # job/
PORTFOLIO = os.path.join(ROOT, "portfolio")


def venv_python():
    """探测 rag-core 的虚拟环境解释器。"""
    cands = [
        os.path.join(ROOT, "rag-core", ".venv", "Scripts", "python.exe"),
        os.path.join(ROOT, "rag-core", ".venv", "bin", "python"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


# (名称, 目录, 脚本, 端口, 额外参数) —— 仅含求职作品集演示件，不含 sdr-agent
SERVICES = [
    ("rag",      "rag-core",       "web_demo.py",  8001, ["--case", "medical_qa"]),
    ("doc",      "doc-agent",      "server.py",    8002, []),
    ("meeting",  "meeting-agent",  "server.py",    8003, []),
    ("support",  "support-agent",  "server.py",    8005, []),
]


def main():
    ap = argparse.ArgumentParser(description="一键启动求职作品集本地服务")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--only", default="", help="只启动指定服务，逗号分隔，如 rag,support")
    args = ap.parse_args()

    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}

    vpy = venv_python()
    procs = []

    print("=" * 60)
    print("  求职作品集 · 一键启动")
    print("=" * 60)

    for key, name, script, port, extra in SERVICES:
        if only and key not in only:
            continue
        cwd = os.path.join(ROOT, name)
        if not os.path.isdir(cwd):
            print(f"  [跳过] {name}：目录不存在 ({cwd})")
            continue
        py = vpy if name == "rag-core" else sys.executable
        if name == "rag-core" and vpy is None:
            print(f"  [警告] rag-core/.venv 未找到，将用当前 python 启动 web_demo，"
                  f"若缺 faiss-cpu/sentence-transformers 会失败；可改在 rag-core 环境手动启动。")
        cmd = [py, script, "--port", str(port)] + extra
        log_path = os.path.join(PORTFOLIO, f"{name}_{port}.log")
        logf = open(log_path, "w", encoding="utf-8")
        try:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"  [失败] {name} 启动异常：{e}")
            logf.close()
            continue
        procs.append((name, port, p))
        print(f"  ✓ {name:<13} http://localhost:{port:<5} (pid {p.pid})")

    if not procs:
        print("  没有可启动的服务，退出。")
        return

    print("-" * 60)
    print(f"  已启动 {len(procs)} 个服务；日志见 portfolio/*.log")
    if not args.no_browser:
        time.sleep(2)
        landing = os.path.join(PORTFOLIO, "index.html")
        if os.path.exists(landing):
            webbrowser.open("file://" + landing)
            print("  已打开作品集页面（portfolio/index.html）")
    print("  按 Ctrl+C 停止全部服务…")

    try:
        while True:
            time.sleep(1)
            # 检测是否有服务意外退出
            for name, port, p in procs:
                if p.poll() is not None:
                    print(f"  [注意] {name} (端口 {port}) 已退出，详见 portfolio/{name}_{port}.log")
    except KeyboardInterrupt:
        print("\n  正在停止全部服务…")
        for name, port, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for name, port, p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        print("  已全部停止。")


if __name__ == "__main__":
    main()
