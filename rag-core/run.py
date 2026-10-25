"""run.py - 项目统一入口。

用法（注意：--case 直接写值，不带方括号；默认 case 即 medical_qa，可省略）：
    uv run python run.py data --case medical_qa          # 跑数据流水线
    uv run python run.py ingest --case medical_qa        # 向量化入库（版本跳过重建）
    uv run python run.py ask "你的问题" --case medical_qa          # 问答演示（加 --agent 走 Agent 层）
    uv run python run.py eval --case medical_qa          # 自动评测 + 抽样复核
    uv run python run.py dept-report --case medical_qa   # 科室分布与配置缺口报告

说明：首次运行需在项目根创建 .env（cp .env.example .env 并填值）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def cmd_data(case: str) -> None:
    from rag_core.pipeline import DataPipeline
    from rag_core.config import case_dir
    p = DataPipeline(config_path="data_sources.json", base_dir=str(case_dir(case)))
    p.run()


def cmd_stats(case: str) -> None:
    """统计每个数据源原始 CSV 的总行数（不含表头），帮助决定 limit 写多少。

    更省事的做法：把 data_sources.json 里的 limit 删掉或设 null，即读全量。
    """
    import csv
    from pathlib import Path

    from rag_core.config import case_dir

    base = case_dir(case)
    cfg = json.loads((base / "data_sources.json").read_text(encoding="utf-8"))
    print(f"\n=== 数据源行数统计 (case: {case}) ===")
    total = 0
    for src in cfg.get("sources", []):
        fp = base / "data" / "raw" / src["path"]
        if not fp.exists():
            print(f"  [缺] {src['name']:<16} 文件不存在: {src['path']}")
            continue
        # 兼容多编码：先按配置编码，失败回退 gb18030（与 pipeline 一致）
        enc = src.get("encoding", "utf-8")
        try:
            with open(fp, encoding=enc, newline="") as f:
                n = sum(1 for _ in f) - 1  # 减表头
        except (UnicodeDecodeError, Exception):
            with open(fp, encoding="gb18030", newline="") as f:
                n = sum(1 for _ in f) - 1
        cur_limit = src.get("limit")
        lim_str = "全量" if cur_limit in (None, "") else f"限 {cur_limit}"
        print(f"  {src['name']:<16} 共 {n} 条  (当前 {lim_str})")
        total += n
    print(f"  合计约 {total} 条")
    print(f"\n提示：把 limit 删掉或设 null 即读全量，无需手填具体数字。")


def cmd_ingest(case: str, force_rebuild: bool = False,
               batch_size: int = None, encode_batch: int = None,
               checkpoint_every: int = None) -> None:
    from rag_core.ingest import build as ingest_build
    # 只透传用户显式指定的参数；未指定则用 ingest() 签名里的默认值（单一来源）。
    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if encode_batch is not None:
        kwargs["encode_batch"] = encode_batch
    if checkpoint_every is not None:
        kwargs["checkpoint_every"] = checkpoint_every
    ingest_build(case, force_rebuild=force_rebuild, **kwargs)


def cmd_ask(query: str, case: str, agent: bool) -> None:
    from rag_core.retriever import Retriever
    from rag_core.generator import generate
    from rag_core.agent import RAGAgent
    if agent:
        res = RAGAgent(case).run(query)
    else:
        hits = Retriever(case).retrieve(query)
        res = generate(query, hits)
    print("\n=== 回答 ===")
    print(res.get("answer", ""))
    if res.get("sources"):
        print("\n=== 来源 ===")
        print(res["sources"])
    if res.get("trace_id"):
        print(f"\n=== trace_id ===\n{res['trace_id']}")
    if res.get("steps"):
        print("\n=== 步骤 ===")
        print(json.dumps(res["steps"], ensure_ascii=False))


def cmd_eval(case: str) -> None:
    sys.path.insert(0, str(ROOT))
    from eval.harness import run_eval
    report = run_eval(case)
    print("\n=== 评测报告 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_dept_report(case: str) -> None:
    from rag_core.pipeline import DataPipeline
    from rag_core.config import case_dir
    p = DataPipeline(config_path="data_sources.json", base_dir=str(case_dir(case)))
    p.department_report()


def cmd_verify(case: str) -> None:
    """向量库健康检查：确认已入库条数与数据集一致、索引未损坏。"""
    import json as _json

    from rag_core.ingest import check_health

    rep = check_health(case)
    print("=== 向量库健康检查 ===")
    print(_json.dumps(rep, ensure_ascii=False, indent=2))
    if not rep.get("ok"):
        print("\n修复：python run.py ingest --case %s --force-rebuild" % case)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["data", "ingest", "ask", "eval", "dept-report", "verify", "stats"])
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--case", default="medical_qa")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="ingest 时清空旧集合全量重建（仅 embedding 模型变更时用）")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="ingest 外层批次：一次读多少条（默认见 ingest.py 签名，影响内存、几乎不影响速度）")
    ap.add_argument("--encode-batch", type=int, default=None,
                    help="ingest 内层 encode 批次：torch 一次算多少条（默认见 ingest.py，fp32 下越大越快但更吃内存）")
    ap.add_argument("--checkpoint-every", type=int, default=None,
                    help="ingest 每多少批落盘一次（默认见 ingest.py，越小越抗崩溃）")
    ap.add_argument("--agent", action="store_true")
    args = ap.parse_args()

    if args.command == "data":
        cmd_data(args.case)
    elif args.command == "ingest":
        cmd_ingest(args.case, args.force_rebuild,
                   batch_size=args.batch_size, encode_batch=args.encode_batch,
                   checkpoint_every=args.checkpoint_every)
    elif args.command == "ask":
        if not args.query:
            ap.error("ask 需要加上问题，例如：run.py ask \"孩子发烧怎么办\"")
        cmd_ask(args.query, args.case, args.agent)
    elif args.command == "eval":
        cmd_eval(args.case)
    elif args.command == "dept-report":
        cmd_dept_report(args.case)
    elif args.command == "verify":
        cmd_verify(args.case)
    elif args.command == "stats":
        cmd_stats(args.case)


if __name__ == "__main__":
    main()
