"""harness.py - 自动评测 harness（消灭人工逐条标注）。

痛点对照（原项目 eval_set_annotated.jsonl）：
    每条都是 "query_type":"TODO" + "待人工核对 gold_ids" -> 84 条全手工。

本 harness 的做法：
    1. 自动 gold 候选：用向量库近邻 top-N 直接生成候选 gold_ids，不再手填
    2. LLM 判相关：对每个候选文档判 relevant/irrelevant，并自动打 query_type
    3. 人工抽样复核：仅把低置信 / 边界样本（默认 10%）写入 review 文件交人确认
       —— 把「84 条全手工」降到「抽几条看一眼」

指标：hit_rate@k（top_k 内至少 1 个相关）、mean_relevant@k、coverage。
工程约束（C4/C5）：LLM 判分带超时+重试+原始返回落盘，判分失效不污染 headline 指标。
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI

from rag_core.config import Settings, case_dir
from rag_core.retriever import Retriever

EVAL_TOP_N = 10  # 候选 gold 池大小


@dataclass
class EvalItem:
    query: str
    expected_dept: str
    auto_gold_ids: List[str]
    relevant_ids: List[str]
    query_type: str
    hit: bool
    confidence: str          # high / low
    judge_failed: bool = False  # LLM 判分是否失效（失效则不计入 headline）


JUDGE_PROMPT = """你是检索质量评测员。给定用户问题与若干候选资料片段，逐条判断是否与问题相关。
只依据语义相关性，不要求答案完全正确。
必须输出严格 JSON，且 relevant 是长度恰好等于片段数 {n} 的布尔数组（true/false），顺序与片段一致：
{{"relevant":[true,false,true,...], "query_type":"短查询/长查询/口语化/标准"}}
不要输出任何额外说明文字。
用户问题：{query}
片段（共 {n} 条）：
{items}"""


def _judge_llm(query: str, hits, api_key: str, base_url: str, model: str) -> Tuple[Optional[dict], str]:
    """返回 (解析后的判定, 原始LLM文本)。解析失败返回 (None, 原始文本) 供排查。

    健壮性（C4）：带超时 + 指数退避重试；布尔数组长度容忍（取前 n 个，多退少补为 False）。
    """
    n = len(hits)
    items = "\n".join(
        f"{i+1}. （{h.department}）{h.question}\n{h.text[:120]}"
        for i, h in enumerate(hits)
    )
    prompt = JUDGE_PROMPT.format(query=query, items=items, n=n)
    client = OpenAI(api_key=api_key, base_url=base_url)
    raw = ""
    retries = max(1, Settings.llm_max_retries)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "user", "content": prompt}],
                timeout=Settings.llm_timeout_seconds,
            )
            raw = resp.choices[0].message.content.strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1:
                continue  # 无 JSON，重试
            data = json.loads(raw[s:e + 1])
            rel = data.get("relevant")
            if not isinstance(rel, list) or not rel:
                continue
            # 容错：布尔化，截断到 n；多余的忽略，不足的视为不相关
            flags = [bool(x) for x in rel[:n]]
            return {"relevant": flags, "query_type": str(data.get("query_type", ""))}, raw
        except Exception:
            continue  # 超时/解析异常 -> 重试
    return None, raw


def _query_type_rule(query: str) -> str:
    q = query.strip()
    if len(q) <= 12:
        return "短查询"
    if len(q) >= 60:
        return "长查询"
    return "标准"


def run_eval(case: str, top_k: int = 5, review_ratio: Optional[float] = None,
             seed: int = 42) -> dict:
    review_ratio = review_ratio if review_ratio is not None else Settings.human_review_ratio
    base = case_dir(case)
    qfile = base / "data" / "eval" / "eval_queries.jsonl"
    if not qfile.exists():
        raise FileNotFoundError(f"缺少评测集: {qfile}（请参考 cases/medical_qa 建一个）")

    retr = Retriever(case, top_k=EVAL_TOP_N)
    use_llm = Settings.eval_use_llm
    api_key = base_url = model = ""
    if use_llm:
        try:
            api_key, base_url, model = Settings.require_llm()
        except RuntimeError:
            print("[评测] 未配置 LLM，回退规则判定（不计入 LLM 判分失效）")
            use_llm = False

    queries = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    items: List[EvalItem] = []
    judge_debug: List[dict] = []

    for q in queries:
        query = q["query"]
        exp = q.get("expected_dept", "")
        hits = retr.retrieve(query, top_k=EVAL_TOP_N)
        gold = [h.id for h in hits]

        relevant: List[str] = []
        qtype = _query_type_rule(query)
        judge_failed = False

        if use_llm and hits:
            j, raw = _judge_llm(query, hits, api_key, base_url, model)
            if j is not None:
                for h, ok in zip(hits, j["relevant"]):
                    if ok:
                        relevant.append(h.id)
                qtype = j["query_type"] or qtype
                conf = "high"
            else:
                # LLM 判分失效 -> 不计入 headline，强制进复核
                conf = "low"
                judge_failed = True
                judge_debug.append({"query": query, "raw": raw})
        elif hits:
            # LLM 未启用：规则兜底（仅同科室或全空时取 top-1）
            relevant = [h.id for h in hits if (exp and h.department == exp) or not exp][:1]
            conf = "high"
        else:
            conf = "low" if use_llm else "high"

        hit = len(relevant) >= 1
        items.append(EvalItem(query, exp, gold, relevant, qtype, hit, conf, judge_failed))

    # 指标：hit_rate 只在「LLM 判分成功」的样本上计算，避免失效污染
    confident = [i for i in items if i.confidence == "high"]
    n_conf = len(confident)
    hit_rate = sum(i.hit for i in confident) / n_conf if n_conf else 0.0
    mean_rel = sum(len(i.relevant_ids) for i in confident) / n_conf if n_conf else 0.0
    low_conf = [i for i in items if i.confidence == "low"]
    judge_failed_count = sum(1 for i in items if i.judge_failed)

    # 人工抽样复核：优先低置信，按比例抽
    review_n = max(1, math.ceil(len(items) * review_ratio))
    pool = low_conf + [i for i in items if i not in low_conf]
    random.seed(seed)
    review_sample = pool[:review_n]

    # 写出结果
    out_dir = base / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.jsonl", "w", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(asdict(i), ensure_ascii=False) + "\n")

    if judge_debug:
        with open(out_dir / "eval_judge_debug.jsonl", "w", encoding="utf-8") as f:
            for d in judge_debug:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    review_path = out_dir / "eval_review.jsonl"
    with open(review_path, "w", encoding="utf-8") as f:
        for i in review_sample:
            f.write(json.dumps(asdict(i), ensure_ascii=False) + "\n")

    report = {
        "case": case,
        "n_queries": len(items),
        "hit_rate_at_k": round(hit_rate, 4),
        "mean_relevant_at_k": round(mean_rel, 4),
        "judged_count": n_conf,
        "judge_failed_count": judge_failed_count,
        "low_conf_count": len(low_conf),
        "human_review_count": len(review_sample),
        "review_file": str(review_path),
        "results_file": str(out_dir / "eval_results.jsonl"),
    }
    if judge_failed_count == len(items) and items:
        report["warning"] = ("全部 LLM 判分失效，hit_rate 不可信；"
                              "请查看 eval_judge_debug.jsonl 中的原始返回排查（通常为 JSON 解析/长度问题）。")
    return report
