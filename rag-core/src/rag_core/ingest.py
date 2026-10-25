"""ingest.py - 校验 + 分批向量化 + FAISS 入库（增量 add，支持分块续跑）。

设计要点（生产级）：
    - 入库前强校验：必需字段、id 唯一、版本一致、记录数一致，任一不符即终止（不静默）
    - 增量：只向量化并写入「索引里尚不存在的 id」，已有 id 跳过（分块续跑累加）
    - 版本未变且索引记录数一致 -> 跳过（无操作快速路径）
    - --force-rebuild：清空索引全量重建（仅在 embedding 模型变更时使用）
    - 分批写入，批大小可配；批失败重试 2 次，不静默跳过
    - 嵌入模型路径来自配置（环境变量），不硬编码
    - 落盘：FAISS 显式 write_index + pickle（save），不依赖任何外部析构 / 后台线程
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import PROJECT_ROOT, Settings, case_dir, vector_dir_for
from .vector_store import FaissStore

# 优雅中断：收到 Ctrl+C / 终止信号时置位，当前批次写完后安全退出，
# 避免进程在写索引的中途被强杀。已写入的批次通过显式 save 落盘，下次增量续跑。
_INTERRUPTED = {"flag": False}


def _install_signal_handlers() -> None:
    def handler(signum, frame):  # noqa: ARG001
        if _INTERRUPTED["flag"]:
            raise SystemExit(1)  # 再次中断则立即退出
        _INTERRUPTED["flag"] = True
        print("\n[中断] 收到停止信号，将在当前批次写入完成后安全退出…")

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except Exception:
                pass


def _paths(case: str):
    base = case_dir(case)
    jsonl = base / "data" / "processed" / "rag_dataset.jsonl"
    version_file = base / "data" / "processed" / "data_version.json"
    vec_dir = vector_dir_for(case)  # 向量库目录（data/vectors，已被 .gitignore 覆盖）
    return jsonl, version_file, vec_dir


def load_version(version_file: Path) -> dict:
    with open(version_file, encoding="utf-8") as f:
        return json.load(f)


def validate_records(jsonl: Path, ver: dict) -> tuple[int, str]:
    if not ver.get("version"):
        raise ValueError("版本文件 version 为空，已取消入库")
    required = {"id", "question", "department", "title", "answer"}
    seen, n = set(), 0
    with open(jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"第 {i} 行不是对象，已取消")
            miss = required - rec.keys()
            if miss:
                raise ValueError(f"第 {i} 行缺字段 {sorted(miss)}，已取消")
            if rec.get("data_version") != ver["version"]:
                raise ValueError(f"第 {i} 行 data_version 与版本文件不一致")
            rid = rec["id"]
            if not isinstance(rid, str) or not rid.strip():
                raise ValueError(f"第 {i} 行 id 无效")
            if rid in seen:
                raise ValueError(f"第 {i} 行 id 重复: {rid}")
            seen.add(rid)
            if not str(rec["question"]).strip() or not str(rec["answer"]).strip():
                raise ValueError(f"第 {i} 行 question/answer 为空")
            n += 1
    if n == 0:
        raise ValueError("数据集为空，已取消")
    if n != ver["record_count"]:
        raise ValueError(f"记录数 {n} 与版本文件 {ver['record_count']} 不一致")
    return n, ver["version"]


def _batch(jsonl: Path, size: int):
    buf = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            buf.append(json.loads(line))
            if len(buf) == size:
                yield buf
                buf = []
    if buf:
        yield buf


def _mk_meta(rec: dict) -> dict:
    return {
        "question": rec.get("question", ""),
        "department": rec.get("department", ""),
        "major_department": rec.get("major_department", ""),
        "cross_dept": bool(rec.get("cross_dept", False)),
        "title": rec.get("title", ""),
        "data_version": rec.get("data_version", ""),
    }


def load_embedding_model():
    """加载嵌入模型（bge-m3），带重试与底层错误透出。

    背景：transformers 5.x 的 AutoProcessor.from_pretrained 在内部 fallback
    （AutoTokenizer -> AutoImageProcessor -> ...）中会把每一环的真实失败原因
    吞掉，最终只报笼统的 "Unrecognized processing class"。本机实测其底层原因
    是 Windows 提交内存（commit）瞬时不足（os error 1455 / MemoryError）——
    间歇性资源故障，重试等窗口即可恢复。
    另：torch 默认按全部核心开线程池，每线程都有栈与缓冲开销，这里限制线程数
    以降低加载与 encode 的内存峰值。
    因此：失败时重试（指数退避封顶 30s），并主动探测 AutoTokenizer 的真实错误
    帮助定位；连续失败才抛出，不静默交付半初始化的模型。
    """
    try:
        import torch

        torch.set_num_threads(min(os.cpu_count() or 4, 8))
        dtype = torch.float32  # fp32：CPU 上 bge-m3 的唯一合理档位（bf16 无算子加速、慢 7 倍）
    except Exception:
        dtype = None

    last: Exception | None = None
    for attempt in range(1, 7):
        try:
            # low_cpu_mem_usage：按需映射权重，降低加载峰值内存（比默认省约一半）。
            model_kwargs = {"low_cpu_mem_usage": True}
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            return SentenceTransformer(
                Settings.resolve_model_path(),
                local_files_only=True,
                model_kwargs=model_kwargs,
            )
        except Exception as e:
            last = e
            reason = f"{type(e).__name__}: {str(e)[:160]}"
            try:
                from transformers import AutoTokenizer

                AutoTokenizer.from_pretrained(
                    Settings.resolve_model_path(), local_files_only=True
                )
                detail = "tokenizer 单独加载正常（属瞬时资源故障，等待重试）"
            except Exception as te:
                detail = f"底层 tokenizer 真实错误 {type(te).__name__}: {str(te)[:160]}"
            print(f"[模型加载] 第 {attempt}/6 次失败：{reason}；{detail}")
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("嵌入模型连续 6 次加载失败，已终止入库") from last


def check_health(case: str) -> dict:
    """向量库健康检查：确认数据集与索引一致，且 FAISS 索引能真正加载可查。

    两个硬性检查：
      1) 记录数一致：索引 ntotal 与数据集一致（且版本标记匹配）。
      2) 索引可加载：用真实 query 强制加载 FAISS 索引文件。
    仅数 ntotal 会漏掉「索引文件在但加载即崩」的情况（虽 FAISS 比 chroma 稳得多，
    但仍以真实 query 暴露问题，避免假阳性）。
    """
    jsonl, version_file, vec_dir = _paths(case)
    if not (vec_dir / "faiss.index").exists():
        return {"ok": False, "reason": "向量索引不存在，需先 ingest"}

    try:
        n, version = validate_records(jsonl, load_version(version_file))
    except Exception as e:
        return {"ok": False, "reason": f"数据集校验失败：{type(e).__name__}: {e}"}

    marker = vec_dir / "_version.txt"
    marker_ok = marker.exists() and marker.read_text(encoding="utf-8").strip() == version

    try:
        store = FaissStore.load(vec_dir)
        cnt = store.ntotal()
        index_loadable = False
        if cnt > 0:
            q = np.random.RandomState(0).randn(store.dim).astype("float32")
            q /= np.linalg.norm(q)
            store.search(q.reshape(1, -1), 1)
            index_loadable = True
    except Exception as e:
        return {
            "ok": False,
            "dataset_records": n,
            "version_marker_consistent": marker_ok,
            "reason": f"向量索引读取/查询失败（{type(e).__name__}）：需 --force-rebuild 重建",
            "detail": str(e),
        }

    ok = (cnt == n) and marker_ok and index_loadable
    return {
        "ok": ok,
        "dataset_records": n,
        "collection_records": cnt,
        "version": version,
        "version_marker_consistent": marker_ok,
        "index_loadable": index_loadable,
        "reason": "" if ok else "记录数/版本标记/索引可加载 任一不一致，需 --force-rebuild 重建",
    }


def _self_verify(case: str) -> None:
    """入库结束后在【独立进程】里跑 verify，强制从磁盘重新打开索引并真实 query。

    为什么必须独立进程：若 ingest 与检查在同一进程，内存中的索引对象可能让查询命中
    内存态而漏掉「磁盘索引损坏」。独立进程打开的是纯磁盘状态，才是真相。
    若索引未可靠落盘，直接抛错让 ingest 失败（不静默交付坏库）。
    """
    try:
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run.py"), "verify", "--case", case],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as e:  # subprocess 自身异常（极少见）
        raise RuntimeError(f"入库后自检进程启动失败：{type(e).__name__}: {e}") from e
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or '"ok": false' in out or '"ok":false' in out:
        raise RuntimeError(
            "入库后独立进程 verify 失败：向量索引未可靠落盘，本次入库不可信，请勿依赖此库。\n"
            "修复：python run.py ingest --case %s --force-rebuild\n"
            "verify 输出：\n%s" % (case, out[-2000:])
        )


def ingest(
    case: str,
    batch_size: int = 500,        # 外层批次：一次从 jsonl 读多少条（影响内存占用，几乎不影响速度）
    encode_batch: int = 32,      # 内层 encode 批次：torch 一次算多少条（fp32 下越大越快，但更吃内存）
    checkpoint_every: int = 5,   # 每多少批落盘一次（越大越省 IO，越小越抗崩溃）
    force_rebuild: bool = False,
) -> None:
    jsonl, version_file, vec_dir = _paths(case)
    n, version = validate_records(jsonl, load_version(version_file))
    print(f"数据集 {n} 条（案例 {case}，版本 {version}）")

    marker = vec_dir / "_version.txt"

    dim = 1024  # bge-m3 输出维度
    store: FaissStore | None = None
    existing_ids: set[str] = set()

    # 快速路径：版本未变且索引记录数一致 -> 跳过
    if not force_rebuild and (vec_dir / "faiss.index").exists():
        try:
            store = FaissStore.load(vec_dir)
            if (marker.exists()
                    and marker.read_text(encoding="utf-8").strip() == version
                    and store.ntotal() == n):
                print("数据版本未变且索引记录数一致，跳过入库")
                return
            existing_ids = set(store.ids)
            print(f"索引已有 {len(existing_ids)} 条，将只向量化新增 id（增量续跑）")
        except Exception as e:
            print(f"[warn] 读取已有索引失败（{type(e).__name__}），将全量重建：{e}")
            store = None
            existing_ids = set()
    elif force_rebuild:
        if vec_dir.exists():
            shutil.rmtree(vec_dir)
        print("已强制清空旧索引，准备全量重建")

    if store is None:
        store = FaissStore(dim)

    model = load_embedding_model()
    new_total = 0
    # 用 tqdm 显示整体进度（基于待向量化条数，而非批次）。
    # 增量续跑时 total 只算「尚未入库」的条数，进度条从 0 重新走完剩余部分。
    remaining = n - len(existing_ids)
    try:
        from tqdm import tqdm

        pbar = tqdm(total=remaining, unit="条", desc="向量化", ncols=90)
        _has_pbar = True
    except Exception:
        pbar = None
        _has_pbar = False

    for idx, batch in enumerate(_batch(jsonl, batch_size), 1):
        # 安全中断：批次之间检查，已写批次通过 save 落盘不会半途而废
        if _INTERRUPTED["flag"]:
            print("\n[中断] 已安全停止。已写入的批次已落盘，下次运行会增量续跑。")
            _save_store(store, vec_dir, version, marker)
            if _has_pbar:
                pbar.close()
            return
        todo = [r for r in batch if r["id"] not in existing_ids]
        if not todo:
            continue
        ok = False
        last_err: Exception | None = None
        for attempt in range(1, 3):  # 每批最多重试 2 次（C4）
            try:
                # encode_batch 只控制单次 encode 的内存峰值；fp32 下越大越快，但更吃内存。
                emb = model.encode(
                    [r["question"] for r in todo],
                    batch_size=encode_batch, show_progress_bar=False,
                    normalize_embeddings=True,
                )
                store.add(
                    emb.astype("float32"),
                    [r["id"] for r in todo],
                    [f"{r.get('title', '')}\n{r['answer']}".strip() for r in todo],
                    [_mk_meta(r) for r in todo],
                )
                ok = True
                new_total += len(todo)
                break
            except Exception as e:
                last_err = e
                print(f"\n  [批 {idx} 第 {attempt} 次失败] {type(e).__name__}: {e}")
                time.sleep(min(2 ** attempt, 8))  # 指数退避，封顶 8s
        if not ok:
            if _has_pbar:
                pbar.close()
            raise RuntimeError(f"批 {idx} 入库失败（已重试 2 次）：{last_err}")  # 不静默跳过
        if _has_pbar:
            pbar.update(len(todo))
        # 周期性落盘：checkpoint_every 批显式 save 一次。进程若中途崩溃/被杀，已落盘进度不丢。
        if idx % checkpoint_every == 0:
            _save_store(store, vec_dir, version, marker)
            if _has_pbar:
                pbar.set_postfix_str(f"已落盘 {store.ntotal()} 条", refresh=False)

    if _has_pbar:
        pbar.close()
    _save_store(store, vec_dir, version, marker)
    # 入库后独立进程自检：确保索引真的能从磁盘加载并可查（消灭假 ok）
    _self_verify(case)
    print("完成，索引已显式落盘；集合共 %d 条（本次新增 %d）" % (store.ntotal(), new_total))
    print("入库后独立进程自检通过：索引已可靠落盘")


def _save_store(store: FaissStore, vec_dir: Path, version: str, marker: Path) -> None:
    """显式落盘（FAISS write_index + pickle + 版本标记）。这是持久化的唯一出口。"""
    store.save(vec_dir)
    vec_dir.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(version)


def build(case: str, force_rebuild: bool = False, **kwargs) -> None:
    """ingest 的薄封装：其余参数透传给 ingest（默认值只在 ingest 签名里定义一处）。"""
    ingest(case, force_rebuild=force_rebuild, **kwargs)
