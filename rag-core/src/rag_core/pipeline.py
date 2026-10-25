"""pipeline.py - 领域无关的数据接入层。

数据契约：
    question / answer 为必填；
    department / title 为可选（无科室数据也可入库，仅不参与按科室评估）。

职责（Stage1 -> Stage2）：
    1. 兼容多编码读取 CSV，标准化列名
    2. 清洗 / 去重 / 文本规范化 / 长度过滤 / 隐私脱敏
    3. 可选科室归一（dept_mapping.json 存在时；未知科室进 dept_pending.json 待审）
    4. 生成稳定 record_id 与数据版本指纹（供向量库联动）
    5. 输出 RAG 就绪 JSONL + 校验

设计原则：外部依赖（pandas/pyarrow）缺失即报错，不静默；所有阈值来自数据分布，不拍脑袋。
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


class DataPipeline:
    def __init__(self, config_path: str = "data_sources.json",
                 base_dir: Optional[str] = None):
        # base_dir 缺省时取配置文件所在目录（支持 cases/<name>/ 结构）
        self.base_dir = Path(base_dir) if base_dir else Path(config_path).resolve().parent
        self.config = self._load_config(self.base_dir / config_path)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.interim_dir = self.base_dir / "data" / "interim"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.cache_file = self.interim_dir / ".pipeline_cache.json"
        self._ensure_dirs()

    # ========== 初始化 ==========
    @staticmethod
    def _load_config(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ========== 变更检测 ==========
    @staticmethod
    def _file_hash(p: Path) -> str:
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"sources": {}, "last_run": None}

    def _save_cache(self, cache: dict) -> None:
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def _source_sig(self, path: Path, source: dict) -> str:
        """数据源指纹：文件内容哈希 + 分块 limit（limit 变化也视为变更，触发重处理）"""
        sig = self._file_hash(path)
        # limit 无论 None/数字/空串都参与指纹，避免「有值↔无值」切换被误判为未变更
        sig += f"|limit={source.get('limit')}"
        return sig

    def _is_dirty(self, name: str, path: Path, source: dict) -> bool:
        cache = self._get_cache()
        entry = cache.get("sources", {}).get(name)
        if not entry:
            return True
        return self._source_sig(path, source) != entry.get("hash", "")

    # ========== 科室归一（可选，两层级独立）=========
    # dept_mapping.json   : 小科室别名 -> 标准小科室（仅归一标签，不跨级）
    # dept_taxonomy.json  : 标准小科室 -> 标准大科室（仅用于交叉标记/分析，不改写归档）
    DEPT_UNKNOWN = "未分类"

    def _load_dept_mapping(self):
        mp = self.base_dir / "data" / "dept_mapping.json"
        tx = self.base_dir / "data" / "dept_taxonomy.json"
        pd_ = self.base_dir / "data" / "dept_pending.json"
        mapping = json.load(open(mp, encoding="utf-8")) if mp.exists() else {}
        taxonomy = json.load(open(tx, encoding="utf-8")) if tx.exists() else {}
        pending = json.load(open(pd_, encoding="utf-8")) if pd_.exists() else {}
        return mapping, taxonomy, pending, pd_

    def _normalize_department(self, df, rel_path, mapping, taxonomy, pending, pending_path):
        """两层级科室处理：绝不强制「小科室->大科室」一致。

        - department（小科室）：别名映射归一为标准小科室；未知进待审
        - major_department（大科室）：保留原始归档（已由 stage1 注入为列）；空缺用 taxonomy 兜底
        - cross_dept：标准小科室所属大科室 ≠ 原始归档大科室 -> 标 true（真实跨科问诊，保留不纠正）
        """
        if "major_department" not in df.columns:
            df["major_department"] = self.DEPT_UNKNOWN

        if "department" not in df.columns:
            df["department"] = self.DEPT_UNKNOWN
            df["cross_dept"] = False
            return df

        raw = df["department"].astype(str).str.strip()
        raw = raw.replace({"": self.DEPT_UNKNOWN, "nan": self.DEPT_UNKNOWN, "None": self.DEPT_UNKNOWN})
        known = set(mapping.keys())
        unknown = set(raw) - known - {self.DEPT_UNKNOWN}
        if unknown:
            for dept in unknown:
                cnt = int((raw == dept).sum())
                entry = pending.setdefault(dept, {"total": 0, "files": {}})
                entry["files"][rel_path] = cnt
                entry["total"] = sum(entry["files"].values())
            json.dump(pending, open(pending_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"       [待审科室] {rel_path}: {sorted(unknown)} -> 暂归'{self.DEPT_UNKNOWN}'")

        std = raw.apply(lambda d: mapping.get(d, d))  # 标准小科室
        df["department"] = std

        # 大科室：保留原始归档（列已注入）；空缺用 taxonomy 兜底
        maj = df["major_department"].astype(str).str.strip()
        maj = maj.replace({"nan": "", "None": ""})
        no_maj = maj == ""
        if no_maj.any():
            maj.loc[no_maj] = std.loc[no_maj].apply(lambda d: taxonomy.get(d, self.DEPT_UNKNOWN))
        df["major_department"] = maj

        # 交叉科室：标准小科室所属大科室 ≠ 原始归档大科室 -> 真实跨科问诊，保留并打标
        canon = std.apply(lambda d: taxonomy.get(d, ""))
        df["cross_dept"] = [bool(c and m and m != c) for c, m in zip(canon, maj)]
        return df

    # ========== Stage 1 ==========
    def stage1_raw_to_interim(self) -> Optional[pd.DataFrame]:
        print("\n" + "=" * 60)
        print(" [Stage 1] raw -> interim（清洗标准化）")
        print("=" * 60)

        all_dfs: List[pd.DataFrame] = []
        cache = self._get_cache()
        cache.setdefault("sources", {})
        sources = self.config.get("sources", [])
        print(f"\n共配置 {len(sources)} 个数据源\n")

        for src in sources:
            name = src["name"]
            rel_path = src["path"]
            file_path = self.raw_dir / rel_path
            if not file_path.exists():
                print(f"  [!] 文件不存在，跳过: {rel_path}")
                continue
            if not self._is_dirty(name, file_path, src):
                cached = cache["sources"].get(name, {})
                cp = cached.get("interim_path", "")
                if cp and Path(cp).exists():
                    all_dfs.append(pd.read_parquet(cp))
                    print(f"  [跳过] {name}（未变更，使用缓存）")
                    continue
                print(f"  [跳过] {name}（未变更，但缓存缺失）")
                continue

            print(f"  [处理] {name} ({rel_path})")
            nrows = src.get("limit")
            if nrows is None or nrows == "":
                nrows = None  # None = 读全量，不用猜文件条数
            encoding = src.get("encoding", "utf-8")
            try:
                df = pd.read_csv(file_path, encoding=encoding, nrows=nrows)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="gb18030", nrows=nrows)
                print("       自动切换编码: gb18030")

            df = self._standardize_columns(df)
            # 注入归档大科室：优先 CSV 自带 major_department 列；否则用数据源配置的 department（大科室）
            if "major_department" not in df.columns:
                df["major_department"] = str(src.get("department", ""))

            # 可选科室归一
            if (self.base_dir / "data" / "dept_mapping.json").exists():
                m, tx, p, pp = self._load_dept_mapping()
                df = self._normalize_department(df, rel_path, m, tx, p, pp)
            else:
                df["department"] = self.DEPT_UNKNOWN
                if "major_department" not in df.columns:
                    df["major_department"] = self.DEPT_UNKNOWN
                df["cross_dept"] = False

            # 必填校验
            for col in ("ask", "answer"):
                if col not in df.columns:
                    raise KeyError(f"数据源 {name} 缺少必填列 '{col}'，已终止。")

            before = len(df)
            df = df.dropna(subset=["ask", "answer"])
            if before != len(df):
                print(f"       清洗: 删除空 ask/answer {before - len(df)} 条")

            # 文本规范化
            for col in ("ask", "answer", "title"):
                if col in df.columns:
                    df[col] = (df[col].astype(str)
                               .str.replace(r"[\r\n\t]+", " ", regex=True)
                               .str.replace(r"\s+", " ", regex=True).str.strip())

            # 去重
            b = len(df)
            df = df.drop_duplicates(subset=["ask", "answer"]).reset_index(drop=True)
            df = df.drop_duplicates(subset=["ask"]).reset_index(drop=True)
            if b != len(df):
                print(f"       去重: 删除 {b - len(df)} 条")

            # 长度 / 质量过滤（依据数据分布定阈值，非拍脑袋）
            b = len(df)
            df = df[df["ask"].str.len().between(5, 200)]
            df = df[df["answer"].str.len().between(20, 5000)]
            df = df[df["answer"].str.contains(r"[一-龥]", regex=True)]
            df = df.reset_index(drop=True)
            if b != len(df):
                print(f"       过滤: 剔除低质量 {b - len(df)} 条")

            # 隐私脱敏（仅号码类，正文不删）
            phone, idc = r"1[3-9]\d{9}", r"\d{17}[\dXx]|\d{15}"
            for col in ("ask", "answer"):
                df[col] = df[col].str.replace(phone, "[手机号]", regex=True)
                df[col] = df[col].str.replace(idc, "[证件号]", regex=True)

            # 稳定 record_id
            df["record_id"] = [
                hashlib.sha256(json.dumps([name, str(a), str(an)],
                                          ensure_ascii=False,
                                          separators=(",", ":")).encode("utf-8")).hexdigest()[:32]
                for a, an in zip(df["ask"], df["answer"])
            ]
            df["source_file"] = rel_path

            interim_path = self.interim_dir / f"{name}_cleaned.parquet"
            df.to_parquet(interim_path, index=False)
            all_dfs.append(df)
            cache["sources"][name] = {
                "hash": self._source_sig(file_path, src),
                "interim_path": str(interim_path),
                "record_count": len(df),
                "processed_at": datetime.now().isoformat(),
            }
            print(f"       -> {len(df)} 条已保存")

        if not all_dfs:
            print("  [警告] 没有可处理的数据源")
            return None
        combined = pd.concat(all_dfs, ignore_index=True)
        out = self.interim_dir / "combined_cleaned.parquet"
        combined.to_parquet(out, index=False)
        cache["last_run"] = {
            "output": str(out), "record_count": len(combined),
            "run_at": datetime.now().isoformat(), "stages": ["raw_to_interim"],
        }
        self._save_cache(cache)
        print(f"\n  [完成] 合并 {len(all_dfs)} 源，共 {len(combined)} 条 -> {out}")
        return combined

    # ========== Stage 2 ==========
    def stage2_interim_to_processed(self, df: Optional[pd.DataFrame] = None) -> Optional[Path]:
        print("\n" + "=" * 60)
        print(" [Stage 2] interim -> processed（RAG 就绪 JSONL）")
        print("=" * 60)
        if df is None:
            p = self.interim_dir / "combined_cleaned.parquet"
            if not p.exists():
                print("  [!] 中间数据不存在，请先 stage1")
                return None
            df = pd.read_parquet(p)
        print(f"\n  读取 {len(df)} 条")

        for c in ("record_id", "ask", "answer"):
            if c not in df.columns:
                raise KeyError(f"Stage2 前置校验失败，缺少列: {c}")

        for c in ("department", "major_department", "source_file", "title", "cross_dept"):
            if c in df.columns:
                df[c] = df[c].fillna("")

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row.get("record_id", "")),
                "question": str(row.get("ask", "")),
                "department": str(row.get("department", self.DEPT_UNKNOWN)),
                "major_department": str(row.get("major_department", self.DEPT_UNKNOWN)),
                "cross_dept": bool(row.get("cross_dept", False)),
                "source_file": str(row.get("source_file", "")),
                "title": str(row.get("title", "") or ""),
                "answer": str(row.get("answer", "")),
            })

        # 数据版本指纹（流式 hash，与记录顺序无关）
        digest = hashlib.sha256()
        for rec in sorted(records, key=lambda r: r["id"]):
            payload = json.dumps(rec, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        data_version = digest.hexdigest()[:16]
        for rec in records:
            rec["data_version"] = data_version

        version_info = {"version": data_version, "record_count": len(records),
                        "built_at": datetime.now().isoformat()}
        with open(self.processed_dir / "data_version.json", "w", encoding="utf-8") as f:
            json.dump(version_info, f, ensure_ascii=False, indent=2)
        print(f"  [版本] {data_version} ({len(records)} 条)")

        out = self.processed_dir / "rag_dataset.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [完成] {out}")

        cache = self._get_cache()
        cache["last_run"] = {
            "output": str(out), "record_count": len(records),
            "run_at": datetime.now().isoformat(),
            "stages": ["raw_to_interim", "interim_to_processed"],
            "data_version": data_version,
        }
        self._save_cache(cache)
        return out

    # ========== 辅助 ==========
    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        cmap = {}
        for col in df.columns:
            cl = str(col).strip().lower()
            if cl in ("major_department", "major", "大科室"):
                cmap[col] = "major_department"
            elif "科室" in cl or cl == "department":
                cmap[col] = "department"
            elif cl == "title" or "标题" in cl:
                cmap[col] = "title"
            elif cl == "ask" or "提问" in cl or "问题" in cl:
                cmap[col] = "ask"
            elif cl == "answer" or "回答" in cl:
                cmap[col] = "answer"
        return df.rename(columns=cmap)

    def run(self) -> None:
        combined = self.stage1_raw_to_interim()
        if combined is None:
            print("\n[错误] Stage1 无数据，终止")
            return
        self.stage2_interim_to_processed(combined)
        print("\n# 流水线完成")

    # ========== 科室报告（开放集合治理）=========
    def department_report(self) -> dict:
        """汇总科室分布与配置缺口，指导往两个 json 补什么（开放科室集合用）。

        输出四类信息：
            1) 已归一的标准小科室及计数
            2) 待审科室（dept_pending.json，不在 dept_mapping.json）
            3) 跨科记录（cross_dept=true，原始大科室≠标准小科室所属大科室）
            4) taxonomy 缺口（标准小科室无 taxonomy 映射 -> 跨科检测不可靠，需补）
        """
        import collections

        mp = self.base_dir / "data" / "dept_mapping.json"
        tx = self.base_dir / "data" / "dept_taxonomy.json"
        pd_ = self.base_dir / "data" / "dept_pending.json"
        mapping = json.load(open(mp, encoding="utf-8")) if mp.exists() else {}
        taxonomy = json.load(open(tx, encoding="utf-8")) if tx.exists() else {}
        pending = json.load(open(pd_, encoding="utf-8")) if pd_.exists() else {}

        # 取记录：优先 processed jsonl，回退 interim parquet
        records = []
        proc = self.processed_dir / "rag_dataset.jsonl"
        if proc.exists():
            for line in open(proc, encoding="utf-8"):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    records.append({
                        "department": r.get("department", ""),
                        "major_department": r.get("major_department", ""),
                        "cross_dept": bool(r.get("cross_dept", False)),
                        "question": r.get("question", ""),
                    })
        else:
            ip = self.interim_dir / "combined_cleaned.parquet"
            if ip.exists():
                df = pd.read_parquet(ip)
                for _, row in df.iterrows():
                    records.append({
                        "department": str(row.get("department", "")),
                        "major_department": str(row.get("major_department", "")),
                        "cross_dept": bool(row.get("cross_dept", False)),
                        "question": str(row.get("ask", "")),
                    })

        small_counter = collections.Counter()
        cross = []
        tax_gap = set()
        for r in records:
            dep = str(r.get("department", ""))
            maj = str(r.get("major_department", ""))
            if dep:
                small_counter[dep] += 1
                if dep not in taxonomy:
                    tax_gap.add(dep)
            if r.get("cross_dept"):
                cross.append((maj, dep, str(r.get("question", ""))))

        report = {
            "case": self.base_dir.name,
            "record_count": len(records),
            "standard_small_depts": dict(small_counter),
            "mapped_aliases": len(mapping),
            "taxonomy_gap": sorted(tax_gap),
            "cross_dept_count": len(cross),
            "cross_dept_samples": [
                {"major": m, "department": d, "question": q} for m, d, q in cross[:20]
            ],
            "pending": pending,
        }

        # ---- 人类可读输出 ----
        print("\n" + "=" * 60)
        print(f" 科室分布报告 (case: {self.base_dir.name}, 共 {len(records)} 条)")
        print("=" * 60)
        print(f"\n[1] 已归一标准小科室（{len(small_counter)} 个）")
        for d, c in small_counter.most_common():
            tag = " ⚠缺 taxonomy" if d in tax_gap else ""
            print(f"     {d:<10} {c}{tag}")
        print(f"\n[2] 待审科室（不在 dept_mapping.json，已自动进 dept_pending.json，照常入库）")
        if pending:
            for d, info in sorted(pending.items()):
                print(f"     {d:<10} 累计 {info.get('total', 0)} 条 -> 需在 dept_mapping.json 补别名映射")
        else:
            print("     （无）")
        print(f"\n[3] 跨科记录 cross_dept=true（真实跨科问诊，保留未纠正）：{len(cross)} 条")
        for m, d, q in cross[:10]:
            print(f"     大科室={m} / 小科室={d}  | {q[:20]}")
        print(f"\n[4] taxonomy 缺口（以下标准小科室无 taxonomy 映射，跨科检测不可靠，需补入 dept_taxonomy.json）")
        if tax_gap:
            for d in sorted(tax_gap):
                print(f"     {d} -> 标准小科室: 标准大科室")
        else:
            print("     （无）")
        print("\n[建议] 扩展科室时：别名补 dept_mapping.json；小科室→大科室补 dept_taxonomy.json。均无需改代码。")
        return report

    def reset(self) -> None:
        if self.cache_file.exists():
            self.cache_file.unlink()
            print("[已清除] 流水线缓存")
