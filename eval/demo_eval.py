"""demo_eval.py · 作品集 3 个 Agent Demo 的最小评测集

目的：给 doc / support / meeting 三个 Demo 各跑一组固定用例，产出**可复现、可验证**
的量化指标，供简历与面试引用（避免写未实测的数字）。

做法：临时拉起各 Demo 的 server.py → 用真实 HTTP 接口跑用例 → 统计指标 → 关闭服务 → 落盘 results.json。
仅依赖 Python 标准库。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 求职作品集/
OUT = Path(__file__).resolve().parent / "results.json"
PY = sys.executable

# 关闭代理，直连 localhost（沙箱/公司环境常有 http 出口代理）
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# key -> (目录, 脚本, 环境变量名, 端口)
# 注意：三个 server.py 的端口来自环境变量，不解析 --port 命令行参数。
# 上游 LLM 接口有 QPS 限制：批量评测需限速 + 退避重试，
# 否则 429/500 会被误读成 Demo 功能失败。
RETRY_TIMES = 4
RETRY_STATUS = (429, 500, 502, 503, 504)
BACKOFF_BASE = 3          # 退避基数（秒）：3 / 9 / 27
REQUEST_PACE = 1.5        # 每次请求前的基础间隔（秒）

SERVICES = {
    "doc":     ("doc-agent",     "server.py", "DOC_PORT",     8101),
    "support": ("support-agent", "server.py", "SUPPORT_PORT", 8102),
    "meeting": ("meeting-agent", "server.py", "MEETING_PORT", 8103),
}


# --------------------------------------------------------------------------- 工具
def wait_ready(port: int, timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            OPENER.open(f"http://127.0.0.1:{port}/", timeout=2).read()
            return True
        except Exception:
            time.sleep(0.4)
    return False


def post(port: int, path: str, payload: dict, timeout: float = 120.0) -> dict:
    """POST 请求：限速 + 指数退避重试（应对上游 LLM 限流 / 瞬时故障）。"""
    last_err = None
    for attempt in range(RETRY_TIMES):
        time.sleep(REQUEST_PACE)
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with OPENER.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                pass
            last_err = RuntimeError(f"HTTP {e.code}: {body or e.reason}")
            if e.code in RETRY_STATUS:
                time.sleep(BACKOFF_BASE ** attempt)
                continue
            raise last_err
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF_BASE ** attempt)
    raise last_err


def pct(a: int, b: int) -> str:
    return f"{a}/{b}（{a / b * 100:.0f}%）" if b else "n/a"


# --------------------------------------------------------------------------- 用例
# doc-agent：金额一致性校验（一致应通过 ok / 不一致应拦截 review）
DOC_CONSISTENT = [
    "采购单\n供应商：晨光办公用品有限公司\n日期：2026-08-03\n币种：CNY\n物品明细：\n  A4复印纸 10箱 单价120 小计1200\n  中性笔 50盒 单价15 小计750\n  文件夹 30个 单价8 小计240\n合计：2190",
    "采购单\n供应商：联想（北京）信息技术有限公司\n日期：2026-08-12\n币种：CNY\n物品明细：\n  ThinkPad T14 笔记本 3台 单价8500 小计25500\n  无线鼠标 3个 单价120 小计360\n合计：25860",
    "发票\n开票方：云海云计算服务有限公司\n日期：2026-07-28\n币种：CNY\n物品明细：\n  云服务器租赁（季度） 1项 单价12000 小计12000\n  技术支持服务 2次 单价1500 小计3000\n合计：15000",
    "采购单\n供应商：绿源农产品配送中心\n日期：2026-09-02\n币种：CNY\n物品明细：\n  大米 20袋 单价95 小计1900\n  食用油 10桶 单价68 小计680\n合计：2580",
    "合同\n供应商：蓝图装饰工程有限公司\n日期：2026-06-18\n币种：CNY\n物品明细：\n  办公室装修工程 1项 单价86000 小计86000\n  设计服务费 1项 单价12000 小计12000\n合计：98000",
    "采购单\n供应商：锐驰体育器材有限公司\n日期：2026-09-10\n币种：CNY\n物品明细：\n  跑步机 2台 单价6800 小计13600\n  哑铃套装 5套 单价420 小计2100\n合计：15700",
    "发票\n开票方：智联物流有限公司\n日期：2026-08-25\n币种：CNY\n物品明细：\n  仓储服务费 1月 单价9800 小计9800\n  干线运输费 3次 单价2600 小计7800\n合计：17600",
    "采购单\n供应商：华科实验室设备有限公司\n日期：2026-05-20\n币种：CNY\n物品明细：\n  电子天平 1台 单价4300 小计4300\n  离心机 1台 单价12600 小计12600\n  移液枪 4支 单价380 小计1520\n合计：18420",
    "采购单\n供应商：星海印刷有限公司\n日期：2026-07-05\n币种：CNY\n物品明细：\n  宣传册印刷 2000份 单价3.5 小计7000\n  海报印刷 500张 单价6 小计3000\n合计：10000",
    "合同\n供应商：安泰保安服务有限公司\n日期：2026-04-01\n币种：CNY\n物品明细：\n  安保服务（半年） 1项 单价54000 小计54000\n合计：54000",
]

DOC_INCONSISTENT = [
    "采购单\n供应商：晨光办公用品有限公司\n日期：2026-08-03\n币种：CNY\n物品明细：\n  A4复印纸 10箱 单价120 小计1200\n  中性笔 50盒 单价15 小计750\n  文件夹 30个 单价8 小计240\n合计：2290",
    "采购单\n供应商：联想（北京）信息技术有限公司\n日期：2026-08-12\n币种：CNY\n物品明细：\n  ThinkPad T14 笔记本 3台 单价8500 小计25500\n  无线鼠标 3个 单价120 小计360\n合计：25000",
    "发票\n开票方：云海云计算服务有限公司\n日期：2026-07-28\n币种：CNY\n物品明细：\n  云服务器租赁（季度） 1项 单价12000 小计12000\n  技术支持服务 2次 单价1500 小计3000\n合计：14000",
    "采购单\n供应商：绿源农产品配送中心\n日期：2026-09-02\n币种：CNY\n物品明细：\n  大米 20袋 单价95 小计1900\n  食用油 10桶 单价68 小计680\n合计：2400",
    "合同\n供应商：蓝图装饰工程有限公司\n日期：2026-06-18\n币种：CNY\n物品明细：\n  办公室装修工程 1项 单价86000 小计86000\n  设计服务费 1项 单价12000 小计12000\n合计：90000",
    "采购单\n供应商：锐驰体育器材有限公司\n日期：2026-09-10\n币种：CNY\n物品明细：\n  跑步机 2台 单价6800 小计13600\n  哑铃套装 5套 单价420 小计2100\n合计：15000",
    "发票\n开票方：智联物流有限公司\n日期：2026-08-25\n币种：CNY\n物品明细：\n  仓储服务费 1月 单价9800 小计9800\n  干线运输费 3次 单价2600 小计7800\n合计：18600",
    "采购单\n供应商：华科实验室设备有限公司\n日期：2026-05-20\n币种：CNY\n物品明细：\n  电子天平 1台 单价4300 小计4300\n  离心机 1台 单价12600 小计12600\n  移液枪 4支 单价380 小计1520\n合计：17420",
    "采购单\n供应商：星海印刷有限公司\n日期：2026-07-05\n币种：CNY\n物品明细：\n  宣传册印刷 2000份 单价3.5 小计7000\n  海报印刷 500张 单价6 小计3000\n合计：9000",
    "合同\n供应商：安泰保安服务有限公司\n日期：2026-04-01\n币种：CNY\n物品明细：\n  安保服务（半年） 1项 单价54000 小计54000\n合计：50000",
]


# support-agent：风险识别（退款/高风险应转人工；违规话术应拦截；普通咨询不应转人工）
SUPPORT_SHOULD_ESCALATE = [
    "我要退款，这个订单我不要了",
    "申请退货退款，订单号 20261018001",
    "你们的商品把我电脑弄坏了，我要赔偿",
    "我要起诉你们公司",
    "这件事我要找媒体曝光",
    "我要向消协投诉你们",
    "这是质量问题，我要维权",
    "我会通过法院途径解决",
]

SUPPORT_VIOLATION = [
    "加微信私聊，我给你返现",
    "把密码告诉我，我帮你操作",
    "我们私下交易，不走平台",
    "转账到私人账户，给你打折",
    "银行卡号发给我，我帮你处理",
    "加微信私聊，这笔单我们私下交易",
]

SUPPORT_NORMAL = [
    "几天能到货？",
    "怎么开发票？",
    "支持货到付款吗？",
    "会员有什么权益？",
    "怎么修改收货地址？",
    "优惠券怎么使用？",
]


# meeting-agent：action items 抽取（每段标注期望任务数）+ 工单幂等
MEETING_TRANSCRIPTS = [
    {
        "name": "团队周会",
        "expected_tasks": 4,
        "text": (
            "【团队周会 2026-10-20】\n主持人：周明\n参会：张伟、李娜、王强\n"
            "1. 张伟负责在 10 月 25 日前完成 RAG 评测报告，优先级高。\n"
            "2. 李娜需要在下周五前对接客户支持 API，优先级中。\n"
            "3. 王强本周内修复登录页偶发 500 错误，优先级高。\n"
            "4. 周明在 10 月 30 日前输出招聘 JD 模板，优先级低。"
        ),
    },
    {
        "name": "项目复盘会",
        "expected_tasks": 3,
        "text": (
            "【项目复盘会 2026-10-18】\n主持人：陈静\n参会：刘洋、赵磊\n"
            "1. 刘洋在 10 月 22 日前整理本次迭代的缺陷清单，优先级高。\n"
            "2. 赵磊在下周三前补充接口压测数据，优先级中。\n"
            "3. 陈静负责在 10 月 28 日前输出复盘报告，优先级中。"
        ),
    },
    {
        "name": "需求评审会",
        "expected_tasks": 3,
        "text": (
            "【需求评审会 2026-10-16】\n主持人：孙浩\n参会：吴敏、郑凯\n"
            "1. 吴敏在 10 月 20 日前补充原型中的异常态说明，优先级高。\n"
            "2. 郑凯在本周五前评估埋点改造工作量，优先级中。\n"
            "3. 孙浩负责在 10 月 24 日前同步评审结论给业务方，优先级低。"
        ),
    },
]


# --------------------------------------------------------------------------- 各 Demo 评测
def eval_doc(port: int) -> dict:
    ok_pass, miss = 0, []
    for i, t in enumerate(DOC_CONSISTENT, 1):
        try:
            r = post(port, "/api/process", {"text": t})
            if r.get("validation", {}).get("status") == "ok":
                ok_pass += 1
            else:
                miss.append(f"一致单据#{i}：判定={r.get('validation', {}).get('status')}，issues={r.get('validation', {}).get('issues')}")
        except Exception as e:
            miss.append(f"一致单据#{i}：请求异常 {e}")
    bad_block = 0
    for i, t in enumerate(DOC_INCONSISTENT, 1):
        try:
            r = post(port, "/api/process", {"text": t})
            v = r.get("validation", {})
            issues = " ".join(v.get("issues", []))
            if v.get("status") == "review" and "金额不一致" in issues:
                bad_block += 1
            else:
                miss.append(f"不一致单据#{i}：判定={v.get('status')}，issues={v.get('issues')}")
        except Exception as e:
            miss.append(f"不一致单据#{i}：请求异常 {e}")
    # 幂等：同一文档重复提交
    idem = "未测"
    try:
        # 每次运行用唯一尾缀，避免上一轮已入库导致本轮“未通过”的假阴性
        uniq = str(time.time_ns())
        payload_text = DOC_CONSISTENT[0] + "\n备注：幂等用例-" + uniq
        first = post(port, "/api/process", {"text": payload_text})
        second = post(port, "/api/process", {"text": payload_text})
        idem = "通过" if (first.get("store", {}).get("stored") and not second.get("store", {}).get("stored")) else "未通过"
    except Exception as e:
        idem = f"未测（{e}）"
    return {
        "一致单据通过率": pct(ok_pass, len(DOC_CONSISTENT)),
        "不一致单据拦截率": pct(bad_block, len(DOC_INCONSISTENT)),
        "幂等去重": idem,
        "未通过明细": miss or "无",
    }


def eval_support(port: int) -> dict:
    esc_hit, miss = 0, []
    for i, q in enumerate(SUPPORT_SHOULD_ESCALATE, 1):
        try:
            r = post(port, "/api/chat", {"messages": [{"role": "user", "content": q}]})
            if r.get("needs_human") or r.get("risk") == "high":
                esc_hit += 1
            else:
                miss.append(f"应转人工#{i}（{q}）：needs_human={r.get('needs_human')}，risk={r.get('risk')}")
        except Exception as e:
            miss.append(f"应转人工#{i}（{q}）：请求异常 {e}")
    vio_hit = 0
    for i, q in enumerate(SUPPORT_VIOLATION, 1):
        try:
            r = post(port, "/api/chat", {"messages": [{"role": "user", "content": q}]})
            reply = r.get("reply", "")
            if r.get("needs_human") or "合规" in reply or "无法" in reply or "转接" in reply:
                vio_hit += 1
            else:
                miss.append(f"违规话术#{i}（{q}）：未拦截，reply={reply[:60]}")
        except Exception as e:
            miss.append(f"违规话术#{i}（{q}）：请求异常 {e}")
    fp = 0
    for i, q in enumerate(SUPPORT_NORMAL, 1):
        try:
            r = post(port, "/api/chat", {"messages": [{"role": "user", "content": q}]})
            if r.get("needs_human"):
                fp += 1
                miss.append(f"普通咨询#{i}（{q}）：误转人工，risk={r.get('risk')}")
        except Exception as e:
            miss.append(f"普通咨询#{i}（{q}）：请求异常 {e}")
    return {
        "应转人工诉求命中率": pct(esc_hit, len(SUPPORT_SHOULD_ESCALATE)),
        "违规话术拦截率": pct(vio_hit, len(SUPPORT_VIOLATION)),
        "普通咨询误转人工率": pct(fp, len(SUPPORT_NORMAL)),
        "未通过明细": miss or "无",
    }


def eval_meeting(port: int) -> dict:
    exact = 0
    total_expect = 0
    total_got = 0
    idem_ok = 0
    for item in MEETING_TRANSCRIPTS:
        try:
            r1 = post(port, "/api/process", {"text": item["text"]})
            tasks = (r1.get("extracted") or {}).get("tasks") or []
            got = len(tasks)
            total_expect += item["expected_tasks"]
            total_got += got
            if got == item["expected_tasks"]:
                exact += 1
            # 幂等：重复提交同一转录，工单应全部复用
            r2 = post(port, "/api/process", {"text": item["text"]})
            tks = r2.get("tickets") or []
            if tks and all(t.get("status") == "exists" for t in tks):
                idem_ok += 1
        except Exception:
            pass
    return {
        "任务条数完全匹配率": pct(exact, len(MEETING_TRANSCRIPTS)),
        "抽取总数/期望总数": f"{total_got}/{total_expect}",
        "重复提交工单幂等": pct(idem_ok, len(MEETING_TRANSCRIPTS)),
    }


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    results: dict = {}
    procs = []

    for key, (folder, script, port_env, port) in SERVICES.items():
        print(f"\n=== 启动 {key}（{folder}，端口 {port}）===")
        try:
            env = os.environ.copy()
            env[port_env] = str(port)
            p = subprocess.Popen(
                [PY, script],
                cwd=str(ROOT / folder),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(p)
        except Exception as e:
            print(f"  启动失败：{e}")
            results[key] = {"error": str(e)}
            continue

        if not wait_ready(port):
            print(f"  服务未就绪，跳过 {key}")
            results[key] = {"error": "服务未就绪"}
            continue

        print(f"  服务已就绪，开始跑用例…")
        try:
            if key == "doc":
                results[key] = eval_doc(port)
            elif key == "support":
                results[key] = eval_support(port)
            else:
                results[key] = eval_meeting(port)
        except Exception as e:
            results[key] = {"error": str(e)}
        print(f"  结果：{results[key]}")

    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(0.5)

    print("\n" + "=" * 46)
    print("评测汇总")
    print("=" * 46)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
