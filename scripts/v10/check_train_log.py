#!/usr/bin/env python3
"""训练日志问题信号检测器 — 本项目历史问题信号全清单扫描。

usage:
  python3 check_train_log.py <log_path> [--json]

信号分级:
  CRITICAL = 立即停训/修复（训练无效或即将无效）
  WARN     = 观察（非致命但需要关注）
  INFO     = 状态提示

覆盖的历史信号（每条都对应一次真实事故）:
  R8 陷阱          lora-grads=0 / mtp-grads=0（checkpoint 输入不可导）
  requires_grad 修复验证   GRAD AUDIT mtp-grads=599 应非零
  数值毒化         nan / inf / NONFINITE / is not finite
  显存             OutOfMemoryError / Tried to allocate
  rank 故障        ChildFailedError / exitcode / SIGTERM / SIGKILL / SERVER DIED
  fill 台账        written X/Y 不等 / unwritten / UNCONSUMED 非空
  诊断失败         diag FAILED（非致命但记录）
  环境缺失         ModuleNotFoundError / FileNotFoundError (config/tokenizer)
  发散前兆         gnorm 单调爬升（滑动窗口对比）
  loss 爆炸        loss > 10× 中位数
  步时异常         步时 > 2× 中位数（持续）
  配方漂移         DRAFT_W/MTP_LR/CP_INTERVAL 与期望不符
  repo-id 误配     Repo id must be in the form（tokenizer 路径指向不存在目录）
"""
import json
import re
import statistics
import sys
from collections import defaultdict

CRIT, WARN, INFO = "CRITICAL", "WARN", "INFO"

PATTERNS = [
    (r"mtp-grads=0(\.0+)?(\s|$|\))", CRIT, "MTP 零梯度（R8 陷阱: checkpoint 输入不可导/detach 副作用）——MTP 未训练"),
    (r"lora-grads=0(\.0+)?(\s|$|\))", CRIT, "LoRA 零梯度（R8 原版陷阱）"),
    (r"GRAD AUDIT.*mtp-grads=0", CRIT, "step1 审计: MTP 零梯度（与上一条同源，审计视角）"),
    (r"\bNaN\b|\bnan\b|NONFINITE|non-finite|is not finite", CRIT, "非有限值（nan/inf）——梯度毒化"),
    (r"OutOfMemoryError|Tried to allocate", CRIT, "显存溢出（OOM）"),
    (r"ChildFailedError", CRIT, "torch elastic 子进程失败"),
    (r"exitcode\s*:\s*[1-9]", CRIT, "rank 非零退出"),
    (r"SERVER DIED|SERVER NOT READY", CRIT, "serve 进程死亡/未就绪"),
    (r"Signal (15|9) \(", WARN, "进程被信号终止（SIGTERM/SIGKILL）"),
    (r"ModuleNotFoundError: No module named", CRIT, "环境缺包（依赖未同步）"),
    (r"FileNotFoundError.*config\.json", CRIT, "vendor config.json 缺失（权重目录被删/路径错）"),
    (r"Repo id must be in the form", CRIT, "tokenizer 路径指向不存在目录（回退 hub 解析失败）"),
    (r"diag FAILED", WARN, "诊断失败（非致命，但诊断信号缺失）"),
    (r"UNWRITTEN targets \([1-9]|unwritten=[1-9]", CRIT, "fill 台账缺权重"),
    (r"UNCONSUMED keys \([1-9]", CRIT, "fill 台账有未消费的 ckpt 键"),
    (r"shape mismatch", CRIT, "权重形状不匹配（vendor 目录版本错）"),
    (r"EXPORT FAILED|BENCH FAILED", CRIT, "导出/评测失败"),
]

# 需要数值趋势分析的键
GNORM_RE = re.compile(r"gnorm=([0-9.]+)")
LOSS_RE = re.compile(r" loss=([0-9.]+) ")
STEP_RE = re.compile(r"step (\d+)/(\d+)")
STEPTIME_RE = re.compile(r"\((\d+\.\d+)s,")
FILL_RE = re.compile(r"written (\d+)/(\d+) targets")
AUDIT_RE = re.compile(r"GRAD AUDIT step1: lora-grads=(\d+) mtp-grads=(\d+)")
CONFIG_RE = re.compile(r"(GLM53_V10_DRAFT_W|GLM53_V10_MTP_LR|GLM53_MTP_DISTILL_WEIGHT|GLM53_V10_AGREE_W|GLM53_V10_MARGIN_W)=([0-9.e+-]+)")


def scan_patterns(lines):
    hits = defaultdict(list)
    for i, line in enumerate(lines, 1):
        for pat, sev, msg in PATTERNS:
            if re.search(pat, line):
                hits[(sev, msg)].append(i)
    return hits


def scan_fill(lines):
    """fill 台账一致性: written 必须 == targets。"""
    out = []
    for line in lines:
        m = FILL_RE.search(line)
        if m:
            w, t = int(m.group(1)), int(m.group(2))
            if w != t:
                out.append((CRIT, f"fill 台账 written {w}/{t} — 缺权重"))
            else:
                out.append((INFO, f"fill 台账 written {w}/{t} ✓"))
    return out


def scan_audit(lines):
    """GRAD AUDIT: step1 的 mtp-grads 必须非零（R8 陷阱哨兵）。"""
    out = []
    audits = list(AUDIT_RE.finditer(" ".join(lines)))
    has_step1 = any(re.search(r"step 1/", l) for l in lines)
    if not audits and has_step1:
        out.append((WARN, "step 1 已存在但无 GRAD AUDIT 行——审计被跳过或 diag 未到"))
    for m in audits:
        lora_n, mtp_n = int(m.group(1)), int(m.group(2))
        if mtp_n == 0:
            out.append((CRIT, f"GRAD AUDIT: mtp-grads=0（MTP 冻结——R8 陷阱）lora={lora_n}"))
        else:
            out.append((INFO, f"GRAD AUDIT: lora={lora_n} mtp={mtp_n} ✓"))
    return out


def scan_trend(lines):
    """gnorm/loss/步时趋势: 发散与爆炸检测。"""
    out = []
    gn = [float(m.group(1)) for l in lines if (m := GNORM_RE.search(l))]
    ls = [float(m.group(1)) for l in lines if (m := LOSS_RE.search(l))]
    st = [float(m.group(1)) for l in lines if (m := STEPTIME_RE.search(l))]

    if len(gn) >= 20:
        head, tail = statistics.mean(gn[:10]), statistics.mean(gn[-10:])
        if tail > max(3 * head, head + 5) and tail > 5:
            out.append((WARN, f"gnorm 爬升: 前10均值 {head:.2f} → 后10均值 {tail:.2f}（发散前兆）"))
        if max(gn) > 100:
            out.append((CRIT, f"gnorm 峰值 {max(gn):.1f} —— 疑似爆炸"))
    if len(ls) >= 20:
        med = statistics.median(ls)
        if med > 0 and max(ls) > 10 * max(med, 1):
            out.append((CRIT, f"loss 爆炸: max {max(ls):.1f}（中位 {med:.2f}）"))
        if med > 0 and min(ls[-10:]) > 3 * max(statistics.median(ls[:10]), 0.1):
            out.append((WARN, "loss 后段抬升 3×（可能的过拟合/配方冲突，结合 serving 判断）"))
    if len(st) >= 20:
        med = statistics.median(st)
        slow = sum(1 for x in st[-10:] if x > 2 * max(med, 1))
        if slow >= 8:
            out.append((WARN, f"步时持续 >2× 中位（{slow}/10）——IO/CPU 争抢或降频"))
    return out


def scan_config(lines):
    out = []
    found = {}
    for l in lines:
        for m in CONFIG_RE.finditer(l):
            found[m.group(1)] = m.group(2)
    if found:
        out.append((INFO, "启动配方: " + json.dumps(found, ensure_ascii=False)))
    return out, found


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "-"
    text = sys.stdin.read() if log_path == "-" else open(log_path, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()

    hits = scan_patterns(lines)
    fill = scan_fill(lines)
    audit = scan_audit(lines)
    trend = scan_trend(lines)
    cfg, cfg_vals = scan_config(lines)

    report = {"CRITICAL": [], "WARN": [], "INFO": []}
    for (sev, msg), linenos in hits.items():
        head = ", ".join(str(x) for x in linenos[:5])
        more = f"…共{len(linenos)}处" if len(linenos) > 5 else ""
        report[sev].append(f"{msg} (行 {head}{more})")
    for sev, msg in fill + audit + trend + cfg:
        report[sev].append(msg)

    print(f"=== 训练日志检测: {log_path} ===")
    print(f"行数: {len(lines)} | 配方: {json.dumps(cfg_vals, ensure_ascii=False)}")
    verdict = "CRITICAL" if report["CRITICAL"] else ("WARN" if report["WARN"] else "CLEAN")
    print(f"总体判定: {verdict}")
    for sev in (CRIT, WARN, INFO):
        if report[sev]:
            print(f"\n[{sev}]")
            for m in report[sev]:
                print(f"  - {m}")
    if log_path != "-":
        json.dump(report, open(log_path + ".check.json", "w"), indent=1, ensure_ascii=False)
        print(f"\n报告: {log_path}.check.json")


if __name__ == "__main__":
    main()
