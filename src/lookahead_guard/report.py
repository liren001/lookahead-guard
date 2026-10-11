# -*- coding: utf-8 -*-
"""《判据自检报告》生成器 —— 这就是交付给客户的那张纸。

一页里必须能让甲方自己核对：**我用什么命令验的 / 输出是什么 / 对照是什么 / 结论是什么**。
写不出这一页的结论，就不算结论。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .blind import BlindResult
from .permtest import PermResult
from .primitives import PrimitiveResult
from .scan import Finding

__all__ = ["render_report", "CheckRow"]

CheckRow = tuple  # (名称, 方法/命令, 结果, 证据)


def _v(ok: Optional[bool]) -> str:
    if ok is None:
        return "—"
    return "✅ 通过" if ok else "⛔ 不通过"


def render_report(subject: str,
                  criteria: str,
                  primitive_results: Optional[list[PrimitiveResult]] = None,
                  blind: Optional[BlindResult] = None,
                  findings: Optional[list[Finding]] = None,
                  perm: Optional[PermResult] = None,
                  extra_checks: Optional[list[CheckRow]] = None,
                  generated_at: Optional[str] = None,
                  command: str = "") -> str:
    """生成 markdown 报告。所有传入的检查都会逐条列证据；缺哪项就标"—"。"""
    ts = generated_at or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    rows: list[CheckRow] = []

    if primitive_results is not None:
        npass = sum(r.passed for r in primitive_results)
        failed = [r for r in primitive_results if not r.passed]
        ev = f"{npass}/{len(primitive_results)} 个原语通过尾随语义对拍"
        if failed:
            f0 = failed[0]
            ev += f"；首个不符: {f0.name} (t={f0.first_bad_t}, 得 {f0.got} 应为 {f0.want})"
        rows.append(("A 原语语义自检", "对 a=1..30 与真·尾随窗口逐格对拍",
                     _v(not failed), ev))

    if blind is not None:
        ev = (f"t_probe={blind.t_probe}；改变 {blind.n_changed} 格"
              + (f"；最晚受污染 t={max(blind.bad_t)}" if blind.bad_t else ""))
        rows.append(("B 面板级未来盲检测", "扰动 t 之后的数据，比较 t 及以前的输出",
                     _v(blind.future_blind), ev))

    if findings is not None:
        hi = [f for f in findings if f.severity == "HIGH"]
        med = [f for f in findings if f.severity == "MEDIUM"]
        lo = [f for f in findings if f.severity == "LOW"]
        ev = f"HIGH {len(hi)} / MEDIUM {len(med)} / LOW {len(lo)}"
        if hi:
            ev += f"；首条: {hi[0].file}:{hi[0].line} {hi[0].rule}"
        rows.append(("C 静态哨兵", "AST 扫描高危写法",
                     _v(not hi), ev))

    if perm is not None:
        rows.append(("D 置换检验", "循环移位零分布，双尾 p",
                     _v(perm.significant),
                     f"IC={perm.ic:+.4f}, p={perm.p_value:.4f}, n={perm.n}"))

    for r in (extra_checks or []):
        rows.append(r)

    verdict_ok = all(
        r[2] == "✅ 通过" for r in rows) and bool(rows)
    if not rows:
        verdict_ok = False

    L: list[str] = []
    L.append("# 判据自检报告")
    L.append("")
    L.append(f"- **对象**：{subject}")
    L.append(f"- **生成时间**：{ts}")
    if command:
        L.append(f"- **复现命令**：`{command}`")
    L.append(f"- **判据**：{criteria}")
    L.append(f"- **结论**：{'✅ **通过**' if verdict_ok else '⛔ **不通过**'}")
    L.append("")
    L.append("## 检查清单")
    L.append("")
    L.append("| # | 检查项 | 方法 | 结果 | 证据 |")
    L.append("|---|---|---|---|---|")
    for i, (nm, how, res, ev) in enumerate(rows, 1):
        L.append(f"| {i} | {nm} | {how} | {res} | {ev} |")
    L.append("")

    if blind is not None and not blind.future_blind:
        L.append("## ⛔ 阻断项详情（B 层）")
        L.append("")
        L.append(f"- 在 t={blind.t_probe} 处改写其后的数据后，**t 及以前的输出有 {blind.n_changed} 格发生变化**。")
        L.append(f"- 受污染的 t：{blind.bad_t[:20]}{' ...' if len(blind.bad_t) > 20 else ''}")
        L.append("- 含义：该信号函数在 t 处用到了 t 之后的信息 —— **任何基于它的回测结论都不成立**。")
        L.append("")

    if findings:
        hi = [f for f in findings if f.severity == "HIGH"]
        if hi:
            L.append("## ⛔ 阻断项详情（C 层，HIGH）")
            L.append("")
            for f in hi:
                L.append(f"- `{f.file}:{f.line}` **{f.rule}** — {f.message}")
                L.append(f"  - 原文：`{f.snippet}`")
                L.append(f"  - 改法：{f.fix}")
            L.append("")

    L.append("## 口径声明")
    L.append("")
    L.append("- 本报告只回答**可机械验证**的问题（原语是否尾随 / 是否读未来 / 是否显著）。")
    L.append("- **不预测收益**，也不对策略的盈利能力作任何判断。")
    L.append("- A/B/C/D 四层的判据均为可复算的确定性检查；证据列给出的是实际数值，不是描述。")
    L.append("")
    return "\n".join(L)


def render_scan_only(subject: str, findings: list[Finding], command: str = "") -> str:
    return render_report(subject=subject,
                         criteria="静态写法不出现已知的未来函数模式",
                         findings=findings, command=command)
