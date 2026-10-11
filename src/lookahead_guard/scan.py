# -*- coding: utf-8 -*-
"""C 层：静态哨兵 —— 用 AST 扫未来函数的高危写法。

比正则强的地方：能看清关键字参数。`rolling(20, center=True)` 和 `rolling(20)` 只差一个
关键字，正则分不出，AST 分得出。

规则只报"写法确定有问题"的，不猜意图；每条都给出**为什么**和**怎么改**。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["Finding", "scan_source", "scan_paths", "SEVERITY_ORDER"]

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# scipy.ndimage 的 filter1d 族：默认 origin=0 => 居中窗口 => 含未来
FILTER1D = {
    "minimum_filter1d", "maximum_filter1d", "uniform_filter1d", "correlate1d",
    "median_filter", "gaussian_filter1d", "percentile_filter", "rank_filter",
    "convolve1d", "generic_filter1d", "uniform_filter", "gaussian_filter",
}




@dataclass
class Finding:
    file: str
    line: int
    col: int
    rule: str
    severity: str
    message: str
    fix: str
    snippet: str

    def as_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "col": self.col, "rule": self.rule,
                "severity": self.severity, "message": self.message, "fix": self.fix,
                "snippet": self.snippet}


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _kwarg(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _const(node) -> Optional[object]:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
            and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


MODULES = {"np", "numpy", "pd", "pandas", "scipy", "math"}


def _stat_base(call: ast.Call) -> Optional[str]:
    """取出统计量的**被统计对象名**。`x.mean()` -> "x"；`np.mean(x)` -> "x"。"""
    recv = call.func.value if isinstance(call.func, ast.Attribute) else None
    if isinstance(recv, ast.Name):
        if recv.id not in MODULES:
            return recv.id
        if call.args and isinstance(call.args[0], ast.Name):
            return call.args[0].id
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: list[str]):
        self.path = path
        self.lines = lines
        self.findings: list[Finding] = []
        self._seen: set = set()

    def _snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def _add(self, node, rule, severity, message, fix):
        key = (getattr(node, "lineno", 0), rule)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(Finding(
            file=self.path, line=getattr(node, "lineno", 0), col=getattr(node, "col_offset", 0),
            rule=rule, severity=severity, message=message, fix=fix,
            snippet=self._snippet(getattr(node, "lineno", 0))))

    # -- 规则实现 ------------------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        name = _call_name(node)

        # 1) filter1d 族：默认居中窗口
        if name in FILTER1D:
            origin = _kwarg(node, "origin")
            if origin is None:
                self._add(node, "centered-filter1d", "HIGH",
                          f"`{name}(...)` 未指定 origin —— scipy 默认 origin=0 即【居中】窗口，"
                          f"窗口覆盖 [t-k, t+k]，含未来数据。",
                          "改用尾随: `pd.Series(x).rolling(n, min_periods=1).min()`；"
                          "或 `np.lib.stride_tricks.sliding_window_view`。")
            else:
                ov = _const(origin)
                if ov == 0:
                    self._add(node, "centered-filter1d", "HIGH",
                              f"`{name}(..., origin=0)` 是【居中】窗口，含未来数据。",
                              "去掉 origin=0，换成 rolling 尾随写法。")

        # 2) rolling/ewm 显式 center=True
        if name in {"rolling", "expanding"}:
            c = _kwarg(node, "center")
            if c is not None and _const(c) is True:
                self._add(node, "centered-rolling", "HIGH",
                          "`.rolling(..., center=True)` 把窗口居中，末 k 格含未来数据。",
                          "删掉 center=True，用尾随窗口；需要居中平滑就接受 n//2 的滞后并显式注明。")

        # 3) shift 负数
        if name == "shift":
            periods = _kwarg(node, "periods")
            if periods is None and node.args:
                periods = node.args[0]
            pv = _const(periods)
            if isinstance(pv, (int, float)) and pv < 0:
                self._add(node, "negative-shift", "MEDIUM",
                          f"`.shift({pv})` 取的是**未来**第 {abs(int(pv))} 格。",
                          "⚠️ 两种情形截然不同：(a) 构造**前向收益/label** —— 正常，忽略本条；"
                          "(b) 混进**特征/信号** —— 未来函数，必须改。请人工确认方向。")

        # 4) 向后填充
        if name in {"bfill", "backfill"} or (
                name == "fillna" and _const(_kwarg(node, "method")) in {"bfill", "backfill"}):
            self._add(node, "backfill", "MEDIUM",
                      "向后填充（bfill/backfill）用**未来**的观测值填补当前缺失。",
                      "优先前向填充 ffill；若缺失是末尾区间，直接截断该区段。")

        # 5) interpolate 双向
        if name == "interpolate":
            d = _const(_kwarg(node, "limit_direction"))
            if d in {"both", "backward"}:
                self._add(node, "interpolate-forward", "MEDIUM",
                          f"`.interpolate(limit_direction='{d}')` 会向未来方向插值。",
                          "时间序列只用 limit_direction='forward'。")

        # 6) np.roll 环绕
        if name == "roll":
            self._add(node, "np-roll", "MEDIUM",
                      "`np.roll` 环绕移位：首部会补上尾部的值（未来）。",
                      "用 pandas `.shift()` 并显式处理首部 NaN。")

        # 7) 时间序列切分未禁 shuffle
        if name == "train_test_split":
            s = _kwarg(node, "shuffle")
            if s is None or _const(s) is True:
                self._add(node, "shuffled-split", "LOW",
                          "`train_test_split` 默认 shuffle=True，时间序列会**用未来训练、用过去检验**。",
                          "传 `shuffle=False`；更稳的做法是按时间切分，不要用随机切分。")

        self.generic_visit(node)

    # 8) 全样本标准化/归一化 —— 只报"确定是标准化形状"的，避免把横截面统计误伤
    def visit_BinOp(self, node: ast.BinOp):
        ops = {type(n.op) for n in ast.walk(node) if isinstance(n, ast.BinOp)}
        has_sub, has_div = ast.Sub in ops, ast.Div in ops
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]

        hit = None
        if has_sub and has_div:          # (x - x.mean()) / x.std()
            hit = self._match_stat(calls, {"mean", "std", "median"})
            rule_msg = ("用了全样本的中心/尺度统计量做标准化 —— "
                        "每一格都看到了整段数据的均值/标准差，含未来。")
        elif has_div:                    # x / x.max()
            hit = self._match_stat(calls, {"max", "min"})
            rule_msg = ("用全样本极值做归一化 —— 每一格都看到了整段数据的最大/最小值，含未来。")

        if hit is not None:
            base, meth = hit
            self._add(node, "full-sample-stat", "MEDIUM",
                      f"对 `{base}` 用了全样本统计量 `.{meth}()`：{rule_msg}",
                      "改成**滚动**统计 `x.rolling(n).mean()`；或只用 t 之前的数据拟合。"
                      "⚠️ 若这是**横截面**统计（同一时刻对多只标的取均值）则合法，请人工确认。")
        self.generic_visit(node)

    @staticmethod
    def _match_stat(calls, methods):
        for c in calls:
            if isinstance(c.func, ast.Attribute) and c.func.attr in methods:
                base = _stat_base(c)
                if base:
                    return base, c.func.attr
        return None


def scan_source(source: str, filename: str = "<string>") -> list[Finding]:
    lines = source.splitlines()
    tree = ast.parse(source, filename=filename)
    v = _Visitor(filename, lines)
    v.visit(tree)
    return sorted(v.findings, key=lambda f: (f.line, SEVERITY_ORDER.get(f.severity, 9)))


def scan_paths(paths: Iterable[str | Path], exclude_dirs=("__pycache__", ".git", ".venv", "venv",
                                                         "node_modules")) -> list[Finding]:
    out: list[Finding] = []
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else [
            f for f in p.rglob("*.py") if not any(d in f.parts for d in exclude_dirs)]
        for f in files:
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
                ast.parse(src, filename=str(f))
            except SyntaxError:
                continue
            except Exception:
                continue
            out.extend(scan_source(src, filename=str(f)))
    return sorted(out, key=lambda f: (f.file, f.line, SEVERITY_ORDER.get(f.severity, 9)))


if __name__ == "__main__":
    import sys
    fs = scan_paths(sys.argv[1:] or ["."])
    for f in fs:
        print(f"{f.severity:6} {f.file}:{f.line} [{f.rule}] {f.message}")
    print(f"\n共 {len(fs)} 条")
