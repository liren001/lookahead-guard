# -*- coding: utf-8 -*-
"""A 层：原语语义自检 —— "这个滚动原语到底看没看到未来？"

为什么需要：一个 `minimum_filter1d(size=20)` 的窗口是【居中】[t-10, t+9]，含未来 9 天，
不是尾随 [t-19, t]。这类错误静默、不报错、回测更好看 —— 它能废掉整批研究。

判据是可机械验证的：拿一个**已知答案的小数组**（a = 1..30），把该原语的输出与
【真·尾随窗口】逐格对拍。不等 ⇒ 它读了未来。

这条闸门的设计要求：**参照与待测必须同窗**。曾有一版把参照窗口写成 len(a)=30 而非 n=20，
于是把对的误判 FAIL、把 a 单调的 .max 假通过 —— 守卫自己变成了新的污染对象。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = [
    "tail_min", "tail_max", "tail_mean", "tail_sum",
    "PrimitiveResult", "check_primitive", "self_test_primitives",
]


# --------------------------------------------------------------------------------------
# 参照实现：真·尾随窗口 [t-n+1, t]（朴素 O(T·n)，只用于对拍，不进生产）
# --------------------------------------------------------------------------------------
def _tail_reduce(a: np.ndarray, n: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    for t in range(len(a)):
        lo = max(0, t - n + 1)
        out[t] = fn(a[lo:t + 1])
    return out


def tail_min(a, n):
    return _tail_reduce(np.asarray(a, float), n, np.nanmin)


def tail_max(a, n):
    return _tail_reduce(np.asarray(a, float), n, np.nanmax)


def tail_mean(a, n):
    return _tail_reduce(np.asarray(a, float), n, np.nanmean)


def tail_sum(a, n):
    return _tail_reduce(np.asarray(a, float), n, np.nansum)


# --------------------------------------------------------------------------------------
# 对拍
# --------------------------------------------------------------------------------------
@dataclass
class PrimitiveResult:
    name: str
    passed: bool
    first_bad_t: Optional[int] = None
    got: Optional[float] = None
    want: Optional[float] = None

    def line(self) -> str:
        if self.passed:
            return f"  PASS  {self.name}"
        return (f"  FAIL  {self.name}\n"
                f"        首个不符 t={self.first_bad_t}: 该原语给 {self.got} / 真尾随应给 {self.want}\n"
                f"        => 它用了 t 之后的数据，不许写进任何信号")


def check_primitive(name: str, fn: Callable, ref: Callable, n: int,
                    a: Optional[np.ndarray] = None) -> PrimitiveResult:
    """对拍 fn(a) 与 ref(a, n)。同窗比较，nan 视为相等。"""
    if a is None:
        a = np.arange(1, 31, dtype=float)
    a = np.asarray(a, dtype=float)
    got, want = np.asarray(fn(a), dtype=float), np.asarray(ref(a, n), dtype=float)
    if got.shape != want.shape:
        return PrimitiveResult(name, False, -1, float("nan"), float("nan"))
    for t in range(len(a)):
        g, w = got[t], want[t]
        if np.isnan(g) and np.isnan(w):
            continue
        if not np.isclose(g, w, equal_nan=True):
            return PrimitiveResult(name, False, t, float(g), float(w))
    return PrimitiveResult(name, True)


def self_test_primitives(verbose: bool = False) -> list[PrimitiveResult]:
    """对常见滚动/滤波原语逐个验尾随语义。返回结果列表（含 FAIL 项）。"""
    import pandas as pd
    from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d, correlate1d

    N = 20

    def _roll(method):
        return lambda x: np.asarray(
            getattr(pd.Series(x).rolling(N, min_periods=1), method)())

    cases = [
        # --- 最危险的一族：scipy filter1d 默认 origin=0 即【居中】，含未来 ---
        (f"scipy.minimum_filter1d(size={N})            [默认 origin=0 => 居中]",
         lambda x: minimum_filter1d(x, N, mode="nearest"), tail_min),
        (f"scipy.maximum_filter1d(size={N})            [默认 origin=0 => 居中]",
         lambda x: maximum_filter1d(x, N, mode="nearest"), tail_max),
        (f"scipy.uniform_filter1d(size={N})            [默认 origin=0 => 居中]",
         lambda x: uniform_filter1d(x, N, mode="nearest"), tail_mean),
        (f"scipy.correlate1d(ones({N})/{N})             [默认 origin=0 => 居中]",
         lambda x: correlate1d(x, np.ones(N) / N, mode="nearest"), tail_mean),
        # --- 显式居中 ---
        (f"pandas.rolling({N}, center=True).mean()      [显式居中]",
         lambda x: np.asarray(pd.Series(x).rolling(N, min_periods=1, center=True).mean()), tail_mean),
        # --- 正确写法：尾随 ---
        (f"pandas.rolling({N}, min_periods=1).min()    [尾随，正确]",
         _roll("min"), tail_min),
        (f"pandas.rolling({N}, min_periods=1).max()    [尾随，正确]",
         _roll("max"), tail_max),
        (f"pandas.rolling({N}, min_periods=1).mean()   [尾随，正确]",
         _roll("mean"), tail_mean),
    ]

    results = [check_primitive(nm, fn, ref, N) for nm, fn, ref in cases]
    if verbose:
        print("=" * 92)
        print(f"A. 原语语义自检 —— 判据: 是否等于【真·尾随 [t-n+1, t]】(a=1..30, n={N})")
        print("=" * 92)
        for r in results:
            print(r.line())
        print()
        print("  决定性小例子: a=1..30, size=20 居中 => t=19 给 min=10, 真尾随应给 min=1;")
        print("                t=29 给 min=20, 真尾随应给 min=11。")
        print("  => 尾随一律改用 pd.Series.rolling(n, min_periods=1)。")
        print()
    return results


if __name__ == "__main__":
    res = self_test_primitives(verbose=True)
    npass = sum(r.passed for r in res)
    print(f"  => A 层: {npass}/{len(res)} 通过")
