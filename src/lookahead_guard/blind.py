# -*- coding: utf-8 -*-
"""B 层：面板级未来盲检测 —— 不依赖回测的决定性判据。

做法：给定信号函数 f(Panel) -> 与 Panel 同形状的输出。
把 **t 之后** 的格加随机扰动后重算；比较 **<= t** 的输出。
若变了 ⇒ f 在 t 处用到了 > t 的数据 ⇒ 未来函数。

⛔ 这是本项目里最硬的一道闸门，原因是它**不依赖收益、不依赖任何"证伪项"**。
   曾经有 8 项证伪检查全部通过，却因为**被测对象本身携带未来函数**而全部无效 ——
   证伪闸门查不出写进被测物里的缺陷。本闸门把被测物自己拿去扰动，因此能查出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = ["BlindResult", "blind_test", "demo_catches_centered_filter"]


@dataclass
class BlindResult:
    future_blind: bool
    n_changed: int
    t_probe: int
    bad_t: list = field(default_factory=list)

    def line(self) -> str:
        if self.future_blind:
            return "  PASS  未来盲: 扰动 t 之后的数据, t 及以前的输出一格未变"
        return (f"  FAIL  未来函数: {self.n_changed} 格改变, 涉及 t = {self.bad_t[:5]}"
                f"{'...' if len(self.bad_t) > 5 else ''}\n"
                f"        最晚受污染的 t = {max(self.bad_t)}")


def blind_test(fn: Callable, P: np.ndarray, t_probe: Optional[int] = None,
               seed: int = 0, jitter: float = 1e6, tol: float = 1e-9,
               verbose: bool = False) -> BlindResult:
    """判 fn 是否读了 P 中 t_probe 之后的数据。P 形状 (T,) 或 (T, S)。"""
    P = np.asarray(P, dtype=float)
    T = P.shape[0]
    if t_probe is None:
        t_probe = T - 2
    if not (0 <= t_probe < T - 1):
        raise ValueError(f"t_probe={t_probe} 必须落在 [0, T-2]，T={T}")

    rng = np.random.default_rng(seed)
    Q = P.copy()
    f0 = np.asarray(fn(P), dtype=float)
    if f0.shape[0] != T:
        raise ValueError(f"fn 输出首维 {f0.shape[0]} != 输入首维 {T}，无法逐格比较")

    Q[t_probe + 1:] += rng.normal(0, jitter, size=Q[t_probe + 1:].shape)
    f1 = np.asarray(fn(Q), dtype=float)

    a, b = f0[:t_probe + 1], f1[:t_probe + 1]
    bad = ~np.isclose(a, b, atol=tol, rtol=0, equal_nan=True)
    if bad.ndim == 1:
        bad = bad[:, None]
    nbad = int(bad.sum())
    bad_t = np.where(bad.any(axis=1))[0].tolist()

    res = BlindResult(future_blind=(nbad == 0), n_changed=nbad,
                      t_probe=t_probe, bad_t=bad_t)
    if verbose:
        print(f"  [blind_test] t_probe={t_probe}  改写未来 {T - t_probe - 1} 行")
        print(res.line())
    return res


def demo_catches_centered_filter(verbose: bool = False) -> bool:
    """自检：本守卫必须能抓住"居中 filter1d"这个真身。抓不住 ⇒ 守卫是摆设。"""
    from scipy.ndimage import minimum_filter1d, maximum_filter1d
    import pandas as pd

    T, S = 60, 1
    rng = np.random.default_rng(7)
    CL = 10 + np.cumsum(rng.normal(0, .3, (T, S)), axis=0)

    def diseased(P):                       # 居中窗口 —— 含未来 9 天
        mn = minimum_filter1d(P, 20, axis=0, mode="nearest")
        mx = maximum_filter1d(P, 20, axis=0, mode="nearest")
        return (P - mn) / np.maximum(mx - mn, 1e-12)

    def healthy(P):                        # 真尾随
        mn = pd.Series(P[:, 0]).rolling(20, min_periods=1).min().to_numpy()[:, None]
        mx = pd.Series(P[:, 0]).rolling(20, min_periods=1).max().to_numpy()[:, None]
        return (P - mn) / np.maximum(mx - mn, 1e-12)

    bad = blind_test(diseased, CL, t_probe=30)
    good = blind_test(healthy, CL, t_probe=30)
    ok = (not bad.future_blind) and good.future_blind
    if verbose:
        print("  [自检] 守卫能否抓住那个真身 (居中 filter1d)?")
        print(f"    病版(居中 filter1d): {'PASS' if bad.future_blind else 'FAIL 判为未来函数'} ({bad.n_changed} 格变)")
        print(f"    真版(rolling 尾随) : {'PASS' if good.future_blind else 'FAIL'} ({good.n_changed} 格变)")
        print(f"    => 守卫{'有效：精确区分了二者' if ok else '失效，不可用！'}")
    return ok


def blind_test_frame(fn: Callable, df, t_probe: Optional[int] = None, **kw) -> BlindResult:
    """pandas DataFrame 版包装：在 df 的每一列上做盲测（列须为数值）。"""
    import pandas as pd
    P = np.asarray(df.select_dtypes("number").to_numpy(), dtype=float)
    res = blind_test(fn=lambda X: fn(pd.DataFrame(X, columns=df.select_dtypes("number").columns)),
                     P=P, t_probe=t_probe, **kw)
    return res


def make_criterion_check(fn: Callable, t_probe: int = 30, seed: int = 0,
                         jitter: float = 1e6) -> Callable[[np.ndarray], tuple]:
    """把盲测包装成"可机械验证的判据"：返回 (bool, 证据串)。"""
    def _check(P: np.ndarray) -> tuple:
        r = blind_test(fn, P, t_probe=t_probe, seed=seed, jitter=jitter)
        ev = ("扰动 t 之后的数据后，t 及以前的输出未变" if r.future_blind
              else f"扰动后 {r.n_changed} 格改变，最晚受污染 t={max(r.bad_t)}")
        return r.future_blind, ev
    return _check
