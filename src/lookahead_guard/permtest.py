# -*- coding: utf-8 -*-
"""D 层：置换检验 —— "这个信号是真的，还是时间对齐挑出来的？"

前视偏差不是唯一的自欺方式。另一条更隐蔽：**在同一段历史上反复挑**，挑出一个看起来显著的
对齐。判据是「这个 IC 在**打乱时间对齐**之后还剩下多少」。

做法：给定信号 s 与前向收益 r，算 IC（Spearman）。然后对 s 做**循环移位**（保持自相关结构，
只打断与 r 的对齐），重复 n 次得到零分布，看真实 IC 的分位。
循环移位比完全打乱更保守：完全打乱会破坏 s 的自相关，得到的零分布过窄、p 值偏乐观。

本项目的经验值：策略 vs 机械买全同域在同一段上做过这个检验 ⇒ t∈[-1.79,+0.44] 不显著。
先做这一步，再决定要不要花时间调参。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PermResult", "spearman_ic", "circular_shift_null", "permutation_test"]


@dataclass
class PermResult:
    ic: float
    p_value: float
    n: int
    n_perm: int
    null_mean: float
    null_std: float
    significant: bool

    def line(self, alpha: float = 0.05) -> str:
        verdict = "显著" if self.significant else "不显著"
        return (f"  IC(Spearman) = {self.ic:+.4f}   p = {self.p_value:.4f}   "
                f"零分布 {self.null_mean:+.4f} ± {self.null_std:.4f}  "
                f"(n={self.n}, {self.n_perm} 次循环移位)  => {verdict} @alpha={alpha}")


def spearman_ic(s: np.ndarray, r: np.ndarray) -> float:
    """Spearman 秩相关。只取两侧都非 nan 的位置。"""
    s = np.asarray(s, float)
    r = np.asarray(r, float)
    m = np.isfinite(s) & np.isfinite(r)
    if m.sum() < 3:
        return float("nan")
    return float(_rankdata(s[m]) @ _rankdata(r[m]) / len(s[m]))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩，标准化的秩（均值 0，方差 1），使点积 == 相关系数。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # 平分秩
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    ranks -= ranks.mean()
    sd = ranks.std()
    if sd > 0:
        ranks /= sd
    return ranks


def circular_shift_null(s: np.ndarray, r: np.ndarray, n_perm: int = 2000,
                        seed: int = 0, min_shift: int = 5) -> np.ndarray:
    """循环移位零分布。min_shift 排除极小位移（几乎等于原对齐）。"""
    s = np.asarray(s, float)
    r = np.asarray(r, float)
    n = len(s)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, max(min_shift + 1, n - min_shift), size=n_perm)
    out = np.empty(n_perm, float)
    for i, k in enumerate(shifts):
        out[i] = spearman_ic(np.roll(s, int(k)), r)
    return out[np.isfinite(out)]


def permutation_test(s, r, n_perm: int = 2000, seed: int = 0, alpha: float = 0.05,
                     min_shift: int = 5) -> PermResult:
    s = np.asarray(s, float)
    r = np.asarray(r, float)
    ic = spearman_ic(s, r)
    null = circular_shift_null(s, r, n_perm=n_perm, seed=seed, min_shift=min_shift)
    if len(null) == 0 or not np.isfinite(ic):
        return PermResult(ic, float("nan"), len(s), 0, float("nan"), float("nan"), False)
    # 双尾
    p = float((np.abs(null) >= abs(ic)).mean())
    return PermResult(ic=ic, p_value=p, n=len(s), n_perm=len(null),
                      null_mean=float(null.mean()), null_std=float(null.std()),
                      significant=bool(p < alpha))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    r = rng.normal(size=500)
    s_true = r + rng.normal(scale=.9, size=500)      # 真信号
    s_noise = rng.normal(size=500)                   # 无关
    print("真信号:", permutation_test(s_true, r).line())
    print("纯噪声:", permutation_test(s_noise, r).line())
