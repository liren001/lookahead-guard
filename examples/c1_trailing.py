# -*- coding: utf-8 -*-
"""✅ 正确样本：与 c1_centered.py 语义相同，但窗口是尾随的 [t-19, t]。

改法只有一行：把 `minimum_filter1d(x, 20)` 换成 `pd.Series(x).rolling(20, min_periods=1).min()`。
把本文件交给 `scan` 应无 HIGH，交给 `blind` 应通过。
"""
import numpy as np
import pandas as pd


def c1(close):
    """position-in-range（尾随版本）。close 形状 (T,) 或 (T, S)。"""
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        mn = pd.Series(close).rolling(20, min_periods=1).min().to_numpy()[:, None]
        mx = pd.Series(close).rolling(20, min_periods=1).max().to_numpy()[:, None]
    else:
        mn = np.column_stack([
            pd.Series(close[:, j]).rolling(20, min_periods=1).min().to_numpy()
            for j in range(close.shape[1])])
        mx = np.column_stack([
            pd.Series(close[:, j]).rolling(20, min_periods=1).max().to_numpy()
            for j in range(close.shape[1])])
    return (close - mn) / np.maximum(mx - mn, 1e-12)
