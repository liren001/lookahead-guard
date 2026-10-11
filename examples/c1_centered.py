# -*- coding: utf-8 -*-
"""⛔ 疾病样本：这是一个真实的未来函数。

`minimum_filter1d` / `maximum_filter1d` 不传 origin 时，scipy 默认 origin=0，
窗口是【居中】的 [t-10, t+9] —— **含未来 9 天**。

它不报错、不警告，反而让回测更好看。这一处曾一次性废掉整批 14 轮研究。
把本文件交给 `lookahead-guard scan` 应报 HIGH，交给 `blind` 应判为未来函数。
"""
import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d


def c1(close):
    """position-in-range：当前价在过去 20 日区间中的位置。"""
    mn = minimum_filter1d(close, 20, axis=0, mode="nearest")   # ⛔ 居中，含未来
    mx = maximum_filter1d(close, 20, axis=0, mode="nearest")   # ⛔ 居中，含未来
    return (close - mn) / np.maximum(mx - mn, 1e-12)
