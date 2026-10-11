# -*- coding: utf-8 -*-
"""lookahead-guard —— 前视偏差体检。

四层，全部是可机械验证的判据：

  A 原语语义自检   —— 滚动/滤波原语是尾随还是居中？（对拍已知答案的小数组）
  B 未来盲检测     —— 扰动 t 之后的数据，t 及以前的输出变了吗？（决定性）
  C 静态哨兵       —— AST 扫出高危写法（居中窗口 / 负 shift / 全样本标准化 …）
  D 置换检验       —— 打乱时间对齐后，IC 还剩多少？（这个信号是真的吗）

设计原则：**只回答能机械验证的问题，不预测收益。**
"""
from .blind import BlindResult, blind_test, blind_test_frame, demo_catches_centered_filter
from .permtest import PermResult, permutation_test, spearman_ic
from .primitives import PrimitiveResult, check_primitive, self_test_primitives
from .report import render_report, render_scan_only
from .scan import Finding, scan_paths, scan_source

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # A
    "self_test_primitives", "check_primitive", "PrimitiveResult",
    # B
    "blind_test", "blind_test_frame", "demo_catches_centered_filter", "BlindResult",
    # C
    "scan_source", "scan_paths", "Finding",
    # D
    "permutation_test", "spearman_ic", "PermResult",
    # report
    "render_report", "render_scan_only",
]
