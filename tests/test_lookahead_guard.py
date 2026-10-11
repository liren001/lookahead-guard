# -*- coding: utf-8 -*-
"""自测：守卫必须能抓住已知的真身，也必须放过已知的正确写法。

跑法：`python tests/test_lookahead_guard.py`（无需 pytest），或 `pytest -q`。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from lookahead_guard import (
    blind_test, scan_source, self_test_primitives, permutation_test, render_report,
)
from lookahead_guard.scan import scan_paths


# ------------------------------------------------------------------ A 层
def test_A_centered_is_flagged():
    res = self_test_primitives(verbose=False)
    by = {r.name: r for r in res}
    centered = [n for n in by if "居中" in n]
    assert centered, "用例里必须包含居中写法"
    for n in centered:
        assert not by[n].passed, f"{n} 应判 FAIL"


def test_A_trailing_passes():
    res = self_test_primitives(verbose=False)
    by = {r.name: r for r in res}
    trailing = [n for n in by if "尾随，正确" in n]
    assert trailing, "用例里必须包含尾随写法"
    for n in trailing:
        assert by[n].passed, f"{n} 应判 PASS"


def test_A_reference_window_is_n_not_len():
    """回归：参照窗口必须等于 n，不能等于 len(a)。"""
    from lookahead_guard.primitives import check_primitive, tail_min
    a = np.arange(1, 31, dtype=float)
    # 真尾随在 n=20 时 t=19 应给 min=1
    want = tail_min(a, 20)
    assert want[19] == 1.0
    assert want[29] == 11.0
    r = check_primitive("tail_min(真尾随)", lambda x: tail_min(x, 20), tail_min, 20, a=a)
    assert r.passed


# ------------------------------------------------------------------ B 层
def test_B_catches_centered_filter():
    from scipy.ndimage import minimum_filter1d

    rng = np.random.default_rng(7)
    P = 10 + np.cumsum(rng.normal(0, .3, (60, 1)), axis=0)

    def diseased(x):
        return minimum_filter1d(x, 20, axis=0, mode="nearest")

    r = blind_test(diseased, P, t_probe=30)
    assert not r.future_blind, "居中窗口必须被判为未来函数"
    assert r.n_changed > 0


def test_B_passes_trailing():
    import pandas as pd

    rng = np.random.default_rng(7)
    P = 10 + np.cumsum(rng.normal(0, .3, (60, 1)), axis=0)

    def healthy(x):
        return pd.Series(x[:, 0]).rolling(20, min_periods=1).min().to_numpy()[:, None]

    r = blind_test(healthy, P, t_probe=30)
    assert r.future_blind, "尾随窗口必须通过"
    assert r.n_changed == 0


def test_B_demo_works():
    from lookahead_guard.blind import demo_catches_centered_filter
    assert demo_catches_centered_filter(verbose=False)


# ------------------------------------------------------------------ C 层
def test_C_flags_centered_filter1d():
    src = "from scipy.ndimage import minimum_filter1d\nx = minimum_filter1d(a, 20)\n"
    fs = scan_source(src)
    assert any(f.rule == "centered-filter1d" and f.severity == "HIGH" for f in fs)


def test_C_flags_rolling_center_true():
    fs = scan_source("y = s.rolling(20, center=True).mean()\n")
    assert any(f.rule == "centered-rolling" for f in fs)


def test_C_flags_negative_shift():
    fs = scan_source("y = s.shift(-1)\n")
    assert any(f.rule == "negative-shift" for f in fs)


def test_C_flags_bfill():
    fs = scan_source("y = s.bfill()\n")
    assert any(f.rule == "backfill" for f in fs)


def test_C_flags_full_sample_stat():
    fs = scan_source("z = (s - s.mean()) / s.std()\n")
    assert any(f.rule == "full-sample-stat" for f in fs)


def test_C_clean_code_has_no_high():
    src = ("y = s.rolling(20, min_periods=1).mean()\n"
           "z = s.shift(1)\n"
           "w = s.ffill()\n")
    fs = scan_source(src)
    assert not [f for f in fs if f.severity == "HIGH"], [f.rule for f in fs]


def test_C_scan_own_examples():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fs = scan_paths([os.path.join(root, "examples")])
    files = {os.path.basename(f.file) for f in fs if f.severity == "HIGH"}
    assert "c1_centered.py" in files, "疾病样本必须被抓到"
    assert "c1_trailing.py" not in files, "正确样本不该被判 HIGH"


# ------------------------------------------------------------------ D 层
def test_D_true_signal_significant():
    rng = np.random.default_rng(1)
    r = rng.normal(size=600)
    s = r + rng.normal(scale=.85, size=600)
    res = permutation_test(s, r, n_perm=500, seed=0)
    assert res.ic > 0.5 and res.significant


def test_D_noise_not_significant():
    rng = np.random.default_rng(2)
    r = rng.normal(size=600)
    s = rng.normal(size=600)
    res = permutation_test(s, r, n_perm=500, seed=0)
    assert not res.significant


# ------------------------------------------------------------------ 报告
def test_report_renders_verdict():
    fs = scan_source("y = s.rolling(20, center=True).mean()\n")
    md = render_report(subject="demo", criteria="无未来函数写法", findings=fs)
    assert "⛔ **不通过**" in md
    assert "centered-rolling" in md

    md2 = render_report(subject="demo", criteria="无未来函数写法", findings=[])
    assert "✅ **通过**" in md2


if __name__ == "__main__":
    g = {k: v for k, v in sorted(globals().items()) if k.startswith("test_")}
    passed = failed = 0
    for name, fn in g.items():
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
