# -*- coding: utf-8 -*-
"""命令行入口：`lookahead-guard <子命令>`。

  selfcheck                       A 层，原语语义自检
  scan PATH [PATH...]             C 层，AST 静态哨兵（可 --json / --report）
  blind --module M --func F       B 层，未来盲检测（--csv/--npz 供数据）
  permtest --csv X --signal S --fwd R   D 层，置换检验
  demo                            四层一次性自检（守卫能不能抓住已知的真身）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，中文报告会乱码
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_callable(target: str):
    """`path/to/mod.py:func` 或 `pkg.mod:func` 两种写法。"""
    if ":" not in target:
        raise SystemExit("必须是 `模块或文件.py:函数名` 形式，例如 examples/c1.py:c1")
    mod_part, func_name = target.rsplit(":", 1)
    p = Path(mod_part)
    if p.suffix == ".py" and p.exists():
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[p.stem] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_part)
    if not hasattr(mod, func_name):
        raise SystemExit(f"{mod_part} 里没有 {func_name}")
    return getattr(mod, func_name)


def _load_matrix(args) -> "object":
    import numpy as np
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if args.col:
            return df[args.col].to_numpy(dtype=float)[:, None]
        num = df.select_dtypes("number")
        return num.to_numpy(dtype=float)
    if args.npz:
        z = np.load(args.npz)
        key = args.col or list(z.keys())[0]
        return np.asarray(z[key], dtype=float)
    raise SystemExit("blind 需要 --csv 或 --npz 提供数据")


def cmd_selfcheck(args) -> int:
    from .primitives import self_test_primitives
    res = self_test_primitives(verbose=True)
    n = sum(r.passed for r in res)
    return 0 if n == len(res) else 1


def cmd_scan(args) -> int:
    from .scan import scan_paths
    from .report import render_scan_only
    findings = scan_paths(args.paths)
    if args.json:
        Path(args.json).write_text(
            json.dumps([f.as_dict() for f in findings], ensure_ascii=False, indent=2),
            encoding="utf-8")
    if args.report:
        cmd = "lookahead-guard scan " + " ".join(args.paths)
        Path(args.report).write_text(
            render_scan_only(subject=" ".join(args.paths), findings=findings, command=cmd),
            encoding="utf-8")
        print(f"报告已写出: {args.report}")
    if not args.json and not args.report or args.verbose:
        for f in findings:
            print(f"{f.severity:6} {f.file}:{f.line} [{f.rule}] {f.message}")
        hi = sum(f.severity == "HIGH" for f in findings)
        print(f"\n共 {len(findings)} 条（HIGH {hi}）")
    return 1 if any(f.severity == "HIGH" for f in findings) else 0


def cmd_blind(args) -> int:
    from .blind import blind_test
    fn = _load_callable(args.module)
    P = _load_matrix(args)
    r = blind_test(fn, P, t_probe=args.t_probe, seed=args.seed, verbose=True)
    return 0 if r.future_blind else 1


def cmd_permtest(args) -> int:
    import numpy as np
    import pandas as pd
    from .permtest import permutation_test
    df = pd.read_csv(args.csv)
    s = df[args.signal].to_numpy(dtype=float)
    if args.fwd:
        r = df[args.fwd].to_numpy(dtype=float)
    else:
        r = np.roll(s, -args.forward_k)
        r[-args.forward_k:] = np.nan
    res = permutation_test(s, r, n_perm=args.n, seed=args.seed, alpha=args.alpha)
    print(res.line(alpha=args.alpha))
    return 0 if res.significant else 1


def cmd_demo(args) -> int:
    from .primitives import self_test_primitives
    from .blind import demo_catches_centered_filter
    print("=" * 92)
    print("lookahead-guard 自检：四层守卫能不能抓住已知的真身？")
    print("=" * 92)
    res = self_test_primitives(verbose=True)
    okA = all(r.passed for r in res if "尾随，正确" in r.name) and \
        any(not r.passed for r in res if "居中" in r.name)
    print("=" * 92)
    print("B 层：能否区分「居中 filter1d」与「rolling 尾随」")
    print("=" * 92)
    okB = demo_catches_centered_filter(verbose=True)
    print()
    print(f"=> A 层语义判别: {'有效' if okA else '失效'}")
    print(f"=> B 层未来盲判别: {'有效' if okB else '失效'}")
    return 0 if (okA and okB) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lookahead-guard",
                                description="前视偏差体检：只做可机械验证的检查，不预测收益。")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selfcheck", help="A 层：原语语义自检").set_defaults(func=cmd_selfcheck)

    s = sub.add_parser("scan", help="C 层：AST 静态哨兵")
    s.add_argument("paths", nargs="+")
    s.add_argument("--json", help="把结果写成 JSON")
    s.add_argument("--report", help="生成《判据自检报告》markdown")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_scan)

    b = sub.add_parser("blind", help="B 层：未来盲检测")
    b.add_argument("--module", required=True, help="`文件.py:函数名`，函数签名 fn(P)->同形状")
    b.add_argument("--csv")
    b.add_argument("--npz")
    b.add_argument("--col", help="CSV/NPZ 里的列名")
    b.add_argument("--t-probe", type=int, default=None)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_blind)

    t = sub.add_parser("permtest", help="D 层：置换检验")
    t.add_argument("--csv", required=True)
    t.add_argument("--signal", required=True)
    t.add_argument("--fwd", help="前向收益列名；不给则用 signal 前移 forward_k 生成")
    t.add_argument("--forward-k", type=int, default=1)
    t.add_argument("--n", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--alpha", type=float, default=0.05)
    t.set_defaults(func=cmd_permtest)

    sub.add_parser("demo", help="四层一次性自检").set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
