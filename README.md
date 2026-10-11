# lookahead-guard

**前视偏差体检。只回答能机械验证的问题，不预测收益。**

一个静默的未来函数可以让整批研究作废，而且它通常**让回测更好看**——所以你不但不会怀疑它，
还会把它当成发现。这个包把"这不是未来函数吧？"从一句祈祷变成一道**可复算的闸门**。

```
pip install lookahead-guard            # 只用 A/C/D 层
pip install "lookahead-guard[full]"    # 加上 B 层需要的 pandas/scipy
```

---

## 它抓的是什么

```python
from scipy.ndimage import minimum_filter1d

def position_in_range(close, n=20):
    mn = minimum_filter1d(close, n, axis=0, mode="nearest")   # ⛔ 没写 origin
    mx = maximum_filter1d(close, n, axis=0, mode="nearest")   # ⛔ 没写 origin
    return (close - mn) / (mx - mn)
```

看起来完全正常。但 scipy 的 `filter1d` 族**默认 `origin=0` 就是居中窗口** `[t-10, t+9]`——
它含未来 9 个交易日。它不报错、不告警，只让回测更漂亮。**这一处曾一次性废掉 14 轮研究。**

`lookahead-guard` 会在三个独立层面把它拦下来：

```console
$ lookahead-guard scan .
HIGH   position.py:4  [centered-filter1d] `minimum_filter1d(...)` 未指定 origin ——
      scipy 默认 origin=0 即【居中】窗口，窗口覆盖 [t-k, t+k]，含未来数据。
      改法：改用尾随 pd.Series(x).rolling(n, min_periods=1).min()
```

```console
$ lookahead-guard selfcheck        # A 层：直接对拍，a=1..30
  FAIL  scipy.minimum_filter1d(size=20)   [默认 origin=0 => 居中]
        首个不符 t=11: 该原语给 2.0 / 真尾随应给 1.0
  PASS  pandas.rolling(20, min_periods=1).min()   [尾随，正确]
```

```console
$ lookahead-guard blind --module position.py:position_in_range --csv prices.csv --col close
  FAIL  未来函数: 35 格改变，最晚受污染的 t = 30
```

---

## 四层判据

| 层 | 名字 | 判据 | 依赖 |
|---|---|---|---|
| **A** | 原语语义自检 | 该滚动/滤波原语是否等于**真·尾随** `[t-n+1, t]`？拿 `a=1..30` 与参照实现逐格对拍 | scipy, pandas |
| **B** | 未来盲检测 | 扰动 `t` 之后的数据后，`t` 及以前的输出**变了没有**？ | 无（决定性） |
| **C** | 静态哨兵 | AST 扫高危写法：居中窗口 / 负 `shift` / 全样本标准化 / `bfill` / 随机切分 … | 无 |
| **D** | 置换检验 | 打乱时间对齐后，IC 还剩多少？ | 无 |

⭐ **B 层是其中唯一的"决定性"判据**：它不依赖收益、不依赖任何"证伪项"。
有过一次教训：8 项证伪检查**全部通过**，却因为**被测对象本身携带未来函数**而统统无效——
证伪闸门查不出写进被测物里的缺陷。B 层直接扰动被测物，所以查得出。

⭐ **D 层的必要性**：前视偏差不是唯一的自欺方式，另一条是**在同一段历史上反复挑**。
D 层用**循环移位**（而不是完全打乱）构造零分布——完全打乱会破坏信号自相关，
零分布过窄、p 值偏乐观。

---

## CLI

```console
lookahead-guard selfcheck                                  # A 层
lookahead-guard scan PATH [PATH...] [--json out.json] [--report out.md]
lookahead-guard blind --module m.py:fn --csv d.csv --col close [--t-probe 30]
lookahead-guard permtest --csv d.csv --signal s --fwd r [--n 2000]
lookahead-guard demo                                       # 四层一次性自检
```

`--report` 生成《判据自检报告》markdown——样例见 `docs/`。

## Python API

```python
from lookahead_guard import (self_test_primitives, blind_test,
                             scan_paths, permutation_test, render_report)

blind_test(my_signal_fn, price_panel, t_probe=30, verbose=True)
findings = scan_paths(["src/"])
permutation_test(signal, forward_return, n_perm=2000)
```

---

## 诚实边界（请先读）

- **本工具不预测收益，也不评价策略好坏。** 它只回答"这段代码/这个信号有没有偷看未来"。
- **C 层是启发式，不是证明。** 它只报"写法确定有问题的"，不报的话**不等于**没有未来函数——
  过 C 层只是必要条件。真正的判据是 B 层。
- **C 层的 `full-sample-stat` 会误报横截面统计**：同一时刻对多只标的取均值是合法的。
  该条一律标 MEDIUM 并提示人工确认。
- **`negative-shift` 通常是无辜的**：构造前向收益 label 时 `shift(-1)` 是标准写法。
  只有它混进特征/信号时才是缺陷。
- **A 层的对拍要求参照与待测同窗**。曾有一版把参照窗口写成 `len(a)` 而非 `n`，
  于是把对的判 FAIL、把单调数组的 `.max` 假通过——**守卫自己变成了新的污染对象**。
  `tests/test_A_reference_window_is_n_not_len` 是这个 bug 的回归测试。

## 测试

```console
python tests/test_lookahead_guard.py     # 16 项，无需 pytest
```

测试的核心要求不是"函数能跑"，而是：**守卫必须能抓住已知的真身，也必须放过已知的正确写法。**

## 许可

MIT
