# 04 — 论文集成模板（可直接替换数字）

本文件提供可直接粘贴到论文实验章节的模板。默认先使用 `interim` 语气，只有达到阈值后再切换为最终语气。

---

## 1) 主结论段模板（中期）

> We evaluate Gemini-3 under a locked RM101 domain-generalization protocol using two training backends (direct-ML and TSPN), each with four seeds `{0, 42, 123, 999}`.  
> The win criterion is fixed as `4-seed mean test_acc > 79.75%`, corresponding to the strongest baseline (WKN) under the same split.  
> At the current stage, the best available PHMGA result is `__BEST_TEST_ACC__`, yielding a gap of `__GAP_PP__` pp against the target threshold.

---

## 2) 判定段模板（Pass / Fail 二选一）

## Pass 模板

> Under the locked protocol, `__TRACK__` achieves a 4-seed mean test accuracy of `__MEAN__%` (`std=__STD__ pp`), exceeding the baseline threshold `79.75%`.  
> Therefore, the claim “Gemini-3 + PHMGA surpasses the compared baseline on RM101 DG” is supported.

## Fail 模板

> Under the locked protocol, neither track exceeds `79.75%` in 4-seed mean test accuracy.  
> The current best is `__MEAN__%` on `__TRACK__`, remaining `__GAP_PP__` pp below the baseline.  
> Therefore, we do not claim superiority at this stage.

---

## 3) 消融机制段模板（A0/A1/A2）

> For mechanism-level interpretation, we additionally evaluate A1 (no-reflect) and A2 (no-prior) after the A0 runs complete.  
> We report `Δ_reflect = A0 - A1` and `Δ_prior = A0 - A2` on test accuracy.  
> Positive `Δ_reflect` indicates benefit from iterative reflection; positive `Δ_prior` indicates benefit from prior-informed initialization.

---

## 4) 可解释性段模板（与性能绑定）

> We report operator-level and wavefilter-level evidence from `operator_importance`, `wavefilters_params`, and `predictions`.  
> Across four seeds, we quantify top-K operator overlap and parameter variance to assess interpretability stability.  
> Even when performance does not exceed the baseline, PHMGA provides diagnostically useful evidence through explicit operator attribution and frequency-band parameterization.

---

## 5) 局限性段模板（必须保留）

> This comparison is restricted to the locked RM101 DG split and the specified four-seed protocol.  
> Runs labeled `protocol_invalid` are excluded from the main table by design.  
> Conclusions should therefore be interpreted as protocol-specific.

---

## 6) 表格模板（主表 + 解释表）

### Table 1. Gemini-3 Dual-Track Main Results (RM101 DG, 4 seeds)

| Track | Seed Set | Mean Test Acc (%) | Std (ddof=0, pp) | Pass `>79.75%` | Notes |
|---|---|---:|---:|---|---|
| Direct ML (`shallow`) | 0/42/123/999 |  |  |  |  |
| TSPN (`tspn`) | 0/42/123/999 |  |  |  |  |

### Table 2. Explainability Stability Summary (Gemini-3)

| Track | Top-K Overlap | WF Param Std | Main Error Pair | Interpretation |
|---|---:|---:|---|---|
| Direct ML (`shallow`) |  |  |  |  |
| TSPN (`tspn`) |  |  |  |  |

