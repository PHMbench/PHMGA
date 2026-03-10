# PHMGA 2.24 收口包：Gemini-3 双主线冲线（Direct ML + TSPN）

本目录是 `2_24` 的单一事实源，用于回答两个问题：

1. Gemini-3 智能体在 RM101 DG 协议下，是否能超过对比基线 `WKN 4-seed mean test_acc=79.75%`。  
2. 即使未超过，如何给出可直接放入论文的性能与可解释性呈现。

---

## 文档索引（执行顺序）

1. `00_gap_assessment.md`  
   先看当前差距与硬判定口径（是否达标）。
2. `01_protocol_lock.md`  
   锁定可比协议，防止“跑出来但不可比”。
3. `02_execution_playbook.md`  
   外部 Agent 可直接复制执行的 runbook（Gemini-3 双主线）。
4. `03_gemini3_explainability_spec.md`  
   解释性证据、图表与跨 seed 稳定性分析规范。
5. `04_paper_integration.md`  
   论文可直接粘贴的结果模板与结论模板。
6. `05_risk_register.md`  
   风险触发条件、分流动作与升级规则。

---

## 与 2_18 / 2_22 / 2_23 的关系

- `2_18`：通用矩阵执行入口（3 LLM / 18 组合）。
- `2_22`：失败恢复作战包（active matrix 恢复与门禁）。
- `2_23`：S3 结构错误最小代码修复分支。
- `2_24`：仅聚焦 Gemini-3 在 RM101 DG 的“是否能过 79.75%”与论文呈现闭环。

---

## 2_24 DoD（完成定义）

1. `2_24` 六份文档齐全且互相引用一致。  
2. Gemini-3 双主线（`shallow`、`tspn`）均完成 `A0_full` 的 4-seed 运行（或有完整阻断证据）。  
3. 输出明确判定：`Pass/Fail`（阈值：`4-seed mean test_acc > 79.75%`）。  
4. 输出论文可用证据：
   - 性能表（mean/std）
   - 可解释性图（Markdown + PNG）
   - 限制声明（preliminary/final 的口径边界）

