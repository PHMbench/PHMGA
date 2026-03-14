# 04 Rebuild Checklist

## 最小目录骨架

```text
PHMGA/
├─ config/
├─ doc/structure/
├─ scripts/
├─ src/
│  ├─ states/ prompts/ agents/
│  ├─ data/ dag/ operators/ bridge/
│  ├─ model/ training/ evaluation/
│  ├─ llm/ config/ utils/
└─ tests/
   ├─ unit/
   └─ smoke/
```

## 各目录一句话职责

- `states/prompts/agents`: 只负责 DAG 生成前端。
- `data/dag/operators`: 只负责协议、结构表示和算子语义。
- `bridge/model/training/evaluation`: 只负责后端编译、训练、评估、报告。
- `scripts`: 只保留 `preflight` 和 `run_case` 两个正式入口。

## 固定重建顺序

1. 写完 `doc/structure/` 和 `del/`。
2. 建立 canonical data protocol。
3. 建立 DAG IR 与 `OperatorCatalog`。
4. 建立 workflow front-end 与 bridge。
5. 建立 `ml` 与 `torch` 路径执行端。
6. 建立 tests 和 README 使用说明。

## 当前仓库检查清单

- `README.md` 是否仍然是唯一通用事实源。
- `AGENTS.md`、`CLAUDE.md`、`GEMINI.md` 是否只保留工具特有约束并显式引用 `README.md`。
- `doc/structure` 是否同步反映当前实现态，而不是回退成抽象蓝图。
- `05_missing_assets_and_roadmap.md` 是否已记录 gap、decision pending 与 recommended default。
- `del/02_agent_review_findings.md` 是否已形成明确的 `P0 / P1 / P2` 修复顺序。
- `config/runs/*.yaml` 是否作为正式运行配置层存在。
- 核心源码模块是否具备模块级说明和关键边界注释。
- `torch` path 是否被准确描述为当前 NumPy fallback 实现。

## 最终验收 5 条标准

1. 新结构文档能单独解释论文版仓库。
2. 仓库中不存在旧 facade、多入口主链和 split 兼容层。
3. `RM101` 与 `Ottawa` 都能跑通三条 graph path。
4. 所有进入 bridge 的 DAG JSON 都经过 schema 校验。
5. 单命令可生成 graph-dependent artifacts 与最终报告。
