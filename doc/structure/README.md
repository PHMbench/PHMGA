# PHMGA Structure Docs

## 这套文档服务谁
- 给继续重构 PHMGA 的开发者：明确每一层为什么存在，避免逻辑重新散回 `case1.py`、`utils` 和历史兼容壳。
- 给做回归和删减的人：先看边界，再决定能不能删、能不能合并、删完要补什么测试。
- 给后续接手的人：快速判断“当前实现态”与 [`v1.0.md`](v1.0.md) 的目标态差距。

## 当前实现态 vs `v1.0` 目标态
- 当前实现态：已经完成 Hydra 主配置、OpenRouter 单入口、`case registry + graph registry + base_runner` 主链、`train/val/test` 规范化、`src/model.py` / `src/utils.py` 等部分冗余删除。
- 正在收口：`case1` 仍然偏重、`src/phm_outer_graph.py` 仍是兼容 facade、`fixed_ids` 兼容入口仍需保留在 resolver 层、部分说明文档和 README 还残留旧术语。
- `v1.0` 目标态：更细的 config groups、明确的数据层和选择预设层、更多 graph 类型、完整的 model/layer/experiment/profile 分层，以及更彻底的 facade 删除。

## 如何阅读
1. 先看 [index.md](index.md)：这是结构地图，说明各层职责、边界和相邻关系。
2. 再看各模块 `structure.md`：每个模块都按统一模板说明“职责/为什么/入口/边界/现状/冗余/下一步”。
3. 最后看 [del/00_redundancy_inventory.md](del/00_redundancy_inventory.md)：这是冗余执行清单，不是历史备注。

## 文档约定
- `已实现`：代码主链已经这样工作，测试应当覆盖。
- `正在收口`：代码正在迁移或仍有兼容层，不能再继续扩散旧模式。
- `v1.0 目标`：在 [`v1.0.md`](v1.0.md) 中定义，但当前仓库尚未完整落地，文档必须明确标注，不能写成“已经实现”。
