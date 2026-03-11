# Graph Structure

## 职责
- 注册并选择 builder/executor graph。
- 为外层工作流提供稳定的图构造入口。
- 把 workflow 拓扑和 agent 实现分开。

## 为什么需要这一层
- graph 是“谁调谁”的结构，不是 agent 逻辑本身。没有这层，`main.py`、`case1.py`、兼容 facade 都会开始拼接 workflow。
- 只有把 graph 变成 registry，可选 workflow 才能稳定测试和替换。

## 正式入口
- `src/graph/registry.py`
- `src/graph/__init__.py`
- `config/graphs/*`

## 输入/输出边界
- 输入：graph 名称和配置。
- 输出：可执行 graph builder / executor graph。
- 不负责：配置 compose、具体 agent 业务逻辑、模型训练细节。

## 当前实现状态
- 已实现：`src/graph/` 成为正式归宿，graph registry 提供选择入口。
- 正在收口：`src/phm_outer_graph.py` 仍作为最后一轮兼容 facade 存在，只能转发不能扩展逻辑。
- `v1.0` 目标但未全落地：`executor_ml`、`executor_automl`、`report_only` 等 graph config 分层。

## 冗余与历史包袱
- `src/phm_outer_graph.py` 和历史 `src/graph.py` 曾同时承载真正实现与兼容入口，造成 graph 责任边界不清。

## 按 v1.0 的下一步
- 继续把剩余 builder/executor 变体纳入 config groups。
- 删除最终不再需要的 facade，只保留 `src/graph/` 注册路径。
