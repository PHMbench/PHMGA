# Utils Structure

## 职责
- 承载真正通用、无业务归属的 helper：序列化、路径、日志、preflight、state IO 等。
- 提供低层支持，但不定义业务流程。

## 为什么需要这一层
- `utils` 最容易退化成“所有没人认领的代码都扔进来”。明确这一层的原因，是为了限制它只能放无副作用、低耦合能力。
- 如果配置 normalize、trainer glue、case 编排继续回流到 `utils`，结构重构会失效。

## 正式入口
- `src/utils/__init__.py`
- `src/utils/paths.py`
- `src/utils/serialization.py`
- `src/utils/preflight.py`

## 输入/输出边界
- 输入：通用路径、状态对象、轻量配置参数。
- 输出：文件、序列化结果、预检报告、路径决议。
- 不负责：LLM provider 选择、graph 编排、模型业务策略。

## 当前实现状态
- 已实现：`src/utils.py` 已删除，`src/utils/` 是唯一实现归宿；state IO 和 preflight 已开始服务 canonical split 合同。
- 正在收口：仍有少量兼容逻辑需要在更明确的层次接管后再移出。
- `v1.0` 目标但未全落地：更干净的 helper 分组，以及更少的跨层业务 helper。

## 冗余与历史包袱
- `src/utils.py` 与 `src/utils/__init__.py` 曾长期双实现，导致行为漂移。
- `utils` 中也曾承载过 split 语义和 case glue，说明边界需要持续看守。

## 按 v1.0 的下一步
- 继续把不再通用的 helper 外迁。
- 保持 `utils` 可测试、无网络、低副作用。
