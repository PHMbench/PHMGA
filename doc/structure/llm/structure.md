# LLM Structure

## 职责
- 统一 OpenRouter 配置、环境变量绑定、最小 client 构造和校验。
- 为 plan/execute/reflect/report 等 agent 提供单一 LLM 入口。

## 为什么需要这一层
- 如果 provider 判断散在 `model`、`configuration.py`、脚本和 agent 内部，任何一次模型切换都会变成全仓行为漂移。
- LLM 层必须只解决“如何访问模型”，不能混入训练和 DAG 业务。

## 正式入口
- `src/config/llm.py`
- `src/llm/openrouter/*`
- `config/llm/provider/openrouter.yaml`

## 输入/输出边界
- 输入：OpenRouter 相关配置和环境变量。
- 输出：规范化后的 LLM 配置、绑定后的环境、client 工厂。
- 不负责：prompt 内容设计、graph 调度、模型训练。

## 当前实现状态
- 已实现：正式 public provider 只保留 `openrouter`，旧 provider 已退出主链。
- 正在收口：个别历史文档还残留 `glm/gemini/openai_compatible` 术语。
- `v1.0` 目标但未全落地：`llm/roles`、`llm/routing`、`llm/structured_outputs` 配置层目前仍是目标结构，不是全量现状。

## 冗余与历史包袱
- `src/configuration.py` 仍保留必要兼容，但不再是正式配置真源。
- 旧 provider 专用 env 已退出公开接口，只能作为迁移期内部兼容。

## 按 v1.0 的下一步
- 把 role/routing/structured output 明确拆成 config groups，而不是继续放在 agent 细节里。
