# Prompts Structure

## 职责
- 承载 prompt 模板、文本渲染 helper 和 prompt 级常量。
- 让 prompt 内容与 agent 控制流解耦。

## 为什么需要这一层
- prompt 是 LLM 行为的输入合同。如果模板散在 agent 逻辑里，任何一次 prompt 修改都会变成业务代码 diff，难以测试也难以复盘。
- 把 prompt 层独立出来，才能明确“这是行为设计，不是执行流程”。

## 正式入口
- `src/prompts/*`
- `src/prompts/research_prompts.py`
- `src/prompts/builder|report|shared`

## 输入/输出边界
- 输入：已规范化的上下文参数。
- 输出：纯文本 prompt 或结构化 prompt 片段。
- 不负责：环境变量读取、yaml 解析、LLM client 构造。

## 当前实现状态
- 已实现：prompt 目录已开始收口，重复 prompt 文件已删除一部分。
- 正在收口：仍有部分 prompt 内容与 agent 文件绑定较紧。
- `v1.0` 目标但未全落地：更成体系的 prompt families、render helpers 和 structured output 配套模板。

## 冗余与历史包袱
- `reflect_prompt.py` / `reflector_prompt.py` 曾并存，说明 prompt 入口一度缺少明确归宿。

## 按 v1.0 的下一步
- 继续把 prompt 渲染逻辑从 agent 中外迁。
- 把各 agent prompt 的输入字段与 schema 对齐，减少手写 JSON 提示漂移。
