# Schemas Structure

## 职责
- 定义配置、graph 选择、state 交互和结构化输入输出的协议对象。
- 让系统边界可以被验证，而不是靠约定和 README。

## 为什么需要这一层
- 没有 schema，重构时最容易发生的事就是“字段名还在，但语义已经变了”。
- `train/val/test` 迁移、Hydra resolved config、bridge/report 工件都需要稳定的结构化边界。

## 正式入口
- `src/schemas/config_schema.py`
- 其他位于 `src/schemas/` 的协议定义

## 输入/输出边界
- 输入：待校验的 payload。
- 输出：结构化对象或明确失败。
- 不负责：业务执行、文件系统副作用、LLM 调用。

## 当前实现状态
- 已实现：config selection 等基础 schema 已建立。
- 正在收口：仍有部分历史 payload 通过宽松 dict 传播。
- `v1.0` 目标但未全落地：更多 graph/data/model/report 协议的系统化整理。

## 冗余与历史包袱
- 过去很多边界直接传裸 dict，导致 `ref_ids/test_ids`、模型字段和运行时派生字段互相污染。

## 按 v1.0 的下一步
- 继续把常用 runtime payload 从裸 dict 收敛成 schema。
- 对 graph compile report、compatibility report、split protocol 等工件补更明确的结构定义。
