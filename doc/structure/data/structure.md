# Data Structure

## 职责
- 定义数据来源模式，如 `fixed_ids` 和 `vibench`。
- 校验 metadata 列、约束数据选择合同、输出 canonical split。
- 为训练和预检提供稳定的数据选择描述，而不是直接传裸列表。

## 为什么需要这一层
- 数据问题和模型问题不是一回事。把模型配置塞回 `data_cfg` 会让任何训练问题都变成“数据层说了算”。
- `train/val/test` 是训练语义；`ref/test` 只是历史叫法。数据层必须负责完成这次语义迁移。

## 正式入口
- `src/config/data.py`
- `src/utils/preflight.py`
- `config/data/*`

## 输入/输出边界
- 输入：metadata dataframe、fixed ids、dataset/backend 选择。
- 输出：`DataSelectionSpec`、metadata 校验报告、canonical `train/val/test` split 描述。
- 不负责：TSPN 参数、模型结构、LLM 配置。

## 当前实现状态
- 已实现：`data.selection.{train_ids,val_ids,test_ids}` 合同、legacy `ref_ids/test_ids` 向 canonical split 的规范化、preflight 基于 canonical split 检查。
- 正在收口：历史文档、个别测试和注释仍有 `ref/test` 术语。
- `v1.0` 目标但未全落地：`data/metadata_schema.yaml`、`data/selection_presets/*`、更独立的数据子目录。

## 冗余与历史包袱
- `case1 -> data_cfg -> trainer` 里曾出现模型字段重复传播：`model_profile`、`autofit_*`、`model_config_path`。
- fixed-ids 旧主链使用 `ref_ids/test_ids`，已经导致 state、trainer、dataset 构造逻辑长期耦合。

## 按 v1.0 的下一步
- 补齐 selection presets，让固定划分、metadata 查询和预定义划分都通过统一 schema 进入。
- 继续把数据构造逻辑从 case helper 中拆离到更独立的数据层。
