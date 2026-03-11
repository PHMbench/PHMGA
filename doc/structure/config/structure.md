# Config Structure

## 职责
- 通过 Hydra 组合 `config/` 下的配置组。
- 在 resolver 层把历史输入规范化为运行时真源，例如 canonical `data.selection`。
- 输出可审计的 resolved config 供 runner、report 和回归测试使用。

## 为什么需要这一层
- 配置是系统的真入口。如果配置解析继续散在 `case1.py`、脚本和环境变量里，就无法知道一次运行究竟用了什么合同。
- 这一层的存在，是为了把“兼容旧输入”和“运行时真实配置”分离开。

## 正式入口
- `config/config.yaml`
- `src/config/loader.py`
- `src/config/resolver.py`
- `src/config/data.py`
- `src/config/llm.py`

## 输入/输出边界
- 输入：Hydra config groups、CLI overrides、兼容期旧字段（如 `ref_ids/test_ids`、旧 top-level `model_config_path`）。
- 输出：结构化 `ResolvedConfig` 和规范化后的 runtime payload。
- 不负责：case 执行、graph 调度、模型训练。

## 当前实现状态
- 已实现：真实 Hydra compose、OpenRouter 配置校验、`data.selection` 规范化、resolved config 落盘。
- 正在收口：少量 top-level 兼容字段仍允许从 ingress 进入，但必须在 resolver 层立即收敛。
- `v1.0` 目标但未全落地：`llm/roles`、`llm/routing`、`llm/structured_outputs`、`experiments/*`、`profiles/*` 的完整分层。

## 冗余与历史包袱
- 旧问题一：`case1.py` 曾把模型解析结果回填到 `data_cfg`，导致数据层和模型层重复持有同一事实。
- 旧问题二：`ref_ids/test_ids` 长期穿透到 runner 和 trainer，导致 split 语义混乱。
- 旧入口 `src/configuration.py` 仍保留兼容壳，但已退出主配置主链。

## 按 v1.0 的下一步
- 把更多 experiment/profile 选择迁到 Hydra group，而不是 case 文件内联字段。
- 继续删除旧 ingress 兼容字段，直到 `ref_ids/test_ids` 只在文档历史记录中存在。
