# PHMGA 2.16 统计视图（更新）

## 1) 历史盘点（保留）
历史审阅记录为 `111` 项问题（高/中/低混合），该数字用于回顾，不再作为当前执行目标。

## 2) 当前收口统计（2026-02-17）

### 关键项闭环率
| 类别 | 总数 | Closed | Open | 备注 |
|---|---:|---:|---:|---|
| Critical（C9/C10/C11/C12/C13/C17/C18） | 7 | 7 | 0 | 已在代码与测试中落地 |
| 新增阻塞（Plan parser、Builder guard、backend fail-fast、parent-cycle） | 4 | 4 | 0 | 本轮收口完成 |
| 可选依赖告警（graphviz/nolds） | 2 | 0 | 2 | warning，不阻塞主链路 |

### 新增/更新回归测试
| 测试文件 | 覆盖点 | 状态 |
|---|---|---|
| `tests/test_inquirer_agent.py` | Pearson 常数向量、cosine 零向量 | Passed |
| `tests/test_dataset_preparer_agent.py` | 父链循环防护 | Passed |
| `tests/test_executor_backend_validation.py` | 非法 `train_backend` fail-fast | Passed |

## 3) 运行验证摘要
- `preflight`：`OK: True`（有 warning）
- 目标 warning：`spectral_entropy/stft` 未注册、`nolds` 缺失、`graphviz` 缺失
- 结论：主流程可运行，风险可诊断

## 4) 后续关注（非阻塞）
1. 完善可选算子注册与依赖安装脚本。
2. 持续消化 Pydantic v2 deprecation warnings。
3. 扩大 RM101 全流程回归覆盖（含真实训练开关）。
