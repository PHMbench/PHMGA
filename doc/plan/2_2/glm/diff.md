# PHMGA-NVTA_2025_Version 与主仓库差异对比

## 概览

`doc/plan/2_2/PHMGA-NVTA_2025_Version/` 是一个**独立运行副本**，包含针对 NVTA 2025 论文的实验脚本和结果。与主仓库相比，它是一个**精简版 + 实验包装层**。

---

## 文件结构对比

### PHMGA-NVTA_2025_Version/
```
PHMGA-NVTA_2025_Version/
├── .env                          # 本地环境配置（DeepSeek API）
├── .env.example                  # 环境变量模板
├── creat_datasets.py             # 数据集生成脚本（从pkl加载PHMState并生成节点数据集）
├── train_shallow_ml.py           # 浅层ML训练脚本（加载npz数据集并训练）
├── Generate_report_final.py      # 报告生成脚本（合并state+ml结果生成final_report.md）
├── final_report.md               # 实验结果报告（已生成的轴承故障诊断报告）
└── src/
    ├── __init__.py
    ├── configuration.py          # 独立的配置类（使用 deepseek-chat）
    └── model/
        └── __init__.py
```

### 主仓库 / (对应文件)
```
/home/user/LQ/B_Signal/PHMGA/
├── src/
│   ├── agents/
│   │   ├── dataset_preparer_agent.py    # 被 creat_datasets.py 调用
│   │   ├── shallow_ml_agent.py          # 被 train_shallow_ml.py 调用
│   │   └── report_agent.py              # 被Generate_report_final.py 调用
│   ├── states/
│   │   └── phm_states.py                # PHMState 定义
│   └── configuration.py (不存在，使用 src.configuration.py 模块)
└── config/
    ├── case_exp2.yaml
    └── case_exp_ottawa.yaml
```

---

## 关键差异

### 1. Configuration 配置类

| 属性 | PHMGA-NVTA_2025_Version | 主仓库 |
|------|-------------------------|--------|
| 位置 | `src/configuration.py` | 不存在（使用 `src.configuration`） |
| 默认 LLM | `deepseek-chat` | `gemini-2.5-pro` 或其他 |
| 结构 | Pydantic BaseModel | 可能有不同的配置方式 |

```python
# PHMGA-NVTA_2025_Version/src/configuration.py
class Configuration(BaseModel):
    phm_model: str = Field(default="deepseek-chat", ...)
    query_generator_model: str = Field(default="deepseek-chat", ...)
    reflection_model: str = Field(default="deepseek-chat", ...)
    answer_model: str = Field(default="deepseek-chat", ...)
```

### 2. 导入路径差异

```python
# PHMGA-NVTA_2025_Version/ 的脚本
from src.agents.shallow_ml_agent import shallow_ml_agent
from src.agents.dataset_preparer_agent import dataset_preparer_agent
from src.agents.report_agent import report_agent_node
from src.states.phm_states import PHMState, InputData
```

主仓库中这些文件位于 `/home/user/LQ/B_Signal/PHMGA/src/agents/`，**导入路径相同**，说明这个文件夹可能是作为独立项目运行的。

### 3. creat_datasets.py vs 主仓库

| 功能 | creat_datasets.py | 主仓库 |
|------|-------------------|--------|
| 数据源 | 硬编码路径 `D:/save/case_exp_ottawa/exp2.5built_state_ottawa.pkl` | 通过 `config/case*.yaml` 配置 |
| 输出 | `generated_datasets/*.npz` | 直接在 DAG 中使用，不保存 npz |
| 状态恢复 | 直接 pickle.load PHMState | 使用 `load_state()` 函数 |

### 4. train_shallow_ml.py vs 主仓库

| 功能 | train_shallow_ml.py | 主仓库 shallow_ml_agent.py |
|------|-------------------|---------------------------|
| 数据源 | 硬编码 `C:\Users\Admin\Desktop\副本3\...\generated_datasets` | 通过 datasets 参数传入 |
| 输出 | `shallow_ml_results.md`, `ml_results.pkl` | 返回字典，由调用者保存 |
| 报告生成 | 内置 markdown 保存 | 返回 metrics_markdown 字符串 |

**核心算法相同**：两者都调用同一个 `shallow_ml_agent()` 函数。

### 5. Generate_report_final.py vs 主仓库

| 功能 | Generate_report_final.py | 主仓库 report_agent.py |
|------|--------------------------|------------------------|
| 状态来源 | `D:/save/case_exp_ottawa/exp2.5built_state_ottawa.pkl` | 通过 PHMState 传入 |
| ML结果来源 | `D:/save/case_exp_ottawa/ml_results.pkl` | `state.ml_results` 属性 |
| 输出 | `final_report.md` | 返回 `final_report` 字符串 |

---

## 路径硬编码问题

PHMGA-NVTA_2025_Version 中的脚本包含**硬编码的 Windows 路径**：

| 文件 | 硬编码路径 |
|------|-----------|
| `creat_datasets.py` | `D:/save/case_exp_ottawa/exp2.5built_state_ottawa.pkl` |
| `train_shallow_ml.py` | `C:\Users\Admin\Desktop\副本3\PHMGA-NVTA_2025_Version\generated_datasets` |
| `Generate_report_final.py` | `D:/save/case_exp_ottawa/...` |

**影响**：这些脚本无法在其他环境直接运行，需要修改路径。

---

## 依赖关系图

```
PHMGA-NVTA_2025_Version 工作流：

1. creat_datasets.py
   └──> 加载 exp2.5built_state_ottawa.pkl (PHMState)
       └──> dataset_preparer_agent(state)
           └──> 输出: generated_datasets/*.npz

2. train_shallow_ml.py
   └──> 加载 generated_datasets/*.npz
       └──> shallow_ml_agent(datasets)
           └──> 输出: shallow_ml_results.md, ml_results.pkl

3. Generate_report_final.py
   └──> 加载 exp2.5built_state_ottawa.pkl (PHMState)
   └──> 加载 ml_results.pkl
       └──> report_agent_node(state)
           └──> 输出: final_report.md
```

---

## 迁移建议

要将 PHMGA-NVTA_2025_Version 的功能整合回主仓库：

1. **移除硬编码路径**：使用命令行参数或配置文件
2. **创建统一的数据工厂**：参考 `goal.md` 中的 `data_factory` 概念
3. **标准化输出目录**：使用 `save/<case_name>/<timestamp>/` 结构
4. **合并配置管理**：统一 LLM 配置和 API key 管理

---

## 文件清单

### 仅存在于 PHMGA-NVTA_2025_Version 的文件：
- `creat_datasets.py`
- `train_shallow_ml.py`
- `Generate_report_final.py`
- `final_report.md` (结果文件)
- `src/configuration.py`
- `.env`, `.env.example`

### 共享的核心文件（通过 import 使用）：
- `src/agents/dataset_preparer_agent.py`
- `src/agents/shallow_ml_agent.py`
- `src/agents/report_agent.py`
- `src/states/phm_states.py`

---

## 结论

PHMGA-NVTA_2025_Version 是一个**实验工作流封装**，包含：
- 数据预处理脚本
- ML 训练脚本
- 报告生成脚本
- 实验结果文档

它与主仓库**共享核心算法**，但通过独立脚本实现了特定的论文实验流程。主要问题是硬编码路径和缺乏配置化。

---

## 已合并回主仓库的可用更新（Merged Back）

1) **GLM/DeepSeek（OpenAI-compatible）LLM 配置方法**
   - 主仓库现支持通过 `LLM_PROVIDER` 切换 `gemini | openai_compatible | deepseek | glm`
   - 通过 `.env` 配置 `*_API_KEY` 与 `*_BASE_URL`（参考仓库根目录 `.env.example`）

2) **把 NVTA 的 3 段脚本封装成可配置 CLI（去硬编码路径）**
   - `scripts/export_node_datasets.py`：从 `PHMState.pkl` 导出节点数据集（npz）
   - `scripts/train_shallow_ml_from_npz.py`：从 npz 训练浅层 ML 并保存结果
   - `scripts/generate_report_from_state.py`：从 state + 可选 ml_results 生成最终报告

3) **路径硬编码相关 bug 修复**
   - 报告生成的 DAG 图片输出路径不再固定为某个用户目录，改为遵循 `PHM_SAVE_DIR`/`state.save_dir`
