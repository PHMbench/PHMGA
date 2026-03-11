# Cases Structure

## 职责
- `cases` 是运行范式入口层，不是实验枚举层。
- 它的职责是接收 resolved config，选择对应 runner，并组织 runtime artifacts。
- 实验差异应主要来自 Hydra config groups 和 CLI overrides，而不是为每个实验新增一个 `caseX.py`。

## 为什么需要这一层
- `config` 负责回答“跑什么”：数据、模型、graph、builder 参数、执行开关。
- `cases` 负责回答“按什么运行范式跑”：把配置交给对应 runner，进入统一运行主链。
- 没有这一层时，`main.py` 会直接耦合具体 runner，运行目录组织和配置落盘也会散回脚本入口。

## 为什么现在看起来复杂
- 当前复杂度主要来自 [src/cases/case1.py](/home/user/LQ/B_Signal/PHMGA/src/cases/case1.py)，它仍承载了过多运行时装配逻辑。
- 这些逻辑多数并不是“case1 专属逻辑”，而是共享 runner 逻辑，例如：
  - preflight 执行和 strict gate
  - LLM 配置绑定与校验
  - builder/executor graph 选择
  - state 恢复与保存策略
  - ablation、source mode、state save mode 解析
  - builder/executor 生命周期调度
- 所以现在的复杂并不说明系统需要 `case1/2/3` 并列扩张，而是说明共享 runner 逻辑还没有完全下沉。

## 推荐心智模型
- `case != experiment != dataset`。
- 推荐理解方式：
  - `config` 决定跑什么数据、什么模型、什么 graph、什么参数。
  - `case runner` 决定按什么运行范式执行。
  - 绝大多数实验应继续复用同一个 runner；当前通常就是 `case1`。
- 换句话说，同一范式下，换数据集、换 split、换模型 profile、换 builder 参数，本质上都应是“换配置”，不是“新增 case 文件”。

## 正式入口
- [main.py](/home/user/LQ/B_Signal/PHMGA/main.py)
- [src/cases/base_runner.py](/home/user/LQ/B_Signal/PHMGA/src/cases/base_runner.py)
- [src/cases/registry.py](/home/user/LQ/B_Signal/PHMGA/src/cases/registry.py)
- [src/cases/case1.py](/home/user/LQ/B_Signal/PHMGA/src/cases/case1.py)

主链是：

- `main.py` 负责 CLI/Hydra compose 入口。
- `base_runner` 负责 resolved config 落盘、runtime artifact 准备、按 `cases.selected` 解析 runner。
- `registry` 负责把 case 名称映射到 runner。
- `case1.py` 是当前唯一内建 runner，而不是“每个实验都应该仿照复制一个文件”的模板。

## 输入/输出边界
- 输入：resolved config、选定 case 名称、save root。
- 输出：runtime config、metadata snapshot、runtime manifest、case 产物目录、实际 runner 调用。
- 不负责：Hydra group 定义、graph 具体实现、模型训练细节、数据 schema 真相。

## 何时需要新建 case runner
- 只有当运行范式发生变化时，才应该新增新的 case runner。
- 典型场景包括：
  - 不再走现有 builder/executor 主链
  - 输入输出合同显著不同
  - 运行生命周期不同，无法通过现有 runner + config 复用
  - 需要完全不同的 artifact 组织或恢复语义

以下情况通常不构成新增 runner 的理由：

- 只换数据集
- 只换 split / IDs / metadata / h5
- 只换模型 profile 或模型配置
- 只换 graph 选择
- 只换 builder 参数、executor 开关或其他常规运行参数

## 推荐用法
- 新实验优先新增或修改 `config/cases/*.yaml`，以及相关 Hydra groups。
- 通过 `data.*`、`model.*`、`graphs.*`、`builder.*`、`run_executor` 等参数组合出具体实验。
- 默认假设是“复用现有 runner，切换 config”。
- 不要因为来了一个新数据任务，就复制出 `case2.py`、`case3.py`。

## 当前状态与差距
- 已实现：`main.py -> base_runner -> registry -> runner` 主链已经成立，runner 选择已经由配置驱动。
- 当前事实：`base_runner` 已经负责 runtime 准备和 runner 分发，`cases.selected` 是当前 runner 选择入口。
- 仍未完全收口：`case1.py` 依然偏重，承载了较多本应继续下沉的共享运行逻辑。
- 目标态是更薄的通用 runner：配置决定实验内容，runner 只承载运行范式。

## 冗余与历史包袱
- `case1.py` 曾把模型配置回填到 `data_cfg`，这让 case 层越界到了 model truth。
- 历史 notebook case 入口已经删除，但仍需防止临时实验逻辑再次回流到 `src/cases/`。
- 当前仍有一部分共享运行逻辑滞留在 `case1.py`，这是过渡实现，不应被当作长期结构目标。

## 按 v1.0 的下一步
- 继续把 `case1.py` 中的共享运行逻辑下沉到更通用的 runner helper。
- 让更多实验差异收敛到 Hydra config groups，而不是 case 文件。
- 保持“新增实验先改配置，新增 runner 只因范式变化”的约束。
