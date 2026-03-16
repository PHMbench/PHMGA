# Operator System Review

**审查日期**: 2026-03-15  
**审查范围**: `src/operators/`, `src/bridge/compiler.py`, `src/agents/execute_agent.py`

## Scope

这份文档是 debug review，不是正式合同源。它只回答三件事：

1. 当前算子系统哪些判断是准确的
2. 之前 review 中哪些结论需要纠正
3. 下一阶段真正值得做的主任务是什么

正式结构边界仍以：

- [README.md](/home/user/LQ/B_Signal/PHMGA/README.md)
- [01_dag_and_operators.md](/home/user/LQ/B_Signal/PHMGA/doc/structure/01_dag_and_operators.md)
- [02_workflow_and_bridge.md](/home/user/LQ/B_Signal/PHMGA/doc/structure/02_workflow_and_bridge.md)

为准。

## Accurate Findings

### 1. `catalog.py` 之前耦合过重

这个判断是准确的。之前的 [catalog.py](/home/user/LQ/B_Signal/PHMGA/src/operators/catalog.py) 同时承担了：

- 具体算子实现
- helper 函数
- alias registry
- catalog assembly
- prompt summary 输出

现在已经按五类拆分到：

- [expand_ops.py](/home/user/LQ/B_Signal/PHMGA/src/operators/expand_ops.py)
- [transform_ops.py](/home/user/LQ/B_Signal/PHMGA/src/operators/transform_ops.py)
- [aggregate_ops.py](/home/user/LQ/B_Signal/PHMGA/src/operators/aggregate_ops.py)
- [multi_ops.py](/home/user/LQ/B_Signal/PHMGA/src/operators/multi_ops.py)
- [decision_ops.py](/home/user/LQ/B_Signal/PHMGA/src/operators/decision_ops.py)

而 [catalog.py](/home/user/LQ/B_Signal/PHMGA/src/operators/catalog.py) 已收敛为 assembly 层。

### 2. multi-parent compiled lineage 仍是核心缺口

这个判断仍然准确，而且优先级高。

当前 workflow 端已经能合法构造：

- `multi.concatenate`
- `multi.cross_correlation`

但 bridge/compiler 仍主要围绕单父 lineage 组织 compiled feature 计划。这意味着：

- DAG 证据层已经支持 multi-parent
- compiled feature lineage 还没有把 multi-parent 当成一等公民

所以下一阶段真正该做的是：

- multi-parent lineage 回溯
- multi-parent compiled feature support

而不是继续无止境补算子。

### 3. 边界测试仍可加强

当前回归已经覆盖：

- richer operator metadata
- planner/executor 对五类算子的最小闭环
- enriched `dag_only` / `ml`

但还有值得补的边界：

- multi-parent compiled feature 行为
- operator arity 错误时的更细粒度校验
- `decision` 作为 side-output 的 bridge/report 一致性

这是准确问题，但优先级低于 multi-parent compiled support 本身。

## Needs Correction

### 1. `BaseIsomorphicOperator` 不是 `BaseModel`

之前的说法不准确。

实际代码见 [base.py](/home/user/LQ/B_Signal/PHMGA/src/operators/base.py)：

- `OperatorSpec` 是 `pydantic.BaseModel`
- `BaseIsomorphicOperator` 是普通 Python 类

也就是说：

- metadata 在 `OperatorSpec`
- runtime behavior 在 `BaseIsomorphicOperator`

不是“所有字段都平铺在 operator 实例上”。

### 2. metadata 不在 operator 实例字段平铺

当前统一合同是：

- `operator.spec.op_uid`
- `operator.spec.schema_category`
- `operator.spec.rank_class`
- `operator.spec.input_spec`
- `operator.spec.output_spec`

所以之前把 review 写成“operator 本体字段设计混乱”不准确。真正的问题不是字段平铺，而是之前 `catalog.py` 把太多职责揉在一起。

### 3. 当前算子数量和覆盖范围已经变化

之前 review 里写“12 个算子”，已经过时。

当前 runnable / half-runnable 范围至少包括：

- `signal.normalize`
- `signal.fft_mag`
- `signal.stft`
- `signal.patch`
- `signal.filter`
- `signal.hilbert_envelope`
- `signal.psd`
- `feature.mean`
- `feature.std`
- `feature.rms`
- `feature.kurtosis`
- `feature.crest_factor`
- `feature.band_power`
- `feature.spectral_centroid`
- `multi.concatenate`
- `multi.cross_correlation`
- `decision.threshold`

因此 review 里关于“当前只实现很小算子集”的表述需要按现状重写。

### 4. `op_name` 和 `name` 不是重复字段

这个判断不成立。

当前设计里：

- `op_name`
  - planner / prompt / alias 侧使用的短 token
- `name`
  - 人类可读标签，供 manifest、report、debug 使用

两者职责不同，不应该因为“看起来相似”就删除其一。

### 5. `legal_paths` 不等于 backend method availability

之前把：

- `legal_paths`

理解成：

- 当前是否已经写了 `forward_pt`

这是不准确的。

当前语义是：

- `legal_paths` 表示该算子在某条 graph path 上是否合法出现
- `backend_availability` 表示当前 runtime 可用的执行表面

因此不能仅因为某个算子没有独立 `forward_pt`，就把它从 `torch` path legality 中删除。当前 `torch` path 仍是受控 fallback 合同，不是完整 PyTorch executor。

### 6. report 不是 provider-backed generation

当前 report 仍然是 deterministic / rule-based renderer。

事实来源：

- [report_agent.py](/home/user/LQ/B_Signal/PHMGA/src/agents/report_agent.py)
- [client.py](/home/user/LQ/B_Signal/PHMGA/src/llm/client.py)

当前行为是：

- `report_agent` 组装上下文
- `OfflineLLM.render_report()` 用固定 markdown 结构输出报告

这不是缺陷，而是当前阶段的有意选择：

- 保持可测
- 保持可复现
- 保持 artifact-to-report 映射稳定

## Recommended Next Work

按当前实现态，下一阶段优先级应固定为：

1. **bridge 的 multi-parent compiled support**
   - 让 `multi.concatenate` / `multi.cross_correlation` 真正进入 compiled feature plan
2. **multi-parent 相关边界测试**
   - 证明 compiled lineage、report、manifest 三者一致
3. **保持 report deterministic**
   - 不在这一轮切 provider-backed report

不建议在下一轮优先做：

- 再次大规模扩 operator schema
- 默认切真实 LLM report
- 提前升级完整 torch trainer

## Direct Conclusion

当前算子系统的真实状态可以概括为：

- 统一合同层是正确的
- `catalog` 解耦是必要且已经开始落地
- 下一阶段的真正瓶颈不是“再补多少算子”，而是 **multi-parent compiled support**
- report 当前保持 deterministic 是合理默认，不应被误诊为必须立即替换的缺陷
