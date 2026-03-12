# 01 DAG And Operators

## DAG IR

正式链路：

`NetworkX DAG -> node_link_data -> DAG JSON -> validated DAG JSON`

每个 DAG 节点必须声明：

- `node_id`
- `op_uid`
- `name`
- `kind`
- `params`
- `parents`
- `in_shape`
- `out_shape`
- `backend_availability`
- `execution_role`

当前代码中的最小对应对象是：

- `DagNode`
- `DagEdge`
- `DagJson`
- `DAGTracker`

每条边只表达依赖与拓扑，不承载隐藏状态。所有进入 bridge 的 DAG JSON 都必须先通过 schema 校验、无环检查和 shape contract 检查。

## 当前最小 shape contract

- 输入节点：`[1, window_size]`
- 变换节点：保持或改变最后一维，但必须显式写入 `in_shape` / `out_shape`
- 特征节点：最小输出是 `[1]`

当前 `execute_agent()` 使用 `normalize -> fft_mag -> feature_*` 的最小链条来产出可桥接的 feature nodes。

## DAGTracker

- 生成期可使用 `networkx`。
- 导出期统一写成 JSON。
- 校验失败的 DAG 不得进入 bridge。

## 一魂三体算子系统

同一个物理/信号处理算子只保留一份语义定义，通过三个后端接口暴露：

```python
class BaseIsomorphicOperator:
    op_uid: str
    name: str

    def forward_np(self, x, **kwargs):
        raise NotImplementedError

    def forward_pt(self, x, **kwargs):
        raise NotImplementedError

    def forward_sym(self, x, **kwargs):
        raise NotImplementedError
```

并非所有算子都必须完整实现三种后端，但每个算子都必须显式声明：

- 参数 schema
- 输入输出 shape rule
- backend availability
- `trainable | fixed | proxy | outer_only`

## 当前实现态

当前 `OperatorSpec` 与 `OperatorCatalog` 已经落地了最小算子集：

- `signal.normalize`
- `signal.fft_mag`
- `feature.mean`
- `feature.std`
- `feature.rms`

需要明确的是：当前仓库里部分算子只是声明 `backend_availability`，并不意味着 `forward_pt` 已经完整实现。当前最小可运行后端仍以 `forward_np` 为主。

## OperatorCatalog

论文版最小目录以统一 `OperatorCatalog` 为准，不再把 `tools`、`bridge`、`model` 视作平行宇宙。当前最小实现包含输入标准化、频域变换和统计特征三类算子，用于支撑 `dag_only`、`ml`、`torch` 三条路径的闭环。
