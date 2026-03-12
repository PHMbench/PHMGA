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

每条边只表达依赖与拓扑，不承载隐藏状态。所有进入 bridge 的 DAG JSON 都必须先通过 schema 校验、无环检查和 shape contract 检查。

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

## OperatorCatalog

论文版最小目录以统一 `OperatorCatalog` 为准，不再把 `tools`、`bridge`、`model` 视作平行宇宙。当前最小实现包含输入标准化、频域变换和统计特征三类算子，用于支撑 `dag_only`、`ml`、`torch` 三条路径的闭环。
