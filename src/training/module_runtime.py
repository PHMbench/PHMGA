"""Torch module runtime built on top of compiled execution plans."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_f
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    torch_f = None

from src.bridge import CompiledExecutionNode, FeaturePipelinePlan, ModelBuildPlan
from src.operators import OperatorCatalog


LEARNABLE_WAVEFILTER_OPS = {
    "signal.wavefilters",
    "signal.wavelet_ricker",
    "signal.wavelet_chirplet",
    "signal.wavelet_laplace",
    "signal.wavelet_morlet",
}


def _require_torch():
    if torch is None or nn is None or torch_f is None:
        raise ModuleNotFoundError("PyTorch is required for GraphModule runtime execution.")
    return torch


class SoftmaxGate(nn.Module):
    """A simple scalar gate that preserves the wrapped operator topology."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.tau = max(float(tau), 1e-6)
        self._last_weights: list[float] = [0.5, 0.5]

    def forward(self, x):
        weights = torch.softmax(self.logits / self.tau, dim=0)
        self._last_weights = [float(value) for value in weights.detach().cpu().tolist()]
        return x * weights[-1]

    def summary(self) -> Dict[str, Any]:
        return {"gate_weights": self._last_weights}


class ChannelSelfAttention(nn.Module):
    """Lightweight channel self-attention that keeps channel-first shape intact."""

    def __init__(self, *, heads: int, tau: float, dropout: float) -> None:
        super().__init__()
        self.heads = max(int(heads), 1)
        self.tau = max(float(tau), 1e-6)
        self.dropout = nn.Dropout(float(dropout))
        self.q_scale = nn.Parameter(torch.ones(self.heads, dtype=torch.float32))
        self.k_scale = nn.Parameter(torch.ones(self.heads, dtype=torch.float32))
        self._last_attention: list[list[float]] = []

    def forward(self, x):
        if x.ndim < 2 or x.shape[0] <= 1:
            self._last_attention = [[1.0]]
            return x
        flat = x.reshape(x.shape[0], -1)
        summaries = flat.mean(dim=-1, keepdim=True)
        head_outputs: list[Any] = []
        attention_matrices: list[Any] = []
        for head_index in range(self.heads):
            query = summaries * self.q_scale[head_index]
            key = summaries * self.k_scale[head_index]
            scores = (query @ key.T) / self.tau
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            attention_matrices.append(weights.detach())
            head_outputs.append(weights @ flat)
        attended = torch.stack(head_outputs, dim=0).mean(dim=0).reshape_as(x)
        self._last_attention = (
            torch.stack(attention_matrices, dim=0).mean(dim=0).detach().cpu().tolist()
        )
        return 0.5 * (x + attended)

    def summary(self) -> Dict[str, Any]:
        return {"attention_summary": self._last_attention}


class BranchAttentionFusion(nn.Module):
    """Attention-style branch scaling used before multi-input operators."""

    def __init__(self, *, tau: float, dropout: float) -> None:
        super().__init__()
        self.tau = max(float(tau), 1e-6)
        self.dropout = nn.Dropout(float(dropout))
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self._last_weights: list[float] = []

    def forward(self, inputs: Iterable[Any]) -> List[Any]:
        inputs_list = list(inputs)
        if len(inputs_list) <= 1:
            self._last_weights = [1.0]
            return inputs_list
        summaries = torch.stack([part.reshape(-1).abs().mean() for part in inputs_list], dim=0)
        weights = torch.softmax((summaries * self.scale.squeeze(0)) / self.tau, dim=0)
        weights = self.dropout(weights)
        self._last_weights = [float(value) for value in weights.detach().cpu().tolist()]
        return [weight * part for weight, part in zip(weights, inputs_list)]

    def summary(self) -> Dict[str, Any]:
        return {"attention_summary": self._last_weights}


class RuntimeNodeModule(nn.Module):
    """One compiled execution node materialized into a torch runtime module."""

    def __init__(
        self,
        node: CompiledExecutionNode,
        *,
        operator,
        phase: str,
        control_default_mode: str,
        tau: float,
        attention_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.node = node
        self.operator = operator
        self.phase = str(phase)
        self.is_multi = node.kind == "multi"
        self.learnable_params = nn.ParameterDict()
        self.static_params: Dict[str, Any] = dict(node.params)
        self.control_mode = "fixed"
        self.gate: SoftmaxGate | None = None
        self.single_attention: ChannelSelfAttention | None = None
        self.multi_attention: BranchAttentionFusion | None = None

        if self.phase == "learnable_control" and node.op_uid in LEARNABLE_WAVEFILTER_OPS:
            for name, value in sorted(node.params.items()):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                self.learnable_params[name] = nn.Parameter(torch.tensor(float(value), dtype=torch.float32))

        requested_mode = control_default_mode if self.phase == "learnable_control" else "fixed"
        if requested_mode == "gated":
            self.control_mode = "gated"
            self.gate = SoftmaxGate(tau=tau)
        elif requested_mode == "attention":
            if self.is_multi:
                self.control_mode = "attention_fusion"
                self.multi_attention = BranchAttentionFusion(tau=tau, dropout=attention_dropout)
            else:
                self.control_mode = "channel_self_attention"
                self.single_attention = ChannelSelfAttention(
                    heads=attention_heads,
                    tau=tau,
                    dropout=attention_dropout,
                )

    def _current_params(self) -> Dict[str, Any]:
        params = dict(self.static_params)
        for name, value in self.learnable_params.items():
            params[name] = value
        return params

    def forward(self, *inputs):
        torch_module = _require_torch()
        params = self._current_params()
        if self.is_multi:
            prepared_inputs = list(inputs)
            if self.multi_attention is not None:
                prepared_inputs = self.multi_attention(prepared_inputs)
            result = self.operator.forward_pt(prepared_inputs, **params)
            if self.gate is not None:
                result = self.gate(result)
        else:
            if len(inputs) != 1:
                raise ValueError(f"Single-input node {self.node.node_id} expected one input, got {len(inputs)}.")
            result = self.operator.forward_pt(inputs[0], **params)
            if self.gate is not None:
                result = self.gate(result)
            if self.single_attention is not None:
                result = self.single_attention(result)
        if not torch_module.is_tensor(result):
            raise TypeError(f"Operator {self.node.op_uid} must return a torch.Tensor in module runtime.")
        return result.to(dtype=torch_module.float32)

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"mode": self.control_mode}
        if self.learnable_params:
            summary["learnable_params"] = {
                name: float(value.detach().cpu().item())
                for name, value in self.learnable_params.items()
            }
        if self.gate is not None:
            summary.update(self.gate.summary())
        if self.single_attention is not None:
            summary.update(self.single_attention.summary())
        if self.multi_attention is not None:
            summary.update(self.multi_attention.summary())
        return summary


class OperatorModuleFactory:
    """Materialize compiled execution nodes into runtime wrappers."""

    def __init__(
        self,
        catalog: OperatorCatalog,
        *,
        phase: str,
        control_default_mode: str,
        tau: float,
        attention_heads: int,
        attention_dropout: float,
    ) -> None:
        self.catalog = catalog
        self.phase = phase
        self.control_default_mode = control_default_mode
        self.tau = tau
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

    def build(self, node: CompiledExecutionNode) -> RuntimeNodeModule:
        return RuntimeNodeModule(
            node,
            operator=self.catalog.get(node.op_uid),
            phase=self.phase,
            control_default_mode=self.control_default_mode,
            tau=self.tau,
            attention_heads=self.attention_heads,
            attention_dropout=self.attention_dropout,
        )


class GraphModule(nn.Module):
    """Execute a compiled subgraph as a torch module."""

    def __init__(
        self,
        plan: FeaturePipelinePlan | ModelBuildPlan,
        catalog: OperatorCatalog,
        *,
        phase: str,
        control_default_mode: str,
        tau: float,
        attention_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.execution_nodes = list(plan.execution_nodes)
        self.output_specs = list(plan.output_specs)
        factory = OperatorModuleFactory(
            catalog,
            phase=phase,
            control_default_mode=control_default_mode,
            tau=tau,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
        )
        self.node_modules = nn.ModuleDict(
            {
                node.node_id: factory.build(node)
                for node in self.execution_nodes
                if node.kind != "input"
            }
        )

    @staticmethod
    def _ordered_inputs(values_by_node: Dict[str, Any], node: CompiledExecutionNode) -> List[Any]:
        if node.input_bindings:
            binding_items = sorted(
                node.input_bindings.items(),
                key=lambda item: int(item[0].removeprefix("arg")),
            )
            return [values_by_node[parent_id] for _, parent_id in binding_items]
        return [values_by_node[parent_id] for parent_id in node.parents]

    def forward_features(self, window):
        torch_module = _require_torch()
        if window.ndim != 2:
            raise ValueError(f"GraphModule expects one channel-first window, got shape {tuple(window.shape)}.")
        values_by_node: Dict[str, Any] = {}
        for node in self.execution_nodes:
            if node.kind == "input":
                if node.channel_index is None:
                    raise ValueError(f"Input node {node.node_id} is missing channel_index.")
                values_by_node[node.node_id] = window[[node.channel_index], :].to(dtype=torch_module.float32)
                continue
            runtime_node = self.node_modules[node.node_id]
            ordered_inputs = self._ordered_inputs(values_by_node, node)
            values_by_node[node.node_id] = runtime_node(*ordered_inputs)
        outputs = [values_by_node[spec.output_node_id].reshape(-1).to(dtype=torch_module.float32) for spec in self.output_specs]
        if not outputs:
            return torch_module.zeros((0,), dtype=torch_module.float32, device=window.device)
        return torch_module.cat(outputs, dim=0)

    def forward(self, windows):
        torch_module = _require_torch()
        if windows.ndim == 2:
            return self.forward_features(windows).reshape(1, -1)
        if windows.ndim != 3:
            raise ValueError(f"GraphModule expects batch windows with rank 3, got rank {windows.ndim}.")
        return torch_module.stack([self.forward_features(sample) for sample in windows], dim=0)

    def control_statistics(self) -> Dict[str, Dict[str, Any]]:
        summaries: Dict[str, Dict[str, Any]] = {}
        for node_id, runtime_node in self.node_modules.items():
            summary = runtime_node.summary()
            if summary.get("mode") != "fixed" or summary.get("learnable_params"):
                summaries[node_id] = summary
        return summaries
