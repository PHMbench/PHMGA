# Learnable Control Ablation

## Control Modes

- `fixed`
- `gated`
- `attention`

## Attention Semantics

- 单父节点：`channel_self_attention`
- 多父节点：`attention_fusion`

## Metrics To Record

- test accuracy
- macro-F1
- output-node importance
- gate weights summary
- attention weights summary

## Decision Boundary

- `decision` 节点不进入训练主链
- learnable control 只作用于 runtime wrapper，不回流到 bridge
