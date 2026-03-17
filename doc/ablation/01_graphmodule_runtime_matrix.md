# GraphModule Runtime Matrix

| experiment_id | dataset | path | phase | control_mode | notes |
| --- | --- | --- | --- | --- | --- |
| ottawa_torch_compiled_v1 | Ottawa | torch | compiled | fixed | baseline |
| ottawa_torch_module_v1 | Ottawa | torch | module_runtime | fixed | GraphModule only |
| ottawa_torch_learnable_gate_v1 | Ottawa | torch | learnable_control | gated | gate only |
| ottawa_torch_learnable_attention_v1 | Ottawa | torch | learnable_control | attention | channel self-attention + attention fusion |
| rm101_torch_compiled_v1 | RM101 | torch | compiled | fixed | baseline |
| rm101_torch_module_v1 | RM101 | torch | module_runtime | fixed | GraphModule only |
| rm101_torch_learnable_gate_v1 | RM101 | torch | learnable_control | gated | gate only |
| rm101_torch_learnable_attention_v1 | RM101 | torch | learnable_control | attention | channel self-attention + attention fusion |
