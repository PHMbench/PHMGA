# Stage B Backend Comparison

Current active Stage B set and selection state are sourced from `selection_status.json`.

- active_bigmodel: `bigmodel / glm-4.7-flash`
- active_codex: `codex_cli / gpt-5.3-codex`
- active_openrouter: `openrouter / z-ai/glm-4.5-air:free`
- selected_backend_status: `pending`

| experiment_id | dataset | graph_path | status | subdoc |
| --- | --- | --- | --- | --- |
| ottawa_ml_codex_v3 | Ottawa | ml | pending | [subdoc](../subdocs/ottawa_ml_codex_v3.md) |
| ottawa_ml_openrouter_glm_v2 | Ottawa | ml | reject | [subdoc](../subdocs/ottawa_ml_openrouter_glm_v2.md) |
| ottawa_ml_openrouter_nemotron_v3 | Ottawa | ml | accept | [subdoc](../subdocs/ottawa_ml_openrouter_nemotron_v3.md) |
| ottawa_ml_bigmodel_glm47_v1 | Ottawa | ml | accept | [subdoc](../subdocs/ottawa_ml_bigmodel_glm47_v1.md) |
| rm101_ml_codex_v3 | RM101 | ml | pending | [subdoc](../subdocs/rm101_ml_codex_v3.md) |
| rm101_ml_openrouter_glm_v2 | RM101 | ml | pending | [subdoc](../subdocs/rm101_ml_openrouter_glm_v2.md) |
| rm101_ml_openrouter_nemotron_v3 | RM101 | ml | reject | [subdoc](../subdocs/rm101_ml_openrouter_nemotron_v3.md) |
| rm101_ml_bigmodel_glm47_v1 | RM101 | ml | reject | [subdoc](../subdocs/rm101_ml_bigmodel_glm47_v1.md) |
