# Stage B Backend Comparison

Current active Stage B set and selection state are sourced from `selection_status.json`.

- active_codex: `codex_cli / gpt-5.3-codex`
- active_openrouter: `openrouter / z-ai/glm-4.5-air:free`
- selected_backend_status: `pending`

- historical_failure_not_in_active_set: `ottawa_ml_openrouter_v1`, `rm101_ml_openrouter_v1`

| experiment_id | dataset | graph_path | status | subdoc |
| --- | --- | --- | --- | --- |
| ottawa_ml_codex_v1 | Ottawa | ml | reject | [subdoc](../subdocs/ottawa_ml_codex_v1.md) |
| ottawa_ml_openrouter_glm_v1 | Ottawa | ml | pending | [subdoc](../subdocs/ottawa_ml_openrouter_glm_v1.md) |
| rm101_ml_codex_v1 | RM101 | ml | reject | [subdoc](../subdocs/rm101_ml_codex_v1.md) |
| rm101_ml_openrouter_glm_v1 | RM101 | ml | pending | [subdoc](../subdocs/rm101_ml_openrouter_glm_v1.md) |
| ottawa_ml_openrouter_v1 | Ottawa | ml | reject | [subdoc](../subdocs/ottawa_ml_openrouter_v1.md) |
| rm101_ml_openrouter_v1 | RM101 | ml | reject | [subdoc](../subdocs/rm101_ml_openrouter_v1.md) |
