# Stage B Pending Closure Contract

Generated at: 2026-05-31

## Scope

This contract covers only the currently pending Stage B backend-comparison rows:

- `ottawa_ml_codex_v3`
- `rm101_ml_codex_v3`
- `rm101_ml_openrouter_glm_v2`

It does not update the result ledger, select a backend, unlock Stage C/D, or upgrade any manuscript claim. It records local execution readiness and the minimum next commands required to close the pending rows.

## Preflight Evidence

The following preflight commands completed successfully in this workspace:

| experiment_id | preflight command | local readiness signal |
| --- | --- | --- |
| `ottawa_ml_codex_v3` | `python main.py runtime.action=preflight +runs=ottawa_ml_codex_v3` | Ottawa real-data config loaded; `ml` graph path; 36 samples split 24/6/6; window 32768, stride 16384; `codex` binary found. |
| `rm101_ml_codex_v3` | `python main.py runtime.action=preflight +runs=rm101_ml_codex_v3` | RM101 real-data config loaded; `ml` graph path; 240 samples split 166/34/40; window 4096, stride 2048; `codex` binary found. |
| `rm101_ml_openrouter_glm_v2` | `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py runtime.action=preflight +runs=rm101_ml_openrouter_glm_v2` | RM101 real-data config loaded; `ml` graph path; 240 samples split 166/34/40; window 4096, stride 2048; `OPENROUTER_API_KEY` presence detected. |

Preflight proves configuration, data access, and transport setup only. It does not prove artifact-contract pass, feature-separability pass, metrics, or selection eligibility.

## Full-Run Attempt Evidence

`ottawa_ml_codex_v3` was run once with:

```bash
python main.py +runs=ottawa_ml_codex_v3
```

The command exited with code 0 and emitted a complete artifact bundle in `artifacts/paper/ottawa_ml_codex_v3`. The test metrics were accuracy `0.8770491803278688` and macro-F1 `0.8774661249538376`; `feature_separability_summary.json` reported decision `pass`.

This is not clean Codex backend evidence. The same artifact bundle records provider fallback:

- `planner_transport_trace.json` reports `planner_smoke` status `returncode_error`.
- The stderr preview reports `failed to initialize in-process app-server client: Read-only file system (os error 30)`.
- `workflow_state.json` records `provider_plan_fallback: LLMProviderError; generated deterministic normalize/fft/feature plan`.
- `final_report.md` reports deterministic provider fallback after `LLMProviderError`.

Therefore `ottawa_ml_codex_v3` remains `needs_rerun` / not selection-eligible until a clean Codex provider run is authorized and reviewed. A non-sandbox retry was requested with a separate output directory, but approval was rejected because it may contact an external Codex provider and write new artifacts without explicit user authorization.

## Execution Priority

1. Rerun the Codex v3 pair only after explicit authorization for a clean Codex provider run:
   - `python main.py +runs=ottawa_ml_codex_v3`
   - `python main.py +runs=rm101_ml_codex_v3`
2. Run `rm101_ml_openrouter_glm_v2` only as a diagnostic/backfill row unless the rejected Ottawa OpenRouter GLM row is explicitly reopened:
   - `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py +runs=rm101_ml_openrouter_glm_v2`

Rationale: in the current Stage B state, the Codex v3 pair is the only pending same-backend Ottawa/RM101 pair that can unlock `selected_global_best_backend` without reopening a rejected row. `rm101_ml_openrouter_glm_v2` cannot by itself unlock selection because `ottawa_ml_openrouter_glm_v2` is already rejected. The first Ottawa Codex v3 attempt produced a useful artifact bundle but not clean Codex-provider evidence.

## Artifact Gate

Each completed Stage B run must emit all hard-gate artifacts before it can be accepted:

- `validated_dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `feature_list.json`
- `feature_separability_summary.json`
- `artifact_index.json`
- `metrics.json`
- `final_report.md`

For `ml` rows, `feature_separability_summary.json` must support a non-collapsed feature pipeline with an explicit separability conclusion.

## Writeback Order

For each full run:

1. Run the command from this contract or the manual runbook.
2. Inspect the target artifact directory.
3. Write `doc/experiments/handoff/results/<experiment_id>.md`.
4. Update `doc/experiments/01_result_ledger.md`.
5. Let the harness engineer set `artifact_contract_pass`, `feature_separability_pass`, and `selection_eligible`.
6. Let the coordinator update `selection_status.json` and `doc/experiments/02_main_tables.md` only after gate review.

## Failure Handling

- If a Codex full run fails before artifact emission, record it as runtime failure with the failing command and stdout/stderr summary.
- If an OpenRouter full run hits rate limiting again, keep it as provider-bound reject or needs-rerun evidence; do not reinterpret it as a local proxy failure unless the error proves that.
- Do not write a pending row into Table 1, Table 2, selected-backend state, or manuscript performance claims.

## Manuscript Implication

Until at least one backend has accepted Ottawa and RM101 Stage B rows, the P2 manuscript may report only bounded Stage B ledger evidence and RM101 simulated-planner mechanism evidence. Cross-dataset performance, selected-backend readiness, Stage C/D success, statistical significance, and final submission readiness remain blocked.
