# Worker Result

- worker_id: codex-local
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: rm101_ml_openrouter_nemotron_v3
- command: `env -u OPENROUTER_API_KEY -u BIGMODEL_API_KEY -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /mnt/k/2_work/lqql_os/lqql_06_工作与项目/03_论文流水线/p02_agent_langraph/.venv/bin/python main.py +runs=rm101_ml_openrouter_nemotron_v3 runtime.output_dir=artifacts/paper/rm101_ml_openrouter_nemotron_v3_qualityfix1 +runtime.provider_retry_backoff_sec=5 +runtime.provider_retry_max_backoff_sec=20`
- start_time: 2026-05-04T15:50:00+08:00
- end_time: 2026-05-04T16:15:38+08:00
- provider/model: openrouter / nvidia/nemotron-3-super-120b-a12b:free
- output_dir: `artifacts/paper/rm101_ml_openrouter_nemotron_v3_qualityfix1`
- artifact_contract_pass: pass
- feature_separability_pass: pass
- status: reject
- ledger_updated: yes

## Artifact Checklist

- validated_dag.json: yes
- compiled_dag_manifest.json: yes
- feature_pipeline.json: yes
- feature_list.json: yes
- feature_separability_summary.json: yes
- artifact_index.json: yes
- metrics.json: yes
- final_report.md: yes

## Metrics Summary

- train accuracy: 0.3318407319116036
- train macro_f1: 0.2605086545945874
- val accuracy: 0.24284366152878264
- val macro_f1: 0.19280607248842027
- test accuracy: 0.2371657754010695
- test macro_f1: 0.18337824193501234

## Gate Summary

The artifact contract passed and the feature separability gate passed with `feature_count=41`, `non_empty_feature_count=41`, `constant_feature_count=0`, `train_val_rank_corr=0.6148083623693381`, and `decision=pass`. The row remains rejected because `workflow_state.json:path_artifacts.workflow_exit` records `max_iterations=4 reached before finish`, `compiled_for_rejection_evidence=true`, and `last_reflection_decision=need_replan`.

## Notes

The free OpenRouter model returned empty or malformed planner payloads in multiple rounds. Local planner guards and deterministic provider fallbacks preserved a complete reject-evidence bundle, but the final report was rendered by deterministic provider fallback after an `LLMProviderError`. This row is not selection-eligible and must not enter main tables.
