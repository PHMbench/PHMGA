# Worker Result

- worker_id: codex-local
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: ottawa_ml_openrouter_nemotron_v3
- command: `env -u OPENROUTER_API_KEY -u BIGMODEL_API_KEY -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy /mnt/k/2_work/lqql_os/lqql_06_工作与项目/03_论文流水线/p02_agent_langraph/.venv/bin/python main.py +runs=ottawa_ml_openrouter_nemotron_v3 runtime.output_dir=artifacts/paper/ottawa_ml_openrouter_nemotron_v3_qualityfix1 +runtime.provider_retry_backoff_sec=5 +runtime.provider_retry_max_backoff_sec=20`
- start_time: 2026-05-04T15:40:00+08:00
- end_time: 2026-05-04T15:42:59+08:00
- provider/model: openrouter / nvidia/nemotron-3-super-120b-a12b:free
- output_dir: `artifacts/paper/ottawa_ml_openrouter_nemotron_v3_qualityfix1`
- artifact_contract_pass: pass
- feature_separability_pass: pass
- status: accept
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

- train accuracy: 0.8258196721311475
- train macro_f1: 0.8248225746457779
- val accuracy: 0.8852459016393442
- val macro_f1: 0.8844618900893694
- test accuracy: 0.8770491803278688
- test macro_f1: 0.8774661249538376

## Gate Summary

The artifact contract passed and the run emitted a complete Stage B artifact bundle. The feature separability gate passed with `feature_count=10`, `non_empty_feature_count=10`, `constant_feature_count=0`, `train_val_rank_corr=0.7696969696969697`, and `decision=pass`. `workflow_state.json` did not record `compiled_for_rejection_evidence`; the run reached a finish decision through deterministic quality fallback after an invalid provider reflection response.

## Notes

The planner initially returned an empty plan-like JSON payload, so the model-path aggregate feature guard replaced it with a deterministic feature plan. The final report was provider-authored and artifact-consistent. This row is selection-eligible for the Ottawa side only; `selected_global_best_backend` remains pending because the matching RM101 OpenRouter Nemotron row is rejected.
