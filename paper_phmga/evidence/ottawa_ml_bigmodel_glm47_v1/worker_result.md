# Worker Result

- worker_id: codex-local
- ticket_id: 06_backend_comparison_owner.md
- run_type: backend_comparison
- experiment_id: ottawa_ml_bigmodel_glm47_v1
- command: `env -u OPENROUTER_API_KEY -u BIGMODEL_API_KEY /mnt/k/2_work/lqql_os/lqql_06_工作与项目/03_论文流水线/p02_agent_langraph/.venv/bin/python main.py +runs=ottawa_ml_bigmodel_glm47_v1 runtime.output_dir=artifacts/paper/ottawa_ml_bigmodel_glm47_v1_qualityfix2 +runtime.provider_retry_backoff_sec=5 +runtime.provider_retry_max_backoff_sec=20`
- start_time: 2026-05-04T14:42:00+08:00
- end_time: 2026-05-04T14:45:00+08:00
- provider/model: bigmodel / glm-4.7-flash
- output_dir: `artifacts/paper/ottawa_ml_bigmodel_glm47_v1_qualityfix2`
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

- train accuracy: 0.8039617486338798
- train macro_f1: 0.8041346990544024
- val accuracy: 0.819672131147541
- val macro_f1: 0.8209876827463954
- test accuracy: 0.7254098360655737
- test macro_f1: 0.7226535613558728

## Gate Summary

The artifact contract passed and the run emitted a complete Stage B artifact bundle. The feature separability gate passed with `train_val_rank_corr=0.9491525423728814`, `feature_count=9`, and `decision=pass`. DAG quality reached `finish_candidate`, and `workflow_state.json` recorded `halt_reason=null`.

## Notes

The final report used deterministic provider fallback after a report-stage provider error. This does not change the artifact-derived metrics and gate outcomes. The row is selection-eligible for the Ottawa side of Stage B, but `selected_global_best_backend` remains pending because the matching RM101 BigModel row remains rejected.
