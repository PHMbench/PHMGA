# Ticket: backend-comparison-owner

## Goal

运行 Stage B backend comparison，验证 PHMGA 全链路、artifact contract 和 feature separability gate。

## Allowed Commands

```bash
python main.py runtime.action=preflight +runs=ottawa_ml_codex_v3
python main.py +runs=ottawa_ml_codex_v3
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py runtime.action=preflight +runs=ottawa_ml_openrouter_nemotron_v3
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py +runs=ottawa_ml_openrouter_nemotron_v3
python main.py runtime.action=preflight +runs=rm101_ml_codex_v3
python main.py +runs=rm101_ml_codex_v3
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py runtime.action=preflight +runs=rm101_ml_openrouter_nemotron_v3
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py +runs=rm101_ml_openrouter_nemotron_v3
```

## Expected Experiment IDs

- `ottawa_ml_codex_v3`
- `ottawa_ml_openrouter_nemotron_v3`
- `rm101_ml_codex_v3`
- `rm101_ml_openrouter_nemotron_v3`

## Worker ID

- `backend-comparison-owner`

## Result Files

- `doc/experiments/handoff/results/ottawa_ml_codex_v3.md`
- `doc/experiments/handoff/results/ottawa_ml_openrouter_nemotron_v3.md`
- `doc/experiments/handoff/results/rm101_ml_codex_v3.md`
- `doc/experiments/handoff/results/rm101_ml_openrouter_nemotron_v3.md`

## Preconditions

- `codex` CLI 已安装且 `codex login` 已完成
- `OPENROUTER_API_KEY` 已设置
- 当前 active Stage B set 以 `doc/experiments/01_result_ledger.md` 顶部 YAML block 为准
- 本轮 OpenRouter formal run 必须清掉本地代理环境变量
- 旧 `*_v1` comparison row 只保留为历史失败记录，不是本轮执行目标

## Success Criteria

- 所有成功运行的目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md`
- ledger 中：
  - 4 行都保持 `backend comparison candidate`
  - 不把任何 comparison row 写进 main rows
  - 普通 worker 不直接把 `selection_eligible` 写成最终事实

## Failure Rule

- 不改 provider family
- 不把 OpenRouter 结果写进 main rows
- 不把 Codex comparison row 写成 frozen main
- 失败时记 `reject`，并写明是 `artifact gate`、`separability gate` 还是 runtime 失败
- 只允许修改本 ticket 对应的 4 行 comparison ledger
