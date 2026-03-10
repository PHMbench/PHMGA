# 02 — Execution Playbook（Gemini-3 双主线可执行手册）

本手册按“先门禁、后双主线、最后判定”执行。默认环境：`conda run -n agent`。

---

## Phase A：先决门禁

## A1. LLM Gate（必须通过）

```bash
cd /home/user/LQ/B_Signal/PHMGA

conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/resolve_case_config.py \
  --base-config config/case_exp_gearbox_rm101.yaml \
  --out-config save/paper_matrix/2_24/_resolved_cases/gate_a_rm101_gemini3.yaml \
  --case-name gate_a_rm101_gemini3 \
  --save-root save/paper_matrix/2_24 \
  --provider openai_compatible \
  --model gemini-3-flash-preview \
  --ablation-mode full \
  --train-backend tspn

conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a2 \
  --config save/paper_matrix/2_24/_resolved_cases/gate_a_rm101_gemini3.yaml
```

通过标准：退出码 `0` 且输出 `GATE_A2_PASS`。

## A2. RM101 metadata 人工修复（你已选择）

- 手工修复 `gear_metadata.xlsx` 中异常 CSV（删除或替换）。  
- 将 `drop_list` 与修复说明记录到 `doc/plan/2_24/05_risk_register.md`。

## A3. 协议一致性检查（每个 run 前）

每个 resolved case 必须满足：
- `data.task_type=DG`
- `data.source_domain_id=[0..8]`
- `data.target_domain_id=[9,10,11]`
- `llm.provider=openai_compatible`
- `llm.query_generator_model=gemini-3-flash-preview`

---

## Phase B：Gemini-3 直接 ML 主线（`train_backend=shallow`）

执行 `A0_full` 的 4-seed：`0,42,123,999`。

```bash
cd /home/user/LQ/B_Signal/PHMGA
mkdir -p save/paper_matrix/2_24/_resolved_cases save/paper_matrix/2_24/_evidence

for seed in 0 42 123 999; do
  case_name="paper_rm101__m2_gemini3__shallow__seed${seed}__A0_full"
  cfg="save/paper_matrix/2_24/_resolved_cases/${case_name}.yaml"
  case_dir="save/paper_matrix/2_24/${case_name}"

  conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/resolve_case_config.py \
    --base-config config/case_exp_gearbox_rm101.yaml \
    --out-config "${cfg}" \
    --case-name "${case_name}" \
    --save-root save/paper_matrix/2_24 \
    --provider openai_compatible \
    --model gemini-3-flash-preview \
    --ablation-mode full \
    --train-backend shallow \
    --train-profile highacc \
    --allow-test-labels true

  CFG_PATH="${cfg}" SEED="${seed}" python - <<'PY'
import os, yaml, pathlib
p = pathlib.Path(os.environ["CFG_PATH"])
seed = int(os.environ["SEED"])
cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
d = cfg.setdefault("data", {})
d["task_type"] = "DG"
d["dataset_name"] = "RM_101_THU_GEARBOX"
d["source_domain_id"] = list(range(9))
d["target_domain_id"] = [9, 10, 11]
d["seed"] = seed
cfg["allow_test_labels_for_reporting"] = True
p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(p)
PY

  FAKE_LLM=false PHM_REPORT_MODE=template PHM_ABLATION_MODE=full \
  conda run -n agent -v PYTHONPATH=$(pwd) python main.py preflight --config "${cfg}"

  FAKE_LLM=false PHM_REPORT_MODE=template PHM_ABLATION_MODE=full \
  conda run --no-capture-output -n agent -v PYTHONPATH=$(pwd) python main.py case1 --config "${cfg}"

  conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/discover_run_artifacts.py \
    --case-dir "${case_dir}" > "save/paper_matrix/2_24/_evidence/${case_name}.artifacts.json"
done
```

当前实现注意事项（2026-02-24 实测）：
- `shallow` 路线可能出现“run 成功但无 `metrics.json` 落盘”的情况。  
- 若发生，统一标记 `evidence_incomplete`，并在 `05_risk_register.md` 登记，不可用于最终 Pass/Fail 判定。

---

## Phase C：Gemini-3 + TSPN 主线（`train_backend=tspn`）

执行 `A0_full` 的 4-seed：`0,42,123,999`。

```bash
cd /home/user/LQ/B_Signal/PHMGA

for seed in 0 42 123 999; do
  case_name="paper_rm101__m2_gemini3__tspn__seed${seed}__A0_full"
  cfg="save/paper_matrix/2_24/_resolved_cases/${case_name}.yaml"
  case_dir="save/paper_matrix/2_24/${case_name}"

  conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/resolve_case_config.py \
    --base-config config/case_exp_gearbox_rm101.yaml \
    --out-config "${cfg}" \
    --case-name "${case_name}" \
    --save-root save/paper_matrix/2_24 \
    --provider openai_compatible \
    --model gemini-3-flash-preview \
    --ablation-mode full \
    --train-backend tspn \
    --train-profile highacc \
    --allow-test-labels true

  CFG_PATH="${cfg}" SEED="${seed}" python - <<'PY'
import os, yaml, pathlib
p = pathlib.Path(os.environ["CFG_PATH"])
seed = int(os.environ["SEED"])
cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
d = cfg.setdefault("data", {})
d["task_type"] = "DG"
d["dataset_name"] = "RM_101_THU_GEARBOX"
d["source_domain_id"] = list(range(9))
d["target_domain_id"] = [9, 10, 11]
d["seed"] = seed
cfg["allow_test_labels_for_reporting"] = True
p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(p)
PY

  FAKE_LLM=false PHM_REPORT_MODE=template PHM_ABLATION_MODE=full \
  conda run -n agent -v PYTHONPATH=$(pwd) python main.py preflight --config "${cfg}"

  FAKE_LLM=false PHM_REPORT_MODE=template PHM_ABLATION_MODE=full \
  conda run --no-capture-output -n agent -v PYTHONPATH=$(pwd) python main.py case1 --config "${cfg}"

  conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/discover_run_artifacts.py \
    --case-dir "${case_dir}" > "save/paper_matrix/2_24/_evidence/${case_name}.artifacts.json"
done
```

---

## Phase D：4-seed 统计与 Pass/Fail 判定

```bash
cd /home/user/LQ/B_Signal/PHMGA

python - <<'PY'
import csv, json, pathlib, statistics

root = pathlib.Path("save/paper_matrix/2_24")
evidence = root / "_evidence"
rows = []
for backend in ("shallow", "tspn"):
    for seed in (0, 42, 123, 999):
        case_name = f"paper_rm101__m2_gemini3__{backend}__seed{seed}__A0_full"
        ap = evidence / f"{case_name}.artifacts.json"
        if not ap.exists():
            rows.append({"backend": backend, "seed": seed, "status": "missing_artifact"})
            continue
        art = json.loads(ap.read_text(encoding="utf-8"))
        mp = pathlib.Path(str(art.get("metrics_path") or ""))
        if not mp.exists():
            rows.append({"backend": backend, "seed": seed, "status": "missing_metrics"})
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        rows.append({
            "backend": backend,
            "seed": seed,
            "status": "ok",
            "test_acc": float(m.get("test_acc", 0.0)),
            "val_acc": float(m.get("val_acc", 0.0)),
            "test_macro_f1": float(m.get("test_macro_f1", 0.0)),
        })

out_csv = root / "gemini3_dualtrack_seed_results.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["backend", "seed", "status", "test_acc", "val_acc", "test_macro_f1"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

summary = {}
for backend in ("shallow", "tspn"):
    ok = [r for r in rows if r.get("backend") == backend and r.get("status") == "ok"]
    vals = [float(r["test_acc"]) for r in ok]
    if vals:
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[backend] = {"n_ok": len(vals), "mean_test_acc": mean, "std_test_acc_ddof0": std}
    else:
        summary[backend] = {"n_ok": 0, "mean_test_acc": None, "std_test_acc_ddof0": None}

threshold = 0.7975
judge = {
    "threshold_test_acc": threshold,
    "tracks": summary,
    "pass": any((v["mean_test_acc"] or 0.0) > threshold for v in summary.values()),
}
out_json = root / "gemini3_dualtrack_judgement.json"
out_json.write_text(json.dumps(judge, indent=2, ensure_ascii=False), encoding="utf-8")
print(out_csv)
print(out_json)
PY
```

通过标准：
- 两条主线都至少有 `n_ok=4`（若有失败，需在风险表记录阻断证据）。  
- 生成 `gemini3_dualtrack_seed_results.csv` 与 `gemini3_dualtrack_judgement.json`。  

---

## 可选 Phase E：补 `A1/A2`（仅用于机制解释）

- 条件：`A0_full` 跑完后再执行。  
- 目的：写作消融机制解释，不影响“是否超过 79.75%”主判定。  
