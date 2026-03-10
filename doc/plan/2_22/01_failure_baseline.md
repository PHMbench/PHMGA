# 2.22 失败基线快照（latest manifest 口径）

快照时间：2026-02-23（本地扫描结果）  
统计规则：每个 `combo` 取 `manifest.jsonl` 最后一条记录。

---

## A. Active Matrix（3 模型）

### `m1_gemini25`

- 状态：`4/6` 成功
- 成功：
  - `ottawa__m1_gemini25__A0_full`
  - `ottawa__m1_gemini25__A1_no_reflect`
  - `ottawa__m1_gemini25__A2_no_prior`
  - `rm101__m1_gemini25__A0_full`
- 失败：
  - `rm101__m1_gemini25__A1_no_reflect` (`rc=2`, `S3: out_total % num_ops`)
  - `rm101__m1_gemini25__A2_no_prior` (`rc=2`, `S3: out_total % num_ops`)

证据路径（示例）：
- `save/paper_matrix/m1_gemini25/_logs/rm101__m1_gemini25__A0_full/run.log`（已不再报 S3）
- `save/paper_matrix/m1_gemini25/_logs/rm101__m1_gemini25__A1_no_reflect/run.log`
- `save/paper_matrix/m1_gemini25/_logs/rm101__m1_gemini25__A2_no_prior/run.log`

### `m2_gemini3`

- 状态：`6/6` 成功
- 说明：当前 active 子集里唯一跑满矩阵的模型目录。

证据路径（示例）：
- `save/paper_matrix/m2_gemini3/manifest.jsonl`

### `m3_glm47`

- 状态：`manifest.jsonl` 当前不存在（尚未形成 active run）。
- 说明：需先完成 Gate-A1/A2 与 pilot。

---

## B. Legacy（归档，不参与 active 执行）

Legacy 详情从本文件移出，统一收敛到：

- `doc/plan/2_22/legacy_glm45_archive.md`

该文档记录：
- `m3_glm45`（GLM-4.5 历史目录）
- `m4_glm47`（旧标签历史目录）

---

## C. 失败证据路径模板（统一）

每个失败组合应至少保存：

- `run_log`: `save/paper_matrix/<llm_tag>/_logs/<combo>/run.log`
- `preflight_log`: `save/paper_matrix/<llm_tag>/_logs/<combo>/preflight.log`

建议附加：
- `resolved_case`: `save/paper_matrix/<llm_tag>/_resolved_cases/<combo>.yaml`

---

## D. 基线重建命令（只读）

```bash
python - <<'PY'
import json
from pathlib import Path

roots = [
    Path("save/paper_matrix/m1_gemini25"),
    Path("save/paper_matrix/m2_gemini3"),
    Path("save/paper_matrix/m3_glm47"),
]
for root in roots:
    m = root / "manifest.jsonl"
    print("\\n===", root)
    if not m.exists():
        print("manifest: MISSING")
        continue
    latest = {}
    for line in m.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            combo = row.get("combo")
            if combo:
                latest[combo] = row
    ok = sum(1 for r in latest.values() if r.get("status") == "ok")
    print("latest=", len(latest), "ok=", ok, "fail=", len(latest) - ok)
PY
```
