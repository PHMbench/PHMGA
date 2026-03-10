# PHMGA v2.18 环境锁定与联通检查（GLM-4.7）

本文件定义执行矩阵前必须通过的阻断检查。  
默认运行环境：`conda run -n agent`。

## 1) 基础依赖

```bash
conda run -n agent python -V
conda run -n agent pip install -U zai librosa nolds antropy graphviz
```

系统依赖：
- 需要 `dot` 二进制（Linux: `sudo apt-get install graphviz`）。

阻断检查：
```bash
conda run -n agent python - <<'PY'
import importlib, shutil, sys
required = ["zai", "librosa", "nolds", "antropy", "graphviz"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
dot_ok = shutil.which("dot") is not None
print({"missing_packages": missing, "graphviz_dot": dot_ok})
sys.exit(0 if not missing and dot_ok else 1)
PY
```

## 2) Active 模型映射（3 模型）

| Tag | Provider | Model |
| --- | --- | --- |
| M1 | `openai_compatible` | `gemini-2.5-flash` |
| M2 | `openai_compatible` | `gemini-3-flash-preview` |
| M3 | `glm` | `GLM-4.7-Flash` |

## 3) `.env` 前置检查（仅 key/base）

`provider/model` 由 `case.yaml -> llm` 决定；`.env` 仅承载 `*_API_KEY`、`*_BASE/*_BASE_URL`。  
检查命令（不打印密钥）：

```bash
conda run -n agent python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=str(Path.cwd() / ".env"), override=False)
keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "GLM_API_KEY", "GLM_API_BASE"]
print({k: bool(os.getenv(k)) for k in keys})
PY
```

## 4) Gate-A1：`zai` SDK 直连 GLM-4.7

```bash
conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a1 \
  --provider glm \
  --model glm-4.7-flash
```

通过标准：
- 退出码 `0`
- 输出包含 `GATE_A1_PASS`

## 5) Gate-A2：PHMGA 路由检查（case 驱动）

先生成带 `llm` 的 resolved case，再走 PHMGA `get_llm()`：

```bash
conda run -n agent python scripts/paper/resolve_case_config.py \
  --base-config config/case_exp_gearbox_rm101.yaml \
  --out-config save/paper_matrix/m3_glm47/_resolved_cases/gate_a_rm101_m3.yaml \
  --case-name gate_a_rm101_m3 \
  --save-root save/paper_matrix/m3_glm47 \
  --provider glm \
  --model GLM-4.7-Flash \
  --ablation-mode full \
  --train-backend tspn

conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a2 \
  --config save/paper_matrix/m3_glm47/_resolved_cases/gate_a_rm101_m3.yaml
```

## 6) Gate-B：单组合 pilot

```bash
scripts/paper/run_combo.sh \
  --llm-tag m3_glm47 \
  --provider glm \
  --model GLM-4.7-Flash \
  --dataset-tag ottawa \
  --case-config config/tspn_case_exp_ottawa.yaml \
  --ablation-tag A0_full \
  --ablation-mode full \
  --output-root save/paper_matrix/m3_glm47 \
  --env agent
```

通过标准：
- `manifest.jsonl` 新增 1 条
- 新增记录 `status=ok`

## 7) 失败分流

1. `403 model access`：先核对账号权限，再核对 case `llm` 与 key/base 平台一致性。  
2. `401/invalid key`：仅修 `.env` key/base，不改 case 模型定义。  
3. `rc=137`：先用 `fast profile` 验证链路，再恢复标准配置。  
