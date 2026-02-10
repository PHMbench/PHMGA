from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _exists(path_value: str | None) -> bool:
    if not path_value:
        return False
    return Path(path_value).expanduser().exists()


def _provider_model_check(env: Dict[str, str]) -> Dict[str, Any]:
    provider = (env.get("LLM_PROVIDER") or "gemini").strip().lower()
    model = (env.get("QUERY_GENERATOR_MODEL") or env.get("PHM_MODEL") or "").strip()
    model_lc = model.lower()
    problems: List[str] = []
    warnings: List[str] = []

    if provider == "glm":
        if model_lc.startswith("gemini"):
            problems.append("LLM_PROVIDER=glm but model looks like a Gemini model.")
        if not env.get("GLM_API_BASE"):
            problems.append("Missing GLM_API_BASE.")
        if not env.get("GLM_API_KEY"):
            problems.append("Missing GLM_API_KEY.")
    elif provider == "gemini":
        if model_lc.startswith("glm") or "deepseek" in model_lc:
            problems.append("LLM_PROVIDER=gemini but model looks OpenAI-compatible.")
        if not env.get("GEMINI_API_KEY"):
            warnings.append("GEMINI_API_KEY is not set.")
    elif provider in {"deepseek", "openai", "openai_compatible"}:
        if not (env.get("OPENAI_API_KEY") or env.get("DEEPSEEK_API_KEY") or env.get("GLM_API_KEY")):
            problems.append("OpenAI-compatible provider selected but no API key found.")
    elif provider == "auto":
        warnings.append("LLM_PROVIDER=auto may route unexpectedly when multiple BASE URLs are set.")
    else:
        problems.append(f"Unsupported LLM_PROVIDER={provider!r}.")

    return {
        "provider": provider,
        "model": model,
        "ok": not problems,
        "errors": problems,
        "warnings": warnings,
    }


def _dependency_status(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False


def _operator_checks(required_ops: List[str]) -> Dict[str, Any]:
    from src.tools.signal_processing_schemas import get_operator

    present: List[str] = []
    missing: List[str] = []
    for op_name in required_ops:
        try:
            get_operator(op_name)
            present.append(op_name)
        except Exception:
            missing.append(op_name)
    return {"required_ops": required_ops, "registered": present, "missing": missing}


def _resolve_case_data_mode(config: Dict[str, Any]) -> str:
    data = dict(config.get("data") or {})
    source_mode = str(data.get("source_mode") or "").strip().lower()
    if source_mode in {"fixed_ids", "vibench"}:
        return source_mode
    backend = str(data.get("backend") or "").strip().lower()
    if backend == "vibench":
        return "vibench"
    return "fixed_ids"


def build_preflight_report(config: Dict[str, Any], *, env: Dict[str, str] | None = None) -> Dict[str, Any]:
    env_map = dict(os.environ)
    if env:
        env_map.update(env)

    source_mode = _resolve_case_data_mode(config)
    data = dict(config.get("data") or {})
    checks: Dict[str, Any] = {"source_mode": source_mode}
    errors: List[str] = []
    warnings: List[str] = []

    if source_mode == "fixed_ids":
        metadata_path = str(config.get("metadata_path") or data.get("metadata_path") or "")
        h5_path = str(config.get("h5_path") or data.get("h5_path") or "")
        ref_ids = list(config.get("ref_ids") or data.get("ref_ids") or [])
        test_ids = list(config.get("test_ids") or data.get("test_ids") or [])
        checks["data"] = {
            "metadata_path": metadata_path,
            "h5_path": h5_path,
            "n_ref_ids": len(ref_ids),
            "n_test_ids": len(test_ids),
            "metadata_exists": _exists(metadata_path),
            "h5_exists": _exists(h5_path),
        }
        if not checks["data"]["metadata_exists"]:
            errors.append(f"metadata_path not found: {metadata_path}")
        if not checks["data"]["h5_exists"]:
            errors.append(f"h5_path not found: {h5_path}")
        if len(ref_ids) == 0:
            errors.append("ref_ids is empty.")
        if len(test_ids) == 0:
            warnings.append("test_ids is empty; n_test will be 0 unless reporting test labels is enabled.")
    else:
        vibench_root = str(data.get("vibench_code_root") or env_map.get("PHM_VIBENCH_CODE_ROOT") or "")
        metadata_file = str(data.get("metadata_file") or "")
        data_dir = str(data.get("data_dir") or "")
        dataset_name = str(data.get("dataset_name") or "")
        checks["data"] = {
            "vibench_code_root": vibench_root,
            "metadata_file": metadata_file,
            "data_dir": data_dir,
            "dataset_name": dataset_name,
            "vibench_code_root_exists": _exists(vibench_root),
            "data_dir_exists": _exists(data_dir),
            "metadata_exists": _exists(metadata_file if Path(metadata_file).is_absolute() else str(Path(data_dir) / metadata_file)),
        }
        if not checks["data"]["vibench_code_root_exists"]:
            errors.append(f"vibench_code_root not found: {vibench_root}")
        if not checks["data"]["data_dir_exists"]:
            errors.append(f"data_dir not found: {data_dir}")
        if not checks["data"]["metadata_exists"]:
            errors.append(f"metadata_file not found: {metadata_file}")
        if not dataset_name:
            warnings.append("dataset_name is empty.")

    provider_check = _provider_model_check(env_map)
    checks["llm"] = provider_check
    errors.extend(provider_check["errors"])
    warnings.extend(provider_check["warnings"])

    required_ops = list(((config.get("preflight") or {}).get("required_ops") or []))
    if not required_ops:
        required_ops = ["mean", "fft", "filter", "hilbert_envelope", "band_power", "spectral_entropy", "stft", "approximate_entropy"]
    op_checks = _operator_checks(required_ops)
    checks["operators"] = op_checks
    if op_checks["missing"]:
        warnings.append(f"Unregistered operators: {', '.join(op_checks['missing'])}")

    deps = {
        "nolds": _dependency_status("nolds"),
        "graphviz_python": _dependency_status("graphviz"),
    }
    checks["dependencies"] = deps
    if not deps["nolds"]:
        warnings.append("Optional dependency 'nolds' is missing; approximate entropy features will fail.")
    if not deps["graphviz_python"]:
        warnings.append("Optional dependency 'graphviz' python package is missing; PNG graph export will fallback to DOT.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def run_preflight_from_config_path(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return build_preflight_report(config)


def write_preflight_report(report: Dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

