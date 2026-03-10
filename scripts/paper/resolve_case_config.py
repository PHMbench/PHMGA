#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a PHM case config for matrix execution.")
    parser.add_argument("--base-config", required=True, help="Base case yaml path.")
    parser.add_argument("--out-config", required=True, help="Output resolved yaml path.")
    parser.add_argument("--case-name", required=True, help="Resolved case name.")
    parser.add_argument("--save-root", required=True, help="Root directory for case outputs.")
    parser.add_argument("--ablation-mode", required=True, help="Ablation mode: full/no_reflect/no_prior.")
    parser.add_argument("--train-backend", default="tspn", help="train_backend value.")
    parser.add_argument(
        "--provider",
        default="",
        help="Optional LLM provider to write into case llm block.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional LLM model to write into case llm block.",
    )
    parser.add_argument(
        "--train-profile",
        default="",
        help="Optional train profile: fast/standard/highacc.",
    )
    parser.add_argument(
        "--compat-profile",
        default="",
        help="Optional compatibility profile (e.g. rm101_strict).",
    )
    parser.add_argument(
        "--operator-contract",
        default="rm101_closed_v1",
        help="Closed-world operator contract name (default: rm101_closed_v1).",
    )
    parser.add_argument(
        "--closed-world",
        default="true",
        help="Whether enforce_tspn_closed_world is enabled.",
    )
    parser.add_argument(
        "--allow-test-labels",
        default="false",
        help="Whether allow_test_labels_for_reporting is enabled.",
    )
    parser.add_argument(
        "--state-save-mode",
        default="auto",
        help="State save mode: auto/full/minimal.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config).resolve()
    out_path = Path(args.out_config).resolve()
    save_root = Path(args.save_root).resolve()
    if not base_path.exists():
        raise SystemExit(f"Base config not found: {base_path}")

    cfg = _load_yaml(base_path)
    case_name = str(args.case_name).strip()
    if not case_name:
        raise SystemExit("--case-name cannot be empty")

    case_dir = save_root / case_name
    cfg["name"] = case_name
    cfg["save_dir"] = str(save_root)
    cfg["state_save_path"] = str(case_dir / "built_state.pkl")
    cfg["report_path"] = str(case_dir / "final_report.md")
    cfg["run_executor"] = True
    cfg["train_backend"] = str(args.train_backend).strip().lower() or "tspn"
    cfg["allow_test_labels_for_reporting"] = _parse_bool(args.allow_test_labels)

    provider = str(args.provider or "").strip().lower()
    model = str(args.model or "").strip()
    if provider or model:
        if not provider or not model:
            raise SystemExit("--provider and --model must be set together.")
        cfg["llm"] = {
            "provider": provider,
            "query_generator_model": model,
            "phm_model": model,
            "reflection_model": model,
            "answer_model": model,
        }

    ablation = dict(cfg.get("ablation") or {})
    ablation["mode"] = str(args.ablation_mode).strip().lower() or "full"
    cfg["ablation"] = ablation

    train_profile = str(args.train_profile or "").strip().lower()
    compat_profile = str(args.compat_profile or "").strip().lower()
    operator_contract = str(args.operator_contract or "rm101_closed_v1").strip().lower() or "rm101_closed_v1"
    enforce_closed_world = _parse_bool(args.closed_world)
    state_save_mode = str(args.state_save_mode or "auto").strip().lower() or "auto"
    if state_save_mode not in {"auto", "full", "minimal"}:
        raise SystemExit(f"Invalid --state-save-mode={args.state_save_mode!r}. Expected one of: auto/full/minimal.")
    data_cfg = dict(cfg.get("data") or {})
    data_cfg["operator_contract"] = operator_contract
    data_cfg["enforce_tspn_closed_world"] = bool(enforce_closed_world)
    data_cfg["state_save_mode"] = state_save_mode
    if compat_profile:
        data_cfg["compat_profile"] = compat_profile
    cfg["data"] = data_cfg

    if train_profile:
        data_cfg = dict(cfg.get("data") or {})
        data_cfg["train_profile"] = train_profile
        profile_defaults = {
            "fast": {"epochs": 20, "patience": 6, "lr": 1e-3, "scheduler": "none", "use_weighted_sampler": False, "label_smoothing": 0.0},
            "standard": {"epochs": 40, "patience": 10, "lr": 5e-4, "scheduler": "plateau", "use_weighted_sampler": False, "label_smoothing": 0.05},
            "highacc": {"epochs": 60, "patience": 15, "lr": 3e-4, "scheduler": "cosine", "use_weighted_sampler": True, "label_smoothing": 0.05},
        }
        defaults = profile_defaults.get(train_profile, {})
        for key, value in defaults.items():
            data_cfg.setdefault(key, value)
        cfg["data"] = data_cfg

    _dump_yaml(out_path, cfg)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
