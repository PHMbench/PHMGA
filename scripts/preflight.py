"""Library preflight checks invoked by the root Hydra entrypoint."""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict

from src.data import build_protocol_from_config
from src.operators import get_operator_catalog


def run_preflight(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one resolved config against the shared runtime contract."""
    protocol = build_protocol_from_config(runtime_config)
    catalog = get_operator_catalog()
    llm_cfg = dict(runtime_config.get("llm", {}))
    provider = str(llm_cfg.get("provider", "codex_cli"))
    mode = str(llm_cfg.get("mode", "offline_stub"))
    llm_transport = {
        "provider": provider,
        "mode": mode,
        "binary_found": True,
    }
    if mode == "provider" and provider == "codex_cli":
        binary_path = shutil.which("codex")
        if not binary_path:
            raise RuntimeError("Codex CLI preflight failed: required `codex` binary was not found on PATH.")
        llm_transport = {
            "provider": provider,
            "mode": mode,
            "binary_found": True,
            "binary_name": "codex",
            "binary_path": binary_path,
        }
    elif mode == "provider" and provider == "openrouter":
        api_key_env = str(llm_cfg.get("api_key_env") or "OPENROUTER_API_KEY")
        credential_found = bool(os.getenv(api_key_env, "").strip())
        if not credential_found:
            raise RuntimeError(
                f"OpenRouter preflight failed: expected non-empty credential in env var {api_key_env}."
            )
        llm_transport = {
            "provider": provider,
            "mode": mode,
            "credential_found": credential_found,
            "api_key_env": api_key_env,
        }
    return {
        "status": "ok",
        "config_name": runtime_config["runtime"]["config_name"],
        "dataset_name": protocol.dataset_name,
        "graph_path": runtime_config["experiment"]["graph_path"],
        "source_mode": protocol.source_mode,
        "sample_count": len(protocol.samples),
        "splits": {
            "train": len(protocol.splits.train_ids),
            "val": len(protocol.splits.val_ids),
            "test": len(protocol.splits.test_ids),
        },
        "window": protocol.window.model_dump(),
        "selected_channels": protocol.selected_channels,
        "operators": [spec.op_uid for spec in catalog.specs()],
        "llm_transport": llm_transport,
    }
