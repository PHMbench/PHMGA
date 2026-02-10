from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.deep_model_train_agent import _resolve_tspn_config
from src.utils.preflight import build_preflight_report


def test_preflight_detects_provider_model_mismatch():
    cfg = {
        "data": {"source_mode": "fixed_ids"},
        "metadata_path": __file__,
        "h5_path": __file__,
        "ref_ids": [1],
        "test_ids": [2],
    }
    report = build_preflight_report(
        cfg,
        env={
            "LLM_PROVIDER": "glm",
            "QUERY_GENERATOR_MODEL": "gemini-2.5-pro",
            "GLM_API_BASE": "https://open.bigmodel.cn/api/paas/v4",
            "GLM_API_KEY": "dummy",
        },
    )
    assert report["ok"] is False
    assert any("Gemini" in item for item in report["errors"])


def test_resolve_tspn_config_num_classes_fail_fast_and_autofit():
    source = str(Path("config") / "model_tspn_basic.yaml")

    with pytest.raises(ValueError, match="num_classes mismatch"):
        _resolve_tspn_config(
            source_model_config_path=source,
            inferred_in_dim=4096,
            inferred_in_channels=2,
            inferred_num_classes=3,
            autofit_dims=True,
            autofit_num_classes=False,
        )

    resolved, info = _resolve_tspn_config(
        source_model_config_path=source,
        inferred_in_dim=2048,
        inferred_in_channels=1,
        inferred_num_classes=3,
        autofit_dims=True,
        autofit_num_classes=True,
    )
    assert resolved.model.in_dim == 2048
    assert resolved.model.in_channels == 1
    assert resolved.model.num_classes == 3
    assert "overrides" in info
