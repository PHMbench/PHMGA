from __future__ import annotations

from pathlib import Path

from src.evaluation import (
    REQUIRED_STAGE_B_ARTIFACTS,
    evaluate_artifact_contract,
    evaluate_feature_separability,
    evaluate_selection_eligibility,
)


def _valid_summary() -> dict[str, object]:
    return {
        "feature_count": 8,
        "non_empty_feature_count": 8,
        "constant_feature_count": 1,
        "top_features": [{"name": "band_power_low", "source_node": "agg_band_power_01", "score": 1.4}],
        "aggregate_scores": {"top5_mean_score": 0.9},
        "split_stability": {"train_val_rank_corr": 0.72},
        "decision": "pass",
    }


def test_artifact_contract_gate_passes_when_required_files_exist(tmp_path: Path):
    for artifact_name in REQUIRED_STAGE_B_ARTIFACTS:
        (tmp_path / artifact_name).write_text("{}", encoding="utf-8")

    assert evaluate_artifact_contract(tmp_path) is True


def test_artifact_contract_gate_fails_when_required_file_is_missing(tmp_path: Path):
    for artifact_name in REQUIRED_STAGE_B_ARTIFACTS:
        if artifact_name == "feature_separability_summary.json":
            continue
        (tmp_path / artifact_name).write_text("{}", encoding="utf-8")

    assert evaluate_artifact_contract(tmp_path) is False


def test_feature_separability_gate_passes_for_noncollapsed_summary():
    assert evaluate_feature_separability(_valid_summary()) is True


def test_feature_separability_gate_fails_for_constant_features():
    summary = _valid_summary()
    summary["constant_feature_count"] = summary["feature_count"]

    assert evaluate_feature_separability(summary) is False


def test_feature_separability_gate_fails_for_missing_top_features():
    summary = _valid_summary()
    summary["top_features"] = []

    assert evaluate_feature_separability(summary) is False


def test_feature_separability_gate_fails_for_missing_rank_correlation():
    summary = _valid_summary()
    summary["split_stability"] = {"train_val_rank_corr": None}

    assert evaluate_feature_separability(summary) is False


def test_selection_eligibility_requires_accept_and_both_gates():
    assert evaluate_selection_eligibility(
        keep="accept",
        artifact_contract_pass=True,
        feature_separability_pass=True,
    ) is True
    assert evaluate_selection_eligibility(
        keep="reject",
        artifact_contract_pass=True,
        feature_separability_pass=True,
    ) is False
    assert evaluate_selection_eligibility(
        keep="accept",
        artifact_contract_pass=False,
        feature_separability_pass=True,
    ) is False
