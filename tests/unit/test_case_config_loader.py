from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from main import main
from src.config import load_case_config, resolve_case_path


def test_load_case_config_defaults_builder_graph_to_with_report(make_case_config):
    case = make_case_config(case_name="case_default_graph", graph="with_report")
    config_path = Path(case["config_path"])
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["builder"].pop("graph")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_case_config(case["case_name"], config_root=case["config_root"])

    assert config["builder"]["graph"] == "with_report"
    assert config["builder"]["min_width"] == 2


def test_resolve_case_path_raises_for_unknown_case(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_case_path("does_not_exist", config_root=tmp_path)


def test_main_returns_nonzero_for_unknown_case():
    assert main(["does_not_exist"]) == 1


def test_case_exp2_includes_rm101_protocol_block():
    config_path = Path("config/case_exp2.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_cfg = payload["data"]
    assert data_cfg["selection"]["dataset_id"] == 101
    assert data_cfg["selection"]["domain_ids"] == [0, 1, 2]
    assert data_cfg["split"]["strategy"] == "stratified_fixed_per_class"
    assert data_cfg["window"]["slice_mode"] == "centered"


def test_case_exp2_paper_uses_ratio_split():
    config_path = Path("config/case_exp2_paper.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_cfg = payload["data"]
    assert data_cfg["selection"]["dataset_id"] == 101
    assert data_cfg["selection"]["domain_ids"] == [0, 1, 2]
    assert data_cfg["split"]["strategy"] == "stratified_ratio"
    assert data_cfg["split"]["train_ratio"] == 0.6
    assert data_cfg["split"]["val_ratio"] == 0.2
    assert data_cfg["split"]["test_ratio"] == 0.2
    assert data_cfg["window"]["slice_mode"] == "sliding"


def test_case_exp2_all_domains_mixed_paper_uses_all_domains_ratio_split():
    config_path = Path("config/case_exp2_all_domains_mixed_paper.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_cfg = payload["data"]
    assert data_cfg["selection"]["dataset_id"] == 101
    assert data_cfg["selection"]["domain_ids"] == list(range(12))
    assert data_cfg["split"]["strategy"] == "stratified_ratio"
    assert data_cfg["split"]["train_ratio"] == 0.6
    assert data_cfg["split"]["val_ratio"] == 0.2
    assert data_cfg["split"]["test_ratio"] == 0.2
    assert data_cfg["window"]["window_size"] == 4096
    assert data_cfg["window"]["stride"] == 4096
    assert data_cfg["window"]["slice_mode"] == "sliding"
