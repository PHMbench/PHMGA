from __future__ import annotations

from pathlib import Path

from src.utils.preflight import build_preflight_report


def _make_vibench_cfg(tmp_path: Path, *, max_depth: int = 8):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw" / "RM_101_THU_GEARBOX"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = data_dir / "gear_metadata.csv"
    metadata.write_text("Id,Name,File,Label\n1,RM_101_THU_GEARBOX,a.csv,0\n", encoding="utf-8")
    code_root = tmp_path / "vibench"
    code_root.mkdir(parents=True, exist_ok=True)
    return {
        "data": {
            "backend": "vibench",
            "source_mode": "vibench",
            "data_dir": str(data_dir),
            "metadata_file": "gear_metadata.csv",
            "dataset_name": "RM_101_THU_GEARBOX",
            "vibench_code_root": str(code_root),
        },
        "builder": {"max_depth": max_depth},
    }


def test_preflight_warns_for_shallow_builder_on_rm101(tmp_path):
    cfg = _make_vibench_cfg(tmp_path, max_depth=2)
    report = build_preflight_report(cfg, env={"FAKE_LLM": "true"})
    assert report["ok"] is True
    assert any("recommended max_depth >= 6" in item for item in report["warnings"])


def test_preflight_warns_when_antropy_missing(tmp_path, monkeypatch):
    cfg = _make_vibench_cfg(tmp_path, max_depth=8)

    def _fake_dep(name: str) -> bool:
        if name == "antropy":
            return False
        return True

    monkeypatch.setattr("src.utils.preflight._dependency_status", _fake_dep)
    report = build_preflight_report(cfg, env={"FAKE_LLM": "true"})
    assert any("antropy" in item for item in report["warnings"])


def test_preflight_no_stft_unregistered_warning_after_resilient_import(tmp_path):
    cfg = _make_vibench_cfg(tmp_path, max_depth=8)
    report = build_preflight_report(cfg, env={"FAKE_LLM": "true"})
    warning_text = "\n".join(report.get("warnings", []))
    assert "Unregistered operators: stft" not in warning_text


def test_preflight_can_block_on_missing_dependencies(tmp_path, monkeypatch):
    cfg = _make_vibench_cfg(tmp_path, max_depth=8)
    cfg["preflight"] = {"block_on_missing_dependencies": ["librosa"]}

    def _fake_dep(name: str) -> bool:
        if name == "librosa":
            return False
        return True

    monkeypatch.setattr("src.utils.preflight._dependency_status", _fake_dep)
    report = build_preflight_report(cfg, env={"FAKE_LLM": "true"})
    assert report["ok"] is False
    assert any("Missing required dependency" in item for item in report["errors"])
