import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_VIBENCH_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(not _enabled(), reason="Set PHM_ENABLE_VIBENCH_TESTS=1 to enable vibench end-to-end tests.")
def test_full_agent_flow_vibench_tspn_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    _ = torch  # keep lint quiet

    vibench_root = os.getenv("PHM_VIBENCH_CODE_ROOT", "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2")
    if not Path(vibench_root).exists():
        pytest.skip(f"Missing vibench code root: {vibench_root!r}")

    # Ensure required vibench deps exist in the current env.
    pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    pytest.importorskip("tqdm")
    pytest.importorskip("pandas")

    # Offline LLM + deterministic report
    monkeypatch.setenv("FAKE_LLM", "true")
    monkeypatch.setenv("PHM_REPORT_MODE", "template")

    from langchain_community.chat_models import FakeListChatModel

    from src import model as phm_model

    phm_model._FAKE_LLM = FakeListChatModel(
        responses=[
            # plan_agent: single band-pass filter on ch1
            '{"plan":[{"parent":"ch1","op_name":"filter","params":{"filter_type":"band","cutoff":[2000,4000]}}]}',
            # reflect_agent: stop after first execute
            '{"decision":"finish","reason":"ok"}',
        ]
    )

    # Prepare vibench minimal metadata (Dummy_Data reader can synthesize missing raw files).
    data_dir = tmp_path / "vibench_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = tmp_path / "metadata.csv"
    meta_path.write_text(
        "\n".join(
            [
                "Id,Name,File,Label",
                "1,Dummy_Data,dummy1.csv,0",
                "2,Dummy_Data,dummy2.csv,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    save_dir = tmp_path / "save"
    state_save_path = tmp_path / "state.pkl"
    report_path = tmp_path / "final_report.md"

    case_yaml = tmp_path / "case.yaml"
    case_yaml.write_text(
        "\n".join(
            [
                "name: e2e_vibench_dummy",
                "user_instruction: diagnose",
                f"state_save_path: {state_save_path}",
                f"report_path: {report_path}",
                "run_executor: true",
                "train_backend: tspn",
                f"save_dir: {save_dir}",
                "builder:",
                "  min_depth: 0",
                "  max_depth: 1",
                "data:",
                "  backend: vibench",
                f"  vibench_code_root: {vibench_root}",
                f"  data_dir: {data_dir}",
                f"  metadata_file: {meta_path}",
                "  dataset_name: Dummy_Data",
                "  task_type: DG",
                "  task_name: Classification",
                "  batch_size: 2",
                "  num_workers: 0",
                "  window_size: 128",
                "  stride: 16",
                "  num_window: 2",
                "  train_ratio: 0.8",
                "  normalization: standardization",
                "  fs_hz: 12000",
                "  debug: true",
                "  debug_epochs: 1",
                "  max_layers: 1",
                "  parallel_ops_per_layer: 2",
                "  out_channels: 2",
                "  scale: 2",
                "  fft_align_strategy: interp",
                "  features: [Mean, Std]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from src.cases.case1 import run_case

    run_case(str(case_yaml))

    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "PHMGA Diagnostic Report (Template)" in report
    assert "Val acc" in report
    assert "Artifacts" in report
