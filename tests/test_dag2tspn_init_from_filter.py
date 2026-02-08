import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_TORCH_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(not _enabled(), reason="Set PHM_ENABLE_TORCH_TESTS=1 to enable torch init-from-DAG tests.")
def test_dag2tspn_wavefilter_init_from_bandpass():
    torch = pytest.importorskip("torch")

    from src.model.explainable.bridge import DAG2ConfigAdapter
    from src.model.explainable.builder import build_tspn_from_config
    from src.model.explainable.config_schema import TSPNConfig
    from src.model.explainable.ops import WaveFilters
    from src.states.phm_states import DAGState, InputData, ProcessedData

    fs_hz = 12000.0
    L = 128

    ch1 = InputData(node_id="ch1", data={}, results={}, parents=[], shape=(1, L, 1), meta={"fs": fs_hz})
    bp = ProcessedData(
        node_id="bp1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="filter",
        results={},
        meta={"params": {"filter_type": "band", "cutoff": [2000.0, 4000.0]}},
        shape=(1, L, 1),
    )

    dag = DAGState(
        user_instruction="test",
        channels=["ch1"],
        nodes={"ch1": ch1, "bp1": bp},
        leaves=["bp1"],
    )

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=1,
        num_classes=2,
        fs_hz=fs_hz,
        max_layers=1,
        parallel_ops_per_layer=2,
        out_channels=2,
        scale=2,
        feature_tokens=["Mean", "Std"],
        fft_align_strategy="interp",
    )
    bridge = adapter.adapt(dag)

    assert "wf_by_op_uid" in bridge.init_metadata
    assert "L1:WF:0" in bridge.init_metadata["wf_by_op_uid"]

    cfg = TSPNConfig.model_validate(bridge.model_config)
    model, _ = build_tspn_from_config(cfg, device="cpu")
    model.init_weights_from_metadata(bridge.init_metadata)

    wf_module = None
    for layer in model.signal_layers:
        for m in layer.modules_dict.values():
            if isinstance(m, WaveFilters):
                wf_module = m
                break
    assert wf_module is not None

    fc_hz = float(wf_module.fc_norm().mean().detach().cpu().item()) * fs_hz
    fb_hz = float(wf_module.fb_norm().mean().detach().cpu().item()) * fs_hz

    assert abs(fc_hz - 3000.0) <= 200.0
    assert abs(fb_hz - 1000.0) <= 200.0
