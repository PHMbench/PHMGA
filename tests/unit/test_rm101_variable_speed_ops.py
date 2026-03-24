from __future__ import annotations

import numpy as np

from src.tools import get_operator


def _sample_inputs(length: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 1.0, length, endpoint=False)
    vibration = (
        np.sin(2.0 * np.pi * 36.0 * time)
        + 0.2 * np.sin(2.0 * np.pi * 72.0 * time)
    ).reshape(1, -1, 1)
    key_phase = np.zeros((length,), dtype=float)
    key_phase[::256] = 1.0
    speed = key_phase.reshape(1, -1, 1)
    torque = (1.0 + 0.2 * np.sin(2.0 * np.pi * 1.0 * time)).reshape(1, -1, 1)
    return vibration, speed, torque


def test_order_track_resample_outputs_fixed_angle_grid():
    vibration, speed, _ = _sample_inputs()
    op = get_operator("order_track_resample")(points_per_rev=128, revolutions=4, parent=["ch6", "ch1"])
    out = op.execute({"ch6": vibration, "ch1": speed})
    assert out.shape == (1, 512, 1)


def test_tsa_cycle_average_outputs_cycle_average():
    vibration, speed, _ = _sample_inputs()
    op = get_operator("tsa_cycle_average")(points_per_rev=128, revolutions=4, parent=["ch6", "ch1"])
    out = op.execute({"ch6": vibration, "ch1": speed})
    assert out.shape == (1, 128, 1)


def test_order_band_energy_and_sideband_ratio_flatten_to_feature_vectors():
    vibration, speed, _ = _sample_inputs()
    order_track = get_operator("order_track_resample")(points_per_rev=128, revolutions=4, parent=["ch6", "ch1"])
    order_signal = order_track.execute({"ch6": vibration, "ch1": speed})

    band_energy = get_operator("order_band_energy")(points_per_rev=128, parent="order_track")
    sideband = get_operator("sideband_ratio")(points_per_rev=128, parent="order_track")

    band_features = band_energy.execute(order_signal)
    sideband_features = sideband.execute(order_signal)
    assert band_features.ndim == 2
    assert band_features.shape[0] == 1
    assert band_features.shape[1] > 0
    assert sideband_features.shape == (1, 1)


def test_torque_normalize_preserves_feature_shape():
    _, _, torque = _sample_inputs()
    feature = np.asarray([[2.0, 4.0, 6.0]], dtype=float)
    op = get_operator("torque_normalize")(mode="abs_mean", parent=["feature", "ch2"])
    out = op.execute({"feature": feature, "ch2": torque})
    assert out.shape == feature.shape
    assert np.isfinite(out).all()
