from __future__ import annotations

from typing import ClassVar, Dict, List, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy import signal

from .signal_processing_schemas import AggregateOp, MultiVariableOp, register_op


def _rising_edges(key_signal: np.ndarray) -> np.ndarray:
    key_1d = np.asarray(key_signal, dtype=float).reshape(-1)
    if key_1d.size < 8:
        return np.asarray([], dtype=int)
    threshold = float((np.max(key_1d) + np.min(key_1d)) / 2.0)
    binary = key_1d > threshold
    edges = np.flatnonzero((~binary[:-1]) & binary[1:]) + 1
    if edges.size >= 2:
        return edges

    prominence = max(float(np.std(key_1d)) * 0.25, 1e-6)
    min_distance = max(int(key_1d.size // 64), 8)
    peaks, _ = signal.find_peaks(key_1d, prominence=prominence, distance=min_distance)
    return np.asarray(peaks, dtype=int)


def _resample_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros((target_length,), dtype=float)
    if values.size == target_length:
        return values.copy()
    if values.size == 1:
        return np.full((target_length,), float(values[0]), dtype=float)
    old_axis = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    new_axis = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(new_axis, old_axis, values).astype(float, copy=False)


def _fixed_cycle_bank(
    vibration: np.ndarray,
    key_signal: np.ndarray,
    *,
    points_per_rev: int,
    revolutions: int,
) -> np.ndarray:
    vib_1d = np.asarray(vibration, dtype=float).reshape(-1)
    edges = _rising_edges(key_signal)
    cycles: List[np.ndarray] = []
    if edges.size >= 2:
        for start, end in zip(edges[:-1], edges[1:]):
            if int(end) - int(start) < 8:
                continue
            cycles.append(_resample_1d(vib_1d[int(start):int(end)], points_per_rev))
    if not cycles:
        fallback = _resample_1d(vib_1d, max(points_per_rev * revolutions, points_per_rev))
        return fallback.reshape(revolutions, points_per_rev)

    if len(cycles) < revolutions:
        original = list(cycles)
        while len(cycles) < revolutions:
            cycles.append(original[len(cycles) % len(original)])
    return np.stack(cycles[:revolutions], axis=0)


def _spectrum_from_signal(x: np.ndarray, *, points_per_rev: int) -> tuple[np.ndarray, np.ndarray]:
    signal_1d = np.asarray(x, dtype=float).reshape(-1)
    spectrum = np.abs(np.fft.rfft(signal_1d))
    orders = np.fft.rfftfreq(signal_1d.size, d=1.0 / float(points_per_rev))
    return orders, spectrum


@register_op
class OrderTrackResampleOp(MultiVariableOp):
    op_name: ClassVar[str] = "order_track_resample"
    description: ClassVar[str] = "Resamples vibration to a fixed angular grid using the speed/key-phase side-input."
    input_spec: ClassVar[str] = "vibration: (B, L, C), speed_key_phase: (B, L, 1)"
    output_spec: ClassVar[str] = "(B, revolutions * points_per_rev, C)"

    points_per_rev: int = Field(256, description="Samples per revolution in the angle domain.")
    revolutions: int = Field(4, description="Number of revolutions to keep in each output sample.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"OrderTrackResampleOp requires 2 inputs, got {len(x)}.")
        vibration, speed_key = list(x.values())
        if vibration.ndim != 3 or speed_key.ndim != 3:
            raise ValueError("OrderTrackResampleOp expects 3D inputs shaped (B, L, C).")

        batch_size, _, channels = vibration.shape
        outputs = np.zeros((batch_size, self.points_per_rev * self.revolutions, channels), dtype=float)
        for batch_index in range(batch_size):
            key_1d = speed_key[batch_index, :, 0]
            for channel_index in range(channels):
                bank = _fixed_cycle_bank(
                    vibration[batch_index, :, channel_index],
                    key_1d,
                    points_per_rev=self.points_per_rev,
                    revolutions=self.revolutions,
                )
                outputs[batch_index, :, channel_index] = bank.reshape(-1)
        return outputs


@register_op
class TSACycleAverageOp(MultiVariableOp):
    op_name: ClassVar[str] = "tsa_cycle_average"
    description: ClassVar[str] = "Computes time synchronous averaging (TSA) using the speed/key-phase side-input."
    input_spec: ClassVar[str] = "vibration: (B, L, C), speed_key_phase: (B, L, 1)"
    output_spec: ClassVar[str] = "(B, points_per_rev, C)"

    points_per_rev: int = Field(256, description="Samples per revolution after synchronization.")
    revolutions: int = Field(4, description="Maximum number of revolutions to average.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"TSACycleAverageOp requires 2 inputs, got {len(x)}.")
        vibration, speed_key = list(x.values())
        if vibration.ndim != 3 or speed_key.ndim != 3:
            raise ValueError("TSACycleAverageOp expects 3D inputs shaped (B, L, C).")

        batch_size, _, channels = vibration.shape
        outputs = np.zeros((batch_size, self.points_per_rev, channels), dtype=float)
        for batch_index in range(batch_size):
            key_1d = speed_key[batch_index, :, 0]
            for channel_index in range(channels):
                bank = _fixed_cycle_bank(
                    vibration[batch_index, :, channel_index],
                    key_1d,
                    points_per_rev=self.points_per_rev,
                    revolutions=self.revolutions,
                )
                outputs[batch_index, :, channel_index] = np.mean(bank, axis=0)
        return outputs


@register_op
class OrderBandEnergyOp(AggregateOp):
    op_name: ClassVar[str] = "order_band_energy"
    description: ClassVar[str] = "Computes energy in selected order bands for angle-domain vibration."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, N * C)"

    points_per_rev: int = Field(256, description="Samples per revolution used for order-domain frequency mapping.")
    bands: List[List[float]] = Field(
        default_factory=lambda: [[1.0, 3.0], [5.0, 7.0], [35.0, 37.0], [71.0, 73.0]],
        description="Order bands as [[low, high], ...].",
    )

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        if x.ndim != 3:
            raise ValueError(f"OrderBandEnergyOp expects (B, L, C), got {x.shape!r}")
        batch_size, _, channels = x.shape
        features: List[np.ndarray] = []
        for channel_index in range(channels):
            rows = []
            for batch_index in range(batch_size):
                orders, spectrum = _spectrum_from_signal(x[batch_index, :, channel_index], points_per_rev=self.points_per_rev)
                row = []
                power = spectrum ** 2
                for low, high in self.bands:
                    mask = (orders >= float(low)) & (orders <= float(high))
                    row.append(float(np.mean(power[mask])) if np.any(mask) else 0.0)
                rows.append(row)
            features.append(np.asarray(rows, dtype=float))
        return np.concatenate(features, axis=1) if features else np.zeros((batch_size, 0), dtype=float)


@register_op
class SidebandRatioOp(AggregateOp):
    op_name: ClassVar[str] = "sideband_ratio"
    description: ClassVar[str] = "Computes sideband-to-centerline ratio around a target order."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, C)"

    points_per_rev: int = Field(256, description="Samples per revolution used for order-domain frequency mapping.")
    center_order: float = Field(36.0, description="Target center order, typically gear-mesh related.")
    sideband_order: float = Field(1.0, description="Spacing between adjacent sidebands in order units.")
    num_sidebands: int = Field(2, description="Number of sideband pairs to include.")
    half_width: float = Field(0.5, description="Integration half-width around each order.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        if x.ndim != 3:
            raise ValueError(f"SidebandRatioOp expects (B, L, C), got {x.shape!r}")
        batch_size, _, channels = x.shape
        outputs = np.zeros((batch_size, channels), dtype=float)
        for batch_index in range(batch_size):
            for channel_index in range(channels):
                orders, spectrum = _spectrum_from_signal(x[batch_index, :, channel_index], points_per_rev=self.points_per_rev)

                def band_sum(center: float) -> float:
                    mask = (orders >= center - self.half_width) & (orders <= center + self.half_width)
                    return float(np.sum(spectrum[mask])) if np.any(mask) else 0.0

                center_value = band_sum(self.center_order)
                side_total = 0.0
                for sideband_index in range(1, int(self.num_sidebands) + 1):
                    shift = sideband_index * float(self.sideband_order)
                    side_total += band_sum(self.center_order - shift)
                    side_total += band_sum(self.center_order + shift)
                outputs[batch_index, channel_index] = side_total / (center_value + 1e-9)
        return outputs


@register_op
class TorqueNormalizeOp(MultiVariableOp):
    op_name: ClassVar[str] = "torque_normalize"
    description: ClassVar[str] = "Normalizes a feature tensor using the torque side-input."
    input_spec: ClassVar[str] = "feature: (B, ...), torque: (B, L, 1) or (B, 1)"
    output_spec: ClassVar[str] = "same as feature"

    eps: float = Field(1e-6, description="Stability epsilon.")
    mode: Literal["abs_mean", "rms"] = Field("abs_mean", description="How to summarize the torque side-input.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"TorqueNormalizeOp requires 2 inputs, got {len(x)}.")
        feature, torque = list(x.values())
        feature = np.asarray(feature, dtype=float)
        torque = np.asarray(torque, dtype=float)
        if feature.ndim < 2:
            raise ValueError("TorqueNormalizeOp expects feature input with a batch axis.")
        if torque.ndim < 2:
            raise ValueError("TorqueNormalizeOp expects torque input with a batch axis.")

        reduce_axes = tuple(range(1, torque.ndim))
        if self.mode == "rms":
            scale = np.sqrt(np.mean(np.square(torque), axis=reduce_axes))
        else:
            scale = np.mean(np.abs(torque), axis=reduce_axes)
        reshape = (scale.shape[0],) + (1,) * (feature.ndim - 1)
        return feature / (scale.reshape(reshape) + float(self.eps))
