"""AGGREGATE operators inspired by the C_Agent tools taxonomy."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats as scipy_stats

from .base import BaseIsomorphicOperator, OperatorSpec
from .common import (
    bridge_pt_via_numpy,
    ensure_float_tensor,
    flatten_numeric,
    flatten_numeric_pt,
    require_torch,
    scalar_feature,
    scalar_output_tensor,
    scalar_feature_pt,
    spectral_matrix,
    spectral_matrix_pt,
)


class MeanFeatureOperator(BaseIsomorphicOperator):
    """Global mean feature."""

    spec = OperatorSpec(
        op_uid="feature.mean",
        op_name="mean",
        name="Mean",
        schema_category="AGGREGATE",
        rank_class="rank_down",
        description="Aggregate a signal or spectrum into its global mean feature.",
        input_spec={"arity": "single", "min_rank": 1, "semantic": "numeric_tensor"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "reduce"},
        input_shape_rule="CxT|CxF|CxFxS",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml"],
        planning_notes="Useful for compact amplitude summaries after a transform branch.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        del kwargs
        return scalar_feature(float(np.asarray(x, dtype=float).mean()))

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"mean({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        del kwargs
        tensor = ensure_float_tensor(x, op_name=self.spec.op_uid)
        return scalar_output_tensor(tensor.mean(), like=tensor)


class StdFeatureOperator(BaseIsomorphicOperator):
    """Global standard-deviation feature."""

    spec = OperatorSpec(
        op_uid="feature.std",
        op_name="std",
        name="Std",
        schema_category="AGGREGATE",
        rank_class="rank_down",
        description="Aggregate a signal or spectrum into its global standard deviation feature.",
        input_spec={"arity": "single", "min_rank": 1, "semantic": "numeric_tensor"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "reduce"},
        input_shape_rule="CxT|CxF|CxFxS",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml"],
        planning_notes="Useful for compact dispersion summaries after a transform branch.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        del kwargs
        return scalar_feature(float(np.asarray(x, dtype=float).std()))

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"std({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        del kwargs
        tensor = ensure_float_tensor(x, op_name=self.spec.op_uid)
        return scalar_output_tensor(tensor.std(unbiased=False), like=tensor)


class RMSFeatureOperator(BaseIsomorphicOperator):
    """Root-mean-square feature."""

    spec = OperatorSpec(
        op_uid="feature.rms",
        op_name="rms",
        name="RMS",
        schema_category="AGGREGATE",
        rank_class="rank_down",
        description="Aggregate a signal or spectrum into a root-mean-square descriptor.",
        input_spec={"arity": "single", "min_rank": 1, "semantic": "numeric_tensor"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "reduce"},
        input_shape_rule="CxT|CxF|CxFxS",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="A standard PHM feature that transfers across classical and trainable paths.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        del kwargs
        array = np.asarray(x, dtype=float)
        return scalar_feature(float(np.sqrt(np.mean(np.square(array)))))

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"rms({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        del kwargs
        torch = require_torch()
        tensor = ensure_float_tensor(x, op_name=self.spec.op_uid)
        return scalar_output_tensor(torch.sqrt(torch.mean(torch.square(tensor))), like=tensor)


class KurtosisFeatureOperator(BaseIsomorphicOperator):
    """Kurtosis feature for impulsiveness detection."""

    spec = OperatorSpec(
        op_uid="feature.kurtosis",
        op_name="kurtosis",
        name="Kurtosis",
        schema_category="AGGREGATE",
        rank_class="rank_down",
        description="Aggregate a signal-like tensor into a scalar kurtosis feature.",
        input_spec={"arity": "single", "min_rank": 1, "semantic": "numeric_tensor"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "reduce"},
        input_shape_rule="CxT|CxF|CxPxL|CxFxS",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Useful for impulsive fault signatures after envelope or patch expansion.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        del kwargs
        flat = flatten_numeric(x)
        if flat.size == 0 or np.allclose(np.var(flat), 0.0):
            return scalar_feature(0.0)
        value = float(scipy_stats.kurtosis(flat, fisher=False, bias=False))
        if not np.isfinite(value):
            value = 0.0
        return scalar_feature(value)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"kurtosis({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        return bridge_pt_via_numpy(x, self.forward_np, op_name=self.spec.op_uid, **kwargs)


class CrestFactorFeatureOperator(BaseIsomorphicOperator):
    """Crest factor feature."""

    spec = OperatorSpec(
        op_uid="feature.crest_factor",
        op_name="crest_factor",
        name="Crest Factor",
        schema_category="AGGREGATE",
        rank_class="rank_down",
        description="Aggregate a waveform or envelope into a scalar crest-factor feature.",
        input_spec={"arity": "single", "min_rank": 1, "semantic": "numeric_tensor"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "reduce"},
        input_shape_rule="CxT|CxPxL",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Useful for transient and impulsive fault behavior in the time or envelope domain.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        del kwargs
        flat = np.abs(flatten_numeric(x))
        rms = float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0
        peak = float(np.max(flat)) if flat.size else 0.0
        value = peak / rms if rms > 0.0 else 0.0
        return scalar_feature(value)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"crest_factor({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        del kwargs
        torch = require_torch()
        flat = torch.abs(flatten_numeric_pt(x))
        if flat.numel() == 0:
            return scalar_feature_pt(0.0)
        rms = torch.sqrt(torch.mean(torch.square(flat)))
        peak = torch.max(flat)
        value = peak / rms if float(rms.item()) > 0.0 else torch.tensor(0.0, dtype=flat.dtype, device=flat.device)
        return scalar_output_tensor(value, like=flat)


class BandPowerFeatureOperator(BaseIsomorphicOperator):
    """Band-power feature for PSD or spectrogram branches."""

    spec = OperatorSpec(
        op_uid="feature.band_power",
        op_name="band_power",
        name="Band Power",
        schema_category="AGGREGATE",
        rank_class="rank_down",
        description="Aggregate a spectrum-like tensor into a scalar band-power feature.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "spectral_tensor"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "reduce"},
        param_schema={"fs": "float", "band_low_hz": "float", "band_high_hz": "float"},
        param_defaults={"band_low_hz": 0.0, "band_high_hz": 200.0},
        param_docs={
            "fs": "Sampling rate in Hz used to derive the frequency grid.",
            "band_low_hz": "Lower bound of the target band in Hz.",
            "band_high_hz": "Upper bound of the target band in Hz.",
        },
        input_shape_rule="CxF|CxFxS",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use on PSD or STFT branches to summarize energy in a fault-relevant band.",
        llm_tunable_params=["band_low_hz", "band_high_hz"],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        fs = float(kwargs.get("fs", 1.0))
        band_low = max(float(kwargs.get("band_low_hz", 0.0)), 0.0)
        band_high = max(float(kwargs.get("band_high_hz", fs / 4.0)), band_low)
        spectral = spectral_matrix(x)
        freqs = np.linspace(0.0, fs / 2.0, spectral.shape[-1], endpoint=True)
        mask = (freqs >= band_low) & (freqs <= band_high)
        if not np.any(mask):
            mask = np.ones_like(freqs, dtype=bool)
        value = float(np.mean(spectral[..., mask]))
        return scalar_feature(value)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"band_power({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        fs = float(kwargs.get("fs", 1.0))
        band_low = max(float(kwargs.get("band_low_hz", 0.0)), 0.0)
        band_high = max(float(kwargs.get("band_high_hz", fs / 4.0)), band_low)
        spectral = spectral_matrix_pt(x).to(dtype=torch.float32)
        freqs = torch.linspace(0.0, fs / 2.0, spectral.shape[-1], device=spectral.device, dtype=spectral.dtype)
        mask = (freqs >= band_low) & (freqs <= band_high)
        if not bool(mask.any()):
            mask = torch.ones_like(freqs, dtype=torch.bool)
        return scalar_output_tensor(spectral[..., mask].mean(), like=spectral)


class SpectralCentroidFeatureOperator(BaseIsomorphicOperator):
    """Spectral centroid feature."""

    spec = OperatorSpec(
        op_uid="feature.spectral_centroid",
        op_name="spectral_centroid",
        name="Spectral Centroid",
        schema_category="AGGREGATE",
        rank_class="rank_down",
        description="Aggregate a spectrum-like tensor into its spectral centroid.",
        input_spec={"arity": "single", "min_rank": 2, "semantic": "spectral_tensor"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "reduce"},
        param_schema={"fs": "float"},
        param_defaults={},
        param_docs={"fs": "Sampling rate in Hz used to derive the frequency grid."},
        input_shape_rule="CxF|CxFxS",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Useful for expressing where spectral energy is concentrated after PSD or STFT.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        fs = float(kwargs.get("fs", 1.0))
        spectral = np.abs(spectral_matrix(x))
        freqs = np.linspace(0.0, fs / 2.0, spectral.shape[-1], endpoint=True)
        weights = spectral.mean(axis=0)
        denom = float(np.sum(weights))
        if denom <= 0.0:
            return scalar_feature(0.0)
        value = float(np.sum(freqs * weights) / denom)
        return scalar_feature(value)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"spectral_centroid({x_sym})"

    def forward_pt(self, x, **kwargs: float):
        torch = require_torch()
        fs = float(kwargs.get("fs", 1.0))
        spectral = torch.abs(spectral_matrix_pt(x).to(dtype=torch.float32))
        freqs = torch.linspace(0.0, fs / 2.0, spectral.shape[-1], device=spectral.device, dtype=spectral.dtype)
        weights = spectral.mean(dim=0)
        denom = torch.sum(weights)
        if float(denom.item()) <= 0.0:
            return scalar_feature_pt(0.0, like=spectral)
        value = torch.sum(freqs * weights) / denom
        return scalar_output_tensor(value, like=spectral)


def get_aggregate_operators() -> Tuple[BaseIsomorphicOperator, ...]:
    """Return the runnable AGGREGATE operators."""

    return (
        MeanFeatureOperator(),
        StdFeatureOperator(),
        RMSFeatureOperator(),
        KurtosisFeatureOperator(),
        CrestFactorFeatureOperator(),
        BandPowerFeatureOperator(),
        SpectralCentroidFeatureOperator(),
    )
