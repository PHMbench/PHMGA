from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


SupportedStatus = Literal["supported", "proxy", "unsupported"]


@dataclass(frozen=True)
class OperatorMapping:
    """Mapping from outer-DAG op name/method to inner TSPN token semantics."""

    op_name: str
    token: Optional[str]
    status: SupportedStatus
    default_params: Dict[str, Any]
    reason: str = ""


@dataclass(frozen=True)
class FeatureMapping:
    """Mapping from aggregate op names to feature token names."""

    op_name: str
    feature_token: Optional[str]
    status: SupportedStatus
    reason: str = ""


UNSUPPORTED_POLICY = {"fallback_to_identity", "drop", "error"}

MAPPING_VERSION = "v1.0"

# NOTE:
# - "supported": can be instantiated as a token in current TSPN.
# - "proxy": mapped to an approximate token (typically "I"), tracked in bridge metadata.
# - "unsupported": cannot be represented in current single-input parallel-token architecture.
OPERATOR_CATALOG: Dict[str, OperatorMapping] = {
    # Baseline tokens
    "identity": OperatorMapping("identity", "I", "supported", {}),
    "i": OperatorMapping("i", "I", "supported", {}),
    "fft": OperatorMapping("fft", "FFT", "supported", {}),
    "hilbert_envelope": OperatorMapping("hilbert_envelope", "HT", "supported", {}),
    "filter": OperatorMapping("filter", "WF", "supported", {}),
    # Additional single-input transforms
    "normalize": OperatorMapping("normalize", "NORM", "supported", {"method": "z_score"}),
    "detrend": OperatorMapping("detrend", "DT", "supported", {"type": "linear"}),
    "integrate": OperatorMapping("integrate", "INT", "supported", {}),
    "differentiate": OperatorMapping("differentiate", "DIFF", "supported", {}),
    "stft": OperatorMapping("stft", "STFT", "supported", {}),
    # Proxy mappings (semantically related, but no exact operator token at this stage)
    "psd": OperatorMapping("psd", "FFT", "proxy", {"align_strategy": "interp"}, reason="psd proxied to fft token"),
    "resample": OperatorMapping("resample", "I", "proxy", {}, reason="resample proxied to identity"),
    "power_to_db": OperatorMapping("power_to_db", "LOG", "proxy", {}, reason="power_to_db approximated with log"),
    "savgol_filter": OperatorMapping("savgol_filter", "I", "proxy", {}, reason="savgol proxied to identity"),
    "denoise_wavelet": OperatorMapping("denoise_wavelet", "WF", "proxy", {}, reason="denoise proxied to wavefilter"),
    "wavelet_transform": OperatorMapping("wavelet_transform", "WF", "proxy", {}, reason="wavelet proxied to wavefilter"),
    "spectrogram": OperatorMapping("spectrogram", "STFT", "proxy", {}, reason="spectrogram proxied to stft"),
    "mel_spectrogram": OperatorMapping("mel_spectrogram", "STFT", "proxy", {}, reason="mel_spectrogram proxied to stft"),
    "patch": OperatorMapping("patch", "I", "proxy", {}, reason="patch proxied to identity"),
    "cepstrum": OperatorMapping("cepstrum", "FFT", "proxy", {}, reason="cepstrum proxied to fft"),
    "log": OperatorMapping("log", "LOG", "supported", {}),
    "squ": OperatorMapping("squ", "SQU", "supported", {}),
    "sin": OperatorMapping("sin", "SIN", "supported", {}),
    # Unsupported multi-input and complex operators
    "subtract": OperatorMapping("subtract", None, "unsupported", {}, reason="multi-input operator"),
    "cross_correlation": OperatorMapping("cross_correlation", None, "unsupported", {}, reason="multi-input operator"),
    "distance": OperatorMapping("distance", None, "unsupported", {}, reason="multi-input operator"),
    "concatenate": OperatorMapping("concatenate", None, "unsupported", {}, reason="multi-input operator"),
    "element_wise_product": OperatorMapping("element_wise_product", None, "unsupported", {}, reason="multi-input operator"),
    "coherence": OperatorMapping("coherence", None, "unsupported", {}, reason="multi-input operator"),
    "arithmetic": OperatorMapping("arithmetic", None, "unsupported", {}, reason="multi-input operator"),
    "phase_difference": OperatorMapping("phase_difference", None, "unsupported", {}, reason="multi-input operator"),
    "convolution": OperatorMapping("convolution", None, "unsupported", {}, reason="multi-input operator"),
    "dtw_distance": OperatorMapping("dtw_distance", None, "unsupported", {}, reason="multi-input operator"),
    "transfer_function": OperatorMapping("transfer_function", None, "unsupported", {}, reason="multi-input operator"),
    "vmd": OperatorMapping("vmd", None, "unsupported", {}, reason="complex decomposition operator"),
    "emd": OperatorMapping("emd", None, "unsupported", {}, reason="complex decomposition operator"),
    "wigner_ville_distribution": OperatorMapping("wigner_ville_distribution", None, "unsupported", {}, reason="complex time-frequency operator"),
    "vqt": OperatorMapping("vqt", None, "unsupported", {}, reason="complex time-frequency operator"),
    "time_delay_embedding": OperatorMapping("time_delay_embedding", None, "unsupported", {}, reason="shape-changing embedding"),
    "pca": OperatorMapping("pca", None, "unsupported", {}, reason="cross-channel projection"),
}


FEATURE_CATALOG: Dict[str, FeatureMapping] = {
    "mean": FeatureMapping("mean", "Mean", "supported"),
    "std": FeatureMapping("std", "Std", "supported"),
    "var": FeatureMapping("var", "Var", "supported"),
    "entropy": FeatureMapping("entropy", "Entropy", "supported"),
    "max": FeatureMapping("max", "Max", "supported"),
    "min": FeatureMapping("min", "Min", "supported"),
    "abs_mean": FeatureMapping("abs_mean", "AbsMean", "supported"),
    "kurtosis": FeatureMapping("kurtosis", "Kurtosis", "supported"),
    "rms": FeatureMapping("rms", "RMS", "supported"),
    "crest_factor": FeatureMapping("crest_factor", "CrestFactor", "supported"),
    "clearance_factor": FeatureMapping("clearance_factor", "ClearanceFactor", "supported"),
    "skew": FeatureMapping("skew", "Skewness", "supported"),
    "shape_factor": FeatureMapping("shape_factor", "ShapeFactor", "supported"),
    "peak_to_peak": FeatureMapping("peak_to_peak", "PeakToPeak", "supported"),
    "zero_crossing_rate": FeatureMapping("zero_crossing_rate", "ZeroCrossingRate", "supported"),
    "spectral_centroid": FeatureMapping("spectral_centroid", "SpectralCentroid", "supported"),
    "spectral_skewness": FeatureMapping("spectral_skewness", "SpectralSkewness", "supported"),
    "spectral_kurtosis": FeatureMapping("spectral_kurtosis", "SpectralKurtosis", "supported"),
    "spectral_flatness": FeatureMapping("spectral_flatness", "SpectralFlatness", "supported"),
    # Non-scalar aggregate in tools is split into scalar feature tokens.
    "hjorth_parameters": FeatureMapping("hjorth_parameters", "HjorthActivity", "proxy", reason="split into 3 scalar features"),
    "band_power": FeatureMapping("band_power", None, "unsupported", reason="multi-band feature (N bands) not scalar"),
    "approximate_entropy": FeatureMapping("approximate_entropy", None, "unsupported", reason="not implemented in torch-side feature head"),
    "permutation_entropy": FeatureMapping("permutation_entropy", None, "unsupported", reason="not implemented in torch-side feature head"),
}


def normalize_method_name(name: str) -> str:
    return str(name or "").strip().lower()


def lookup_operator(name: str) -> OperatorMapping:
    key = normalize_method_name(name)
    mapping = OPERATOR_CATALOG.get(key)
    if mapping is not None:
        return mapping
    return OperatorMapping(op_name=key, token="I", status="proxy", default_params={}, reason="unknown op proxied to identity")


def lookup_feature(name: str) -> FeatureMapping:
    key = normalize_method_name(name)
    mapping = FEATURE_CATALOG.get(key)
    if mapping is not None:
        return mapping
    return FeatureMapping(op_name=key, feature_token=None, status="unsupported", reason="unknown feature op")
