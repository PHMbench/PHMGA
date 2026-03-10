from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Literal, Optional


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

@dataclass(frozen=True)
class CompatibilityProfile:
    """Compatibility constraints for dataset-specific bridge behavior."""

    name: str
    strict_methods: FrozenSet[str]
    aggregate_methods: FrozenSet[str]
    disallow_identity_for: FrozenSet[str]
    fail_on_proxy: bool = False
    fail_on_unsupported: bool = False


@dataclass(frozen=True)
class OperatorContract:
    """Closed-world contract for which operators can appear in DAG->TSPN path."""

    name: str
    allowed_layer_ops: FrozenSet[str]
    allowed_feature_ops: FrozenSet[str]

    @property
    def allowed_ops(self) -> FrozenSet[str]:
        return frozenset(set(self.allowed_layer_ops) | set(self.allowed_feature_ops))


MAPPING_VERSION = "v1.2"

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

_FEATURE_METHODS: FrozenSet[str] = frozenset(FEATURE_CATALOG.keys())
_RM101_STRICT_METHODS: FrozenSet[str] = frozenset(
    {
        # Priority ops from RM101 planning hint / playbook.
        "filter",
        "hilbert_envelope",
        "fft",
        "stft",
        "band_power",
        "cross_correlation",
        "spectral_kurtosis",
        "spectral_centroid",
        # Common transform ops that must remain semantically meaningful.
        "normalize",
        "detrend",
        "integrate",
        "differentiate",
        "psd",
        "spectrogram",
        "mel_spectrogram",
        "power_to_db",
        "resample",
        "savgol_filter",
        "denoise_wavelet",
        "wavelet_transform",
        "cepstrum",
        "patch",
        # Frequent aggregate/feature extraction ops on RM101.
        "mean",
        "std",
        "rms",
        "kurtosis",
        "skew",
        "entropy",
        "crest_factor",
        "shape_factor",
        "clearance_factor",
        "abs_mean",
        "peak_to_peak",
        "zero_crossing_rate",
        "spectral_skewness",
        "spectral_flatness",
        "hjorth_parameters",
    }
)
_RM101_DISALLOW_IDENTITY: FrozenSet[str] = frozenset(
    {
        "filter",
        "hilbert_envelope",
        "fft",
        "stft",
        "normalize",
        "detrend",
        "integrate",
        "differentiate",
        "psd",
        "spectrogram",
        "mel_spectrogram",
        "power_to_db",
        "resample",
        "savgol_filter",
        "denoise_wavelet",
        "wavelet_transform",
        "cepstrum",
        "patch",
    }
)
COMPATIBILITY_PROFILES: Dict[str, CompatibilityProfile] = {
    "default": CompatibilityProfile(
        name="default",
        strict_methods=frozenset(),
        aggregate_methods=_FEATURE_METHODS,
        disallow_identity_for=frozenset(),
        fail_on_proxy=False,
        fail_on_unsupported=False,
    ),
    "rm101_strict": CompatibilityProfile(
        name="rm101_strict",
        strict_methods=_RM101_STRICT_METHODS,
        aggregate_methods=_FEATURE_METHODS,
        disallow_identity_for=_RM101_DISALLOW_IDENTITY,
        fail_on_proxy=True,
        fail_on_unsupported=True,
    ),
}

_RM101_CLOSED_V1_LAYER_OPS: FrozenSet[str] = frozenset(
    {
        "detrend",
        "differentiate",
        "fft",
        "filter",
        "hilbert_envelope",
        "integrate",
        "normalize",
        "stft",
    }
)

_RM101_CLOSED_V1_FEATURE_OPS: FrozenSet[str] = frozenset(
    {
        "abs_mean",
        "clearance_factor",
        "crest_factor",
        "entropy",
        "kurtosis",
        "max",
        "mean",
        "min",
        "peak_to_peak",
        "rms",
        "shape_factor",
        "skew",
        "spectral_centroid",
        "spectral_flatness",
        "spectral_kurtosis",
        "spectral_skewness",
        "std",
        "var",
        "zero_crossing_rate",
    }
)

OPERATOR_CONTRACTS: Dict[str, OperatorContract] = {
    "default": OperatorContract(
        name="rm101_closed_v1",
        allowed_layer_ops=_RM101_CLOSED_V1_LAYER_OPS,
        allowed_feature_ops=_RM101_CLOSED_V1_FEATURE_OPS,
    ),
    "rm101_closed_v1": OperatorContract(
        name="rm101_closed_v1",
        allowed_layer_ops=_RM101_CLOSED_V1_LAYER_OPS,
        allowed_feature_ops=_RM101_CLOSED_V1_FEATURE_OPS,
    ),
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


def resolve_compatibility_profile(name: str | None) -> CompatibilityProfile:
    key = normalize_method_name(name or "") or "default"
    profile = COMPATIBILITY_PROFILES.get(key)
    if profile is None:
        raise ValueError(
            f"Unsupported compatibility profile: {name!r}. "
            f"Expected one of {sorted(COMPATIBILITY_PROFILES.keys())}."
        )
    return profile


def resolve_operator_contract(name: str | None) -> OperatorContract:
    key = normalize_method_name(name or "") or "rm101_closed_v1"
    contract = OPERATOR_CONTRACTS.get(key)
    if contract is None:
        raise ValueError(
            f"Unsupported operator contract: {name!r}. "
            f"Expected one of {sorted(OPERATOR_CONTRACTS.keys())}."
        )
    return contract


def is_contract_allowed(method: str, contract: OperatorContract) -> bool:
    method_key = normalize_method_name(method)
    return method_key in contract.allowed_ops


def contract_method_kind(method: str) -> str:
    method_key = normalize_method_name(method)
    if method_key in FEATURE_CATALOG:
        return "feature"
    if method_key in OPERATOR_CATALOG:
        return "layer"
    return "unknown"
