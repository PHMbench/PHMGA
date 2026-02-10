from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class OpTokenConfig(BaseModel):
    """One signal-processing operator token (torch-side)."""

    token: str = Field(..., description="Operator token, e.g. I/WF/HT/FFT/NORM/DT/INT/DIFF/STFT/LOG/SQU/SIN.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Token parameters (token-specific).")

    class Config:
        extra = "forbid"


class LayerConfig(BaseModel):
    """A signal-processing layer consisting of multiple parallel operator tokens."""

    ops: List[OpTokenConfig] = Field(..., min_length=1)
    gate_temperature: float = Field(default=1.0, ge=1e-6)

    class Config:
        extra = "forbid"


class ModelConfig(BaseModel):
    """TSPN model structure config (immutable truth, edited by outer agent)."""

    name: Literal["tspn"] = "tspn"
    device: str = "cpu"
    num_classes: int = Field(..., ge=2)
    in_dim: int = Field(..., ge=1, description="Signal length L.")
    in_channels: int = Field(..., ge=1, description="Number of channels C in fused view.")
    out_channels: int = Field(default=3, ge=1)
    scale: int = Field(default=4, ge=1)
    skip_connection: bool = True

    # Default init params for WF tokens (can be overridden per-op via params).
    wf_init: Dict[str, float] = Field(
        default_factory=lambda: {"f_c_mu": 0.0, "f_c_sigma": 0.1, "f_b_mu": 0.0, "f_b_sigma": 0.1}
    )
    norm_init: Dict[str, Any] = Field(default_factory=lambda: {"method": "z_score"})
    stft_init: Dict[str, Any] = Field(default_factory=lambda: {"n_fft": 256, "hop_length": 128})
    sin_init: Dict[str, Any] = Field(default_factory=lambda: {"frequency": 1.0})
    preserve_topology: bool = True
    allow_duplicate_tokens: bool = True
    unsupported_policy: Literal["fallback_to_identity", "drop", "error"] = "fallback_to_identity"

    layers: List[LayerConfig] = Field(..., min_length=1)
    features: List[str] = Field(default_factory=lambda: ["Mean", "Std", "RMS"])

    # Soft delete gates (by stable op_uid).
    disabled_ops: Dict[str, float] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class TrainConfig(BaseModel):
    """Inner-loop training hyperparameters."""

    seed: int = 42
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    patience: int = Field(default=10, ge=1, description="Early stopping patience on val metric.")

    # Regularization on operator gates (encourage sparsity / avoid redundancy).
    l1_gate: float = Field(default=0.0, ge=0.0, description="L1 penalty weight on gate values.")
    entropy_gate: float = Field(default=0.0, ge=0.0, description="Entropy penalty weight on gate probabilities.")

    # Smoke-run/debug knobs (must never be enabled by default in production).
    debug: bool = False
    debug_max_samples: int = Field(default=16, ge=1)
    debug_epochs: int = Field(default=1, ge=1)

    class Config:
        extra = "forbid"


class ExplainConfig(BaseModel):
    """Explainability/report controls."""

    topk_ops: int = Field(default=3, ge=1)
    save_wavefilters: bool = True

    class Config:
        extra = "forbid"


class TSPNConfig(BaseModel):
    """Full model_config.yaml schema for TSPN."""

    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional free-form metadata.")

    class Config:
        extra = "forbid"


def validate_config_dict(data: Dict[str, Any]) -> TSPNConfig:
    return TSPNConfig.model_validate(data)
