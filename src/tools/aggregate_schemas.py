from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy.stats import skew, kurtosis
from typing import ClassVar

# 假设基类和注册器位于此
from .signal_processing_schemas import register_op, AggregateOp


# ---------- AGGREGATE (rank ↓) 类 ---------- #
# 这些算子通常沿时间/长度轴（默认为-2）聚合数据，
# 将 (..., L, C) 形状的输入转换为 (..., C) 形状的输出。

@register_op
class MeanOp(AggregateOp):
    op_name: ClassVar[str] = "mean"
    description: ClassVar[str] = "Computes the mean (average) value along the time axis."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.mean(x, axis=self.axis)

@register_op
class StdOp(AggregateOp):
    op_name: ClassVar[str] = "std"
    description: ClassVar[str] = "Computes the standard deviation along the time axis."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.std(x, axis=self.axis)

@register_op
class VarOp(AggregateOp):
    op_name: ClassVar[str] = "var"
    description: ClassVar[str] = "Computes the variance along the time axis."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.var(x, axis=self.axis)

@register_op
class SkewnessOp(AggregateOp):
    op_name: ClassVar[str] = "skew"
    description: ClassVar[str] = "Computes the skewness (asymmetry) of the data distribution."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return skew(x, axis=self.axis)

@register_op
class KurtosisOp(AggregateOp):
    op_name: ClassVar[str] = "kurtosis"
    description: ClassVar[str] = "Computes the kurtosis (tailedness) of the data distribution."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return kurtosis(x, axis=self.axis)


@register_op
class RMSOp(AggregateOp):
    op_name: ClassVar[str] = "rms"
    description: ClassVar[str] = "Computes the Root Mean Square (RMS) of the signal."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.sqrt(np.mean(np.square(x), axis=self.axis))

@register_op
class AbsMeanOp(AggregateOp):
    op_name: ClassVar[str] = "abs_mean"
    description: ClassVar[str] = "Computes the mean of the absolute values of the signal."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.mean(np.abs(x), axis=self.axis)

@register_op
class ShapeFactorOp(AggregateOp):
    op_name: ClassVar[str] = "shape_factor"
    description: ClassVar[str] = "Computes the Shape Factor (RMS / Mean of Absolute Values)."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        rms = np.sqrt(np.mean(np.square(x), axis=self.axis))
        abs_mean = np.mean(np.abs(x), axis=self.axis)
        return rms / (abs_mean + 1e-9) # Add epsilon for stability

@register_op
class CrestFactorOp(AggregateOp):
    op_name: ClassVar[str] = "crest_factor"
    description: ClassVar[str] = "Computes the Crest Factor (Peak / RMS)."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        peak = np.max(np.abs(x), axis=self.axis)
        rms = np.sqrt(np.mean(np.square(x), axis=self.axis))
        return peak / (rms + 1e-9) # Add epsilon for stability

@register_op
class ClearanceFactorOp(AggregateOp):
    op_name: ClassVar[str] = "clearance_factor"
    description: ClassVar[str] = "Computes the Clearance Factor (Peak / (Mean of Sqrt Abs Values)^2)."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        peak = np.max(np.abs(x), axis=self.axis)
        sqrt_abs_mean_sq = np.square(np.mean(np.sqrt(np.abs(x)), axis=self.axis))
        return peak / (sqrt_abs_mean_sq + 1e-9) # Add epsilon for stability

@register_op
class EntropyOp(AggregateOp):
    op_name: ClassVar[str] = "entropy"
    description: ClassVar[str] = "Computes the Shannon entropy of the signal."
    axis: int = Field(-2, description="Axis to perform aggregation on.")
    num_bins: int = Field(100, description="Number of bins for histogram to estimate probability.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        
        def _calculate_entropy_1d(signal_1d: np.ndarray) -> float:
            """Helper to calculate entropy for a 1D signal."""
            # Create a probability distribution using a histogram
            counts, _ = np.histogram(signal_1d, bins=self.num_bins, density=True)
            # Normalize to get probabilities
            probs = counts / np.sum(counts)
            # Filter out zero probabilities to avoid log(0)
            probs = probs[probs > 0]
            # Calculate entropy
            return -np.sum(probs * np.log2(probs))

        # Apply the entropy calculation along the specified axis
        return np.apply_along_axis(_calculate_entropy_1d, self.axis, x)
