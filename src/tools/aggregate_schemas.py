from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats
from pydantic import Field

# 假设基类和注册器位于此
from .signal_processing_schemas import register_op, AggregateOp


# ---------- AGGREGATE (rank ↓) 类 ---------- #
# 这些算子通常沿时间/长度轴（默认为-2）聚合数据，
# 将 (..., L, C) 形状的输入转换为 (..., C) 形状的输出。

@register_op
class MeanOp(AggregateOp):
    op_name: str = "mean"
    description: str = "Computes the mean (average) value along the time axis."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.mean(x, axis=self.axis)

@register_op
class StdOp(AggregateOp):
    op_name: str = "std"
    description: str = "Computes the standard deviation along the time axis."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.std(x, axis=self.axis)

@register_op
class VarOp(AggregateOp):
    op_name: str = "var"
    description: str = "Computes the variance along the time axis."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.var(x, axis=self.axis)

@register_op
class SkewnessOp(AggregateOp):
    op_name: str = "skew"
    description: str = "Computes the skewness (asymmetry) of the data distribution."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return scipy.stats.skew(x, axis=self.axis)

@register_op
class KurtosisOp(AggregateOp):
    op_name: str = "kurtosis"
    description: str = "Computes the kurtosis (tailedness) of the data distribution."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return scipy.stats.kurtosis(x, axis=self.axis)

@register_op
class RMSOp(AggregateOp):
    op_name: str = "rms"
    description: str = "Computes the Root Mean Square (RMS) of the signal."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.sqrt(np.mean(np.square(x), axis=self.axis))

@register_op
class AbsMeanOp(AggregateOp):
    op_name: str = "abs_mean"
    description: str = "Computes the mean of the absolute values of the signal."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.mean(np.abs(x), axis=self.axis)

@register_op
class ShapeFactorOp(AggregateOp):
    op_name: str = "shape_factor"
    description: str = "Computes the Shape Factor (RMS / Absolute Mean)."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        rms = np.sqrt(np.mean(np.square(x), axis=self.axis))
        abs_mean = np.mean(np.abs(x), axis=self.axis)
        return rms / (abs_mean + 1e-9) # Add epsilon for stability

@register_op
class CrestFactorOp(AggregateOp):
    op_name: str = "crest_factor"
    description: str = "Computes the Crest Factor (Peak / RMS)."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        peak = np.max(np.abs(x), axis=self.axis)
        rms = np.sqrt(np.mean(np.square(x), axis=self.axis))
        return peak / (rms + 1e-9) # Add epsilon for stability

@register_op
class ClearanceFactorOp(AggregateOp):
    op_name: str = "clearance_factor"
    description: str = "Computes the Clearance Factor (Peak / (Mean of Sqrt Abs Values)^2)."
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        peak = np.max(np.abs(x), axis=self.axis)
        sqrt_abs_mean_sq = np.square(np.mean(np.sqrt(np.abs(x)), axis=self.axis))
        return peak / (sqrt_abs_mean_sq + 1e-9) # Add epsilon for stability

@register_op
class EntropyOp(AggregateOp):
    op_name: str = "entropy"
    description: str = "Computes the Shannon entropy of the signal."
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
            return -np.sum(probs * np.log2(probs))

        return np.apply_along_axis(_calculate_entropy_1d, axis=self.axis, arr=x)
