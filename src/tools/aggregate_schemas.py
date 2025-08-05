from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy.stats import skew, kurtosis
from typing import ClassVar
import scipy.stats

# 假设基类和注册器位于此
from .signal_processing_schemas import register_op, AggregateOp


# ---------- AGGREGATE (rank ↓) 类 ---------- #
# 这些算子通常沿时间/长度轴（默认为-2）聚合数据，
# 将 (..., L, C) 形状的输入转换为 (..., C) 形状的输出。

@register_op
class MeanOp(AggregateOp):
    op_name: ClassVar[str] = "mean"
    description: ClassVar[str] = "Computes the mean (average) value along the time axis."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.mean(x, axis=self.axis)

@register_op
class StdOp(AggregateOp):
    op_name: ClassVar[str] = "std"
    description: ClassVar[str] = "Computes the standard deviation along the time axis."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.std(x, axis=self.axis)

@register_op
class VarOp(AggregateOp):
    op_name: ClassVar[str] = "var"
    description: ClassVar[str] = "Computes the variance along the time axis."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.var(x, axis=self.axis)

@register_op
class SkewnessOp(AggregateOp):
    op_name: ClassVar[str] = "skew"
    description: ClassVar[str] = "Computes the skewness (asymmetry) of the data distribution."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return skew(x, axis=self.axis)

@register_op
class KurtosisOp(AggregateOp):
    op_name: ClassVar[str] = "kurtosis"
    description: ClassVar[str] = "Computes the kurtosis (tailedness) of the data distribution."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return kurtosis(x, axis=self.axis)

@register_op
class MaxOp(AggregateOp):
    op_name: ClassVar[str] = "max"
    description: ClassVar[str] = "Computes the maximum value along the time axis."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.max(x, axis=self.axis)

@register_op
class MinOp(AggregateOp):
    op_name: ClassVar[str] = "min"
    description: ClassVar[str] = "Computes the minimum value along the time axis."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.min(x, axis=self.axis)


@register_op
class RMSOp(AggregateOp):
    op_name: ClassVar[str] = "rms"
    description: ClassVar[str] = "Computes the Root Mean Square (RMS) of the signal."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.sqrt(np.mean(np.square(x), axis=self.axis))

@register_op
class AbsMeanOp(AggregateOp):
    op_name: ClassVar[str] = "abs_mean"
    description: ClassVar[str] = "Computes the mean of the absolute values of the signal."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.mean(np.abs(x), axis=self.axis)

@register_op
class ShapeFactorOp(AggregateOp):
    op_name: ClassVar[str] = "shape_factor"
    description: ClassVar[str] = "Computes the Shape Factor (RMS / Mean of Absolute Values)."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        rms = np.sqrt(np.mean(np.square(x), axis=self.axis))
        abs_mean = np.mean(np.abs(x), axis=self.axis)
        return rms / (abs_mean + 1e-9) # Add epsilon for stability

@register_op
class CrestFactorOp(AggregateOp):
    op_name: ClassVar[str] = "crest_factor"
    description: ClassVar[str] = "Computes the Crest Factor (Peak / RMS)."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        peak = np.max(np.abs(x), axis=self.axis)
        rms = np.sqrt(np.mean(np.square(x), axis=self.axis))
        return peak / (rms + 1e-9) # Add epsilon for stability

@register_op
class ClearanceFactorOp(AggregateOp):
    op_name: ClassVar[str] = "clearance_factor"
    description: ClassVar[str] = "Computes the Clearance Factor (Peak / (Mean of Sqrt Abs Values)^2)."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        peak = np.max(np.abs(x), axis=self.axis)
        sqrt_abs_mean_sq = np.square(np.mean(np.sqrt(np.abs(x)), axis=self.axis))
        return peak / (sqrt_abs_mean_sq + 1e-9) # Add epsilon for stability

@register_op
class EntropyOp(AggregateOp):
    op_name: ClassVar[str] = "entropy"
    description: ClassVar[str] = "Computes the Shannon entropy of the signal."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
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

@register_op
class BandPowerOp(AggregateOp):
    """Computes the average power in one or more frequency bands.

    The operator expects a time-domain signal of shape ``(B, L, C)``. It
    computes the power spectral density (PSD) via FFT and returns the
    mean power within each provided band.
    """

    op_name: ClassVar[str] = "band_power"
    description: ClassVar[str] = (
        "Computes the average power of the signal in one or more frequency bands."
    )
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., N, C)"  # N = number of bands

    fs: float = Field(..., description="Sampling frequency of the signal.")
    bands: list[tuple[float, float]] = Field(
        ..., description="List of frequency bands as (min_freq, max_freq)."
    )

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """Compute the average power for each band."""
        if x.ndim != 3:
            raise ValueError(
                f"BandPowerOp expects input of shape (B, L, C), got {x.shape}"
            )

        fft_vals = np.fft.rfft(x, axis=-2)
        freqs = np.fft.rfftfreq(x.shape[-2], d=1.0 / self.fs)
        psd = (np.abs(fft_vals) ** 2) / x.shape[-2]

        band_powers = []
        for band in self.bands:
            idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
            if idx.size == 0:
                band_powers.append(np.zeros((x.shape[0], x.shape[2])))
            else:
                band_powers.append(np.mean(psd[:, idx, :], axis=1))

        return np.stack(band_powers, axis=1)

@register_op
class PeakToPeakOp(AggregateOp):
    """Computes the peak-to-peak (P2P) value of the signal."""
    op_name: ClassVar[str] = "peak_to_peak"
    description: ClassVar[str] = "Computes the difference between the maximum and minimum value along the time axis."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        return np.max(x, axis=self.axis) - np.min(x, axis=self.axis)

@register_op
class ZeroCrossingRateOp(AggregateOp):
    """Computes the zero-crossing rate of the signal."""
    op_name: ClassVar[str] = "zero_crossing_rate"
    description: ClassVar[str] = "Computes the rate at which the signal changes from positive to negative or back."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    axis: int = Field(-2, description="Axis to perform aggregation on.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        # The rate is the number of crossings / total number of samples
        return np.mean(np.abs(np.diff(np.sign(x), axis=self.axis)), axis=self.axis) / 2

@register_op
class SpectralCentroidOp(AggregateOp):
    """Computes the spectral centroid, the center of mass of the spectrum."""
    op_name: ClassVar[str] = "spectral_centroid"
    description: ClassVar[str] = "Computes the center of mass of the spectrum, indicating where the energy is concentrated."
    input_spec: ClassVar[str] = "(..., F, C)"
    output_spec: ClassVar[str] = "(..., C)"
    
    fs: float = Field(3125, description="Sampling frequency of the signal.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """Assumes x is in the frequency domain (e.g., output of FFT)."""
        n_fft = (x.shape[-2] - 1) * 2 # Assuming x is from rfft
        freqs = np.fft.rfftfreq(n_fft, d=1./self.fs)
        
        # Ensure freqs align with the feature dimension of x
        freqs = freqs[:x.shape[-2]]
        
        # Add dimensions to freqs for broadcasting: (F) -> (1, F, 1)
        freqs = freqs[np.newaxis, :, np.newaxis]
        
        # x is magnitude spectrum, so power is x**2
        power_spectrum = x**2
        
        # Weighted sum of frequencies
        weighted_sum = np.sum(freqs * power_spectrum, axis=-2)
        total_power = np.sum(power_spectrum, axis=-2)
        
        return weighted_sum / (total_power + 1e-9)

@register_op
class SpectralSkewnessOp(AggregateOp):
    """Computes the skewness of the spectrum."""
    op_name: ClassVar[str] = "spectral_skewness"
    description: ClassVar[str] = "Computes the skewness of the spectrum, indicating its asymmetry."
    input_spec: ClassVar[str] = "(..., F, C)"
    output_spec: ClassVar[str] = "(..., C)"

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """Assumes x is a magnitude or power spectrum."""
        return skew(x, axis=-2)

@register_op
class SpectralKurtosisOp(AggregateOp):
    """Computes the kurtosis of the spectrum."""
    op_name: ClassVar[str] = "spectral_kurtosis"
    description: ClassVar[str] = "Computes the kurtosis of the spectrum, indicating its 'tailedness'."
    input_spec: ClassVar[str] = "(..., F, C)"
    output_spec: ClassVar[str] = "(..., C)"

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """Assumes x is a magnitude or power spectrum."""
        return scipy.stats.kurtosis(x, axis=-2)

@register_op
class HjorthParametersOp(AggregateOp):
    """
    Computes Hjorth parameters (Activity, Mobility, Complexity).
    """
    op_name: ClassVar[str] = "hjorth_parameters"
    description: ClassVar[str] = "Computes Hjorth parameters (Activity, Mobility, Complexity)."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., 3, C)" # 3 features per channel

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        if x.ndim < 2:
            raise ValueError(f"Input for HjorthParametersOp must be at least 2D, but got {x.ndim}D.")
        
        dx = np.diff(x, axis=-2)
        ddx = np.diff(dx, axis=-2)
        
        # Activity (variance)
        activity = np.var(x, axis=-2)
        
        # Mobility
        mobility = np.sqrt(np.var(dx, axis=-2) / activity)
        
        # Complexity
        complexity = np.sqrt(np.var(ddx, axis=-2) / np.var(dx, axis=-2)) / mobility
        
        # Stack the three parameters
        return np.stack([activity, mobility, complexity], axis=-2)


@register_op
class SpectralFlatnessOp(AggregateOp):
    """Computes the spectral flatness, a measure of the 'tonalness' of a spectrum."""
    op_name: ClassVar[str] = "spectral_flatness"
    description: ClassVar[str] = "Computes the spectral flatness (geometric mean / arithmetic mean)."
    input_spec: ClassVar[str] = "(..., F, C)"
    output_spec: ClassVar[str] = "(..., C)"

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """Assumes x is a magnitude spectrum."""
        geometric_mean = scipy.stats.gmean(x + 1e-9, axis=-2)
        arithmetic_mean = np.mean(x, axis=-2)
        return geometric_mean / (arithmetic_mean + 1e-9)

@register_op
class ApproximateEntropyOp(AggregateOp):
    """
    Computes the Approximate Entropy (ApEn), a measure of regularity and complexity.
    """
    op_name: ClassVar[str] = "approximate_entropy"
    description: ClassVar[str] = "Computes the Approximate Entropy (ApEn) to quantify signal regularity."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    
    m: int = Field(2, description="Embedding dimension.")
    r_coeff: float = Field(0.2, description="Filtering level coefficient to determine radius r.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        try:
            import nolds
        except ImportError:
            raise ImportError("nolds is not installed. Please install it with 'pip install nolds'.")

        if x.ndim != 3:
            raise ValueError(f"Input for ApEn must be 3D (B, L, C), but got {x.ndim}D.")

        batch_size, _, channels = x.shape
        results = np.zeros((batch_size, channels))
        for i in range(batch_size):
            for j in range(channels):
                r = self.r_coeff * np.std(x[i, :, j])
                results[i, j] = nolds.sampen(x[i, :, j], emb_dim=self.m, tolerance=r)
        return results

@register_op
class PermutationEntropyOp(AggregateOp):
    """
    Computes the Permutation Entropy, a measure of complexity based on ordinal patterns.
    """
    op_name: ClassVar[str] = "permutation_entropy"
    description: ClassVar[str] = "Computes the Permutation Entropy to quantify signal complexity."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., C)"
    
    order: int = Field(3, description="Order of the permutation.")
    delay: int = Field(1, description="Time delay.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        try:
            from antropy import perm_entropy
        except ImportError:
            raise ImportError("antropy is not installed. Please install it with 'pip install antropy'.")

        if x.ndim != 3:
            raise ValueError(f"Input for Permutation Entropy must be 3D (B, L, C), but got {x.ndim}D.")

        batch_size, _, channels = x.shape
        results = np.zeros((batch_size, channels))
        for i in range(batch_size):
            for j in range(channels):
                results[i, j] = perm_entropy(x[i, :, j], order=self.order, delay=self.delay, normalize=True)
        return results


if __name__ == "__main__":
    print("--- Testing aggregate_schemas.py ---")

    # Create a dummy spectrum: Batch=1, Freq Bins=1025, Channels=1
    fs = 2048
    dummy_spectrum = np.zeros((1, 1025, 1))
    dummy_spectrum[:, 200:300, :] = 5  # Add a clear peak region
    dummy_spectrum += 0.1  # small baseline to avoid zeros

    # 1. Test SpectralCentroidOp
    print("\n1. Testing SpectralCentroidOp...")
    sc_op = SpectralCentroidOp(fs=fs)
    sc_result = sc_op.execute(dummy_spectrum)
    print(f"Input shape: {dummy_spectrum.shape}")
    print(f"Spectral Centroid output shape: {sc_result.shape}")
    print(f"Spectral Centroid value: {sc_result.item():.2f} Hz")
    assert sc_result.shape == (1, 1)
    assert 200 < sc_result.item() < 300  # Check if centroid is in the peak region

    # 2. Test SpectralFlatnessOp
    print("\n2. Testing SpectralFlatnessOp...")
    sf_op = SpectralFlatnessOp()
    sf_result = sf_op.execute(dummy_spectrum)
    print(f"Spectral Flatness output shape: {sf_result.shape}")
    print(f"Spectral Flatness value: {sf_result.item():.4f}")
    assert sf_result.shape == (1, 1)
    assert 0 < sf_result.item() < 1

    # 3. Test PeakToPeakOp
    print("\n3. Testing PeakToPeakOp...")
    dummy_signal = np.sin(np.linspace(0, 10, 1000))[np.newaxis, :, np.newaxis]
    p2p_op = PeakToPeakOp()
    p2p_result = p2p_op.execute(dummy_signal)
    print(f"P2P output shape: {p2p_result.shape}")
    print(f"P2P value: {p2p_result.item():.2f}")
    assert p2p_result.shape == (1, 1)
    assert np.isclose(p2p_result.item(), 2.0)

    # 4. Test HjorthParametersOp
    print("\n4. Testing HjorthParametersOp...")
    hjorth_op = HjorthParametersOp()
    hjorth_result = hjorth_op.execute(dummy_signal)
    print(f"HjorthParametersOp output shape: {hjorth_result.shape}")
    assert hjorth_result.shape == (1, 3, 1)

    # 5. Test ApproximateEntropyOp
    print("\n5. Testing ApproximateEntropyOp...")
    apen_op = ApproximateEntropyOp(m=2, r_coeff=0.2)
    apen_result = apen_op.execute(dummy_signal)
    print(f"ApproximateEntropyOp output shape: {apen_result.shape}")
    assert apen_result.shape == (1, 1)

    # 6. Test PermutationEntropyOp
    print("\n6. Testing PermutationEntropyOp...")
    pent_op = PermutationEntropyOp(order=3, delay=1)
    pent_result = pent_op.execute(dummy_signal)
    print(f"PermutationEntropyOp output shape: {pent_result.shape}")
    assert pent_result.shape == (1, 1)

    # 7. Test BandPowerOp
    print("\n7. Testing BandPowerOp...")
    bp_op = BandPowerOp(fs=100, bands=[(0, 4), (4, 8), (8, 16)])
    bp_result = bp_op.execute(dummy_signal)
    print(f"BandPowerOp output shape: {bp_result.shape}")
    assert bp_result.shape == (1, 3, 1)
    print(f"BandPowerOp values: {bp_result}")
    # 8. Test ZeroCrossingRateOp
    print("\n8. Testing ZeroCrossingRateOp...")
    zcr_op = ZeroCrossingRateOp()
    zcr_result = zcr_op.execute(dummy_signal)
    print(f"ZeroCrossingRateOp output shape: {zcr_result.shape}")
    assert zcr_result.shape == (1, 1)

    # 9. Test SpectralSkewnessOp
    print("\n9. Testing SpectralSkewnessOp...")
    ss_op = SpectralSkewnessOp()
    ss_result = ss_op.execute(dummy_spectrum)
    print(f"SpectralSkewnessOp output shape: {ss_result.shape}")
    assert ss_result.shape == (1, 1)
    # 10. Test SpectralKurtosisOp
    print("\n10. Testing SpectralKurtosisOp...")
    sk_op = SpectralKurtosisOp()
    sk_result = sk_op.execute(dummy_spectrum)
    print(f"SpectralKurtosisOp output shape: {sk_result.shape}")
    assert sk_result.shape == (1, 1)

    # 11. Test SpectralFlatnessOp
    print("\n11. Testing SpectralFlatnessOp...")
    sf_op = SpectralFlatnessOp()
    sf_result = sf_op.execute(dummy_spectrum)
    print(f"SpectralFlatnessOp output shape: {sf_result.shape}")
    assert sf_result.shape == (1, 1)
    print(f"Spectral Flatness value: {sf_result.item():.4f}")

    # 12. Test MeanOp
    print("\n12. Testing MeanOp...")
    mean_op = MeanOp()
    mean_result = mean_op.execute(dummy_signal)
    print(f"MeanOp output shape: {mean_result.shape}")
    assert mean_result.shape == (1, 1)
    print(f"Mean value: {mean_result.item():.4f}")
    # 13. Test StdOp
    print("\n13. Testing StdOp...")
    std_op = StdOp()
    std_result = std_op.execute(dummy_signal)
    print(f"StdOp output shape: {std_result.shape}")
    assert std_result.shape == (1, 1)
    print(f"Std value: {std_result.item():.4f}")
    # 14. Test VarOp
    print("\n14. Testing VarOp...")
    var_op = VarOp()
    var_result = var_op.execute(dummy_signal)
    print(f"VarOp output shape: {var_result.shape}")
    assert var_result.shape == (1, 1)
    print(f"Var value: {var_result.item():.4f}")

    # 15. Test SkewnessOp
    print("\n15. Testing SkewnessOp...")
    skewness_op = SkewnessOp()
    skewness_result = skewness_op.execute(dummy_signal)
    print(f"SkewnessOp output shape: {skewness_result.shape}")
    assert skewness_result.shape == (1, 1)
    print(f"Skewness value: {skewness_result.item():.4f}")

    # 16. Test KurtosisOp
    print("\n16. Testing KurtosisOp...")
    kurtosis_op = KurtosisOp()
    kurtosis_result = kurtosis_op.execute(dummy_signal)
    print(f"KurtosisOp output shape: {kurtosis_result.shape}")
    assert kurtosis_result.shape == (1, 1)
    print(f"Kurtosis value: {kurtosis_result.item():.4f}")
    # 17. Test MaxOp
    print("\n17. Testing MaxOp...")
    max_op = MaxOp()
    max_result = max_op.execute(dummy_signal)
    print(f"MaxOp output shape: {max_result.shape}")
    assert max_result.shape == (1, 1)
    print(f"Max value: {max_result.item():.4f}")
    # 18. Test MinOp
    print("\n18. Testing MinOp...")
    min_op = MinOp()
    min_result = min_op.execute(dummy_signal)
    print(f"MinOp output shape: {min_result.shape}")
    assert min_result.shape == (1, 1)
    print(f"Min value: {min_result.item():.4f}") 

    # 19. Test RMSOp
    print("\n19. Testing RMSOp...")
    rms_op = RMSOp()
    rms_result = rms_op.execute(dummy_signal)
    print(f"RMSOp output shape: {rms_result.shape}")
    assert rms_result.shape == (1, 1)
    print(f"RMS value: {rms_result.item():.4f}")
    # 20. Test AbsMeanOp

    print("\n20. Testing AbsMeanOp...")
    abs_mean_op = AbsMeanOp()
    abs_mean_result = abs_mean_op.execute(dummy_signal)
    print(f"AbsMeanOp output shape: {abs_mean_result.shape}")
    assert abs_mean_result.shape == (1, 1)
    print(f"Abs Mean value: {abs_mean_result.item():.4f}")  

    # 21. Test ShapeFactorOp
    print("\n21. Testing ShapeFactorOp...")
    shape_factor_op = ShapeFactorOp()
    shape_factor_result = shape_factor_op.execute(dummy_signal)
    print(f"ShapeFactorOp output shape: {shape_factor_result.shape}")
    assert shape_factor_result.shape == (1, 1)
    print(f"Shape Factor value: {shape_factor_result.item():.4f}")
    # 22. Test CrestFactorOp

    print("\n22. Testing CrestFactorOp...")
    crest_factor_op = CrestFactorOp()
    crest_factor_result = crest_factor_op.execute(dummy_signal)
    print(f"CrestFactorOp output shape: {crest_factor_result.shape}")
    assert crest_factor_result.shape == (1, 1)
    print(f"Crest Factor value: {crest_factor_result.item():.4f}")
    # 23. Test ClearanceFactorOp
    print("\n23. Testing ClearanceFactorOp...")
    clearance_factor_op = ClearanceFactorOp()
    clearance_factor_result = clearance_factor_op.execute(dummy_signal)
    print(f"ClearanceFactorOp output shape: {clearance_factor_result.shape}")
    assert clearance_factor_result.shape == (1, 1)
    print(f"Clearance Factor value: {clearance_factor_result.item():.4f}")

    # 24. Test EntropyOp
    print("\n24. Testing EntropyOp...")
    entropy_op = EntropyOp()
    entropy_result = entropy_op.execute(dummy_signal)
    print(f"EntropyOp output shape: {entropy_result.shape}")
    assert entropy_result.shape == (1, 1)
    print(f"Entropy value: {entropy_result.item():.4f}")

    print("\n--- aggregate_schemas.py tests passed! ---")
