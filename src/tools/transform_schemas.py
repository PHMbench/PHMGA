# src/tools/signal_ops.py
from __future__ import annotations

from typing import ClassVar, Literal, Dict
import numpy as np
import numpy.typing as npt
import scipy.signal
from pydantic import Field

from .signal_processing_schemas import (
    register_op,
    TransformOp,
)


@register_op  # <-- 确保 FFTOp 也被注册
class FFTOp(TransformOp):
    op_name: ClassVar[str] = "fft"
    description: ClassVar[str] = "Computes the Fast Fourier Transform of a real-valued signal."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., F, C)"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        # rfft 用于实数输入，效率更高。np.abs() 获取幅值谱。
        y = np.abs(np.fft.rfft(x, axis=-2))
        return y


@register_op
class NormalizeOp(TransformOp):
    """Normalize signal using z-score or min-max."""

    op_name: ClassVar[str] = "normalize"
    description: ClassVar[str] = "Normalize signal using z-score or min-max scaling."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"
    method: Literal["z_score", "min_max"] = Field("z_score", description="The normalization method to use.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        if self.method == "z_score":
            mean = np.mean(x, axis=-2, keepdims=True)
            std = np.std(x, axis=-2, keepdims=True)
            return (x - mean) / (std + 1e-9)
        elif self.method == "min_max":
            min_val = np.min(x, axis=-2, keepdims=True)
            max_val = np.max(x, axis=-2, keepdims=True)
            range_val = max_val - min_val
            return (x - min_val) / (range_val + 1e-9)
        else:
            # Pydantic 的验证可以防止这种情况
            raise ValueError(f"Unknown normalization method: {self.method}")


@register_op
class DetrendOp(TransformOp):
    """Remove polynomial trend from signal."""

    op_name: ClassVar[str] = "detrend"
    description: ClassVar[str] = "Remove a linear or constant trend from the signal."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"
    type: Literal["linear", "constant"] = Field("linear", description="The type of trend to remove.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        return scipy.signal.detrend(x, type=self.type, axis=-2)


@register_op
class CepstrumOp(TransformOp):
    """Compute real cepstrum of the signal."""
    op_name: ClassVar[str] = "cepstrum"
    description: ClassVar[str] = "Computes the real cepstrum, useful for detecting harmonic structures."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        spectrum = np.fft.fft(x, axis=-2)
        log_spec = np.log(np.abs(spectrum) + 1e-9)
        cepstrum = np.fft.ifft(log_spec, axis=-2).real
        return cepstrum


@register_op
class FilterOp(TransformOp):
    """Apply a Butterworth filter to the signal."""
    op_name: ClassVar[str] = "filter"
    description: ClassVar[str] = "Apply a Butterworth filter (low-pass, high-pass, or band-pass)."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"
    
    filter_type: Literal["low", "high", "band"] = Field(..., description="Type of filter.")
    fs: float = Field(..., description="Sampling frequency of the signal.")
    cutoff: float | tuple[float, float] = Field(..., description="Cutoff frequency or frequencies.")
    order: int = Field(5, description="Order of the Butterworth filter.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        nyquist = 0.5 * self.fs
        if isinstance(self.cutoff, tuple):
            normal_cutoff = (self.cutoff[0] / nyquist, self.cutoff[1] / nyquist)
        else:
            normal_cutoff = self.cutoff / nyquist
        
        b, a = scipy.signal.butter(self.order, normal_cutoff, btype=self.filter_type, analog=False)
        y = scipy.signal.lfilter(b, a, x, axis=-2)
        return y


@register_op
class HilbertEnvelopeOp(TransformOp):
    """Compute the envelope of the signal using the Hilbert transform."""
    op_name: ClassVar[str] = "hilbert_envelope"
    description: ClassVar[str] = "Computes the envelope of the signal via the Hilbert transform, useful for amplitude modulation analysis."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        # 希尔伯特变换沿时间轴应用
        analytic_signal = scipy.signal.hilbert(x, axis=-2)
        envelope = np.abs(analytic_signal)
        return envelope

@register_op
class ResampleOp(TransformOp):
    """Resample the signal to a new length."""
    op_name: ClassVar[str] = "resample"
    description: ClassVar[str] = "Resample the signal to a new desired length using Fourier method."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L_new, C)"
    
    num: int = Field(..., description="The new number of samples.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        # resample is applied along the time axis
        y = scipy.signal.resample(x, self.num, axis=-2)
        return y

@register_op
class DenoiseWaveletOp(TransformOp):
    """Denoise the signal using wavelet thresholding."""
    op_name: ClassVar[str] = "denoise_wavelet"
    description: ClassVar[str] = "Denoises the signal using wavelet thresholding."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"
    
    wavelet: str = Field("db1", description="Wavelet to use (e.g., 'db1', 'sym8').")
    mode: Literal["soft", "hard"] = Field("soft", description="Thresholding mode.")
    
    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets is not installed. Please install it with 'pip install PyWavelets'.")

        # Denoising is applied per channel, along the time axis
        if x.ndim != 3:
            raise ValueError(f"Input for DenoiseWaveletOp must be 3D (B, L, C), but got {x.ndim}D.")

        denoised = np.zeros_like(x)
        for i in range(x.shape[0]): # Iterate over batch
            for j in range(x.shape[2]): # Iterate over channels
                channel_signal = x[i, :, j]
                coeffs = pywt.wavedec(channel_signal, self.wavelet, mode='symmetric')
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(channel_signal)))
                
                new_coeffs = map(lambda c: pywt.threshold(c, value=threshold, mode=self.mode), coeffs)
                denoised[i, :, j] = pywt.waverec(list(new_coeffs), self.wavelet, mode='symmetric')
        return denoised

@register_op
class PowerSpectralDensityOp(TransformOp):
    """Computes the Power Spectral Density (PSD) of a signal."""
    op_name: ClassVar[str] = "psd"
    description: ClassVar[str] = "Computes the Power Spectral Density (PSD) using Welch's method."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., F, C)" # F is the number of frequency bins from Welch's method
    
    fs: float = Field(..., description="Sampling frequency of the signal.")
    nperseg: int = Field(256, description="Length of each segment for Welch's method.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        """
        Input shape: (B, L, C)
        Output shape: (B, F, C), where F is the number of frequency bins.
        """
        if x.ndim != 3:
            raise ValueError(f"Input for PSD must be 3D (B, L, C), but got {x.ndim}D.")

        # Scipy's welch works on the last axis, so we need to move the L-axis to the end.
        x_transposed = x.transpose(0, 2, 1) # -> (B, C, L)
        
        f, Pxx = scipy.signal.welch(x_transposed, fs=self.fs, nperseg=self.nperseg, axis=-1)
        
        # Pxx shape is (B, C, F). We want (B, F, C).
        return Pxx.transpose(0, 2, 1)

@register_op
class IntegrateOp(TransformOp):
    """
    Computes the cumulative integral of the signal along the time axis.
    Useful for converting acceleration to velocity, or velocity to displacement.
    """
    op_name: ClassVar[str] = "integrate"
    description: ClassVar[str] = "Computes the cumulative integral of the signal."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        return np.cumsum(x, axis=-2)

@register_op
class DifferentiateOp(TransformOp):
    """
    Computes the difference between adjacent elements in the signal (discrete derivative).
    Useful for converting displacement to velocity, or velocity to acceleration.
    """
    op_name: ClassVar[str] = "differentiate"
    description: ClassVar[str] = "Computes the discrete derivative of the signal."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L-1, C)"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        return np.diff(x, axis=-2)

@register_op
class PowerToDecibelOp(TransformOp):
    """
    Converts a power spectrogram or power spectral density to decibel (dB) units.
    """
    op_name: ClassVar[str] = "power_to_db"
    description: ClassVar[str] = "Converts a power spectrum/spectrogram to the decibel (dB) scale."
    input_spec: ClassVar[str] = "(..., F, C) or (..., F, T, C)"
    output_spec: ClassVar[str] = "Same as input"
    
    ref: float = 1.0
    top_db: float | None = 80.0

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        try:
            import librosa
        except ImportError:
            raise ImportError("Librosa is not installed. Please install it with 'pip install librosa'.")
        
        # Librosa's power_to_db works on the power values, shape is maintained.
        return librosa.power_to_db(x, ref=self.ref, top_db=self.top_db)

@register_op
class SavitzkyGolayFilterOp(TransformOp):
    """
    Smooths a signal using a Savitzky-Golay filter.
    """
    op_name: ClassVar[str] = "savgol_filter"
    description: ClassVar[str] = "Smooths a signal using a Savitzky-Golay filter."
    input_spec: ClassVar[str] = "(..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"
    
    window_length: int = Field(..., description="The length of the filter window (must be a positive odd integer).")
    polyorder: int = Field(..., description="The order of the polynomial used to fit the samples.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        return scipy.signal.savgol_filter(x, self.window_length, self.polyorder, axis=-2)

@register_op
class PrincipalComponentAnalysisOp(TransformOp):
    """
    Reduces the dimensionality of the feature space using PCA.
    """
    op_name: ClassVar[str] = "pca"
    description: ClassVar[str] = "Reduces feature dimensionality using Principal Component Analysis (PCA)."
    input_spec: ClassVar[str] = "(B, C')"
    output_spec: ClassVar[str] = "(B, n_components)"
    
    n_components: int = Field(..., description="Number of principal components to keep.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        from sklearn.decomposition import PCA
        
        if x.ndim != 2:
            raise ValueError(f"Input for PCA must be 2D (B, C'), but got {x.ndim}D.")
            
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(x)

# @register_op
# class Topk

# @register_op
# class AdaptiveFilterOp(TransformOp):
#     """
#     Applies an adaptive filter (NLMS) to remove noise from a signal.
#     Requires a desired signal `d` and an input signal `x`.
#     """
#     op_name: ClassVar[str] = "adaptive_filter"
#     description: ClassVar[str] = "Applies an adaptive filter (NLMS) to remove noise."
#     input_spec: ClassVar[str] = "Dict['d', 'x'] where d and x are (L,)"
#     output_spec: ClassVar[str] = "Dict['y', 'e', 'w']" # Output signal, error, final weights
    
#     n: int = Field(4, description="Filter length.")
#     mu: float = Field(0.1, description="Adaptation step size.")

#     def execute(self, x: Dict[str, npt.NDArray], **kw) -> Dict[str, npt.NDArray]:
#         try:
#             import padasip as pa
#         except ImportError:
#             raise ImportError("padasip is not installed. Please install it with 'pip install padasip'.")

#         if "d" not in x or "x" not in x:
#             raise ValueError("AdaptiveFilterOp requires "d" (desired) and "x" (input) signals.")

#         d = x["d"].squeeze()
#         input_x = x["x"].squeeze()
#         if input_x.ndim == 1:
#             input_x = input_x[:, None]

#         f = pa.filters.FilterNLMS(n=self.n, mu=self.mu, w="random")
#         y, e, w = f.run(d, input_x)
        
#         return {"y": y, "e": e, "w": w}


if __name__ == "__main__":
    print("--- Testing transform_schemas.py ---")
    
    # Create a dummy signal: Batch=1, Length=2048, Channels=1
    fs = 1024
    L = 2048
    t = np.linspace(0, L/fs, L, endpoint=False)
    dummy_signal = np.sin(2 * np.pi * 100 * t) + np.random.randn(L) * 0.1
    dummy_signal = dummy_signal[np.newaxis, :, np.newaxis] # Shape it to (1, 2048, 1)

    # 1. Test DifferentiateOp
    print("\n1. Testing DifferentiateOp...")
    diff_op = DifferentiateOp()
    diff_result = diff_op.execute(dummy_signal)
    print(f"Input shape: {dummy_signal.shape}")
    print(f"DifferentiateOp output shape: {diff_result.shape}")
    assert diff_result.shape == (1, 2047, 1)

    # 2. Test PowerSpectralDensityOp
    print("\n2. Testing PowerSpectralDensityOp...")
    psd_op = PowerSpectralDensityOp(fs=fs, nperseg=512)
    psd_result = psd_op.execute(dummy_signal)
    print(f"PSD output shape: {psd_result.shape}")
    assert psd_result.shape == (1, 257, 1)

    # 3. Test PowerToDecibelOp
    print("\n3. Testing PowerToDecibelOp...")
    db_op = PowerToDecibelOp()
    db_result = db_op.execute(psd_result)
    print(f"PowerToDecibelOp output shape: {db_result.shape}")
    assert db_result.shape == psd_result.shape
    assert np.max(db_result) <= 0 # dB of power should be <= 0

    # 4. Test SavitzkyGolayFilterOp
    print("\n4. Testing SavitzkyGolayFilterOp...")
    noisy_signal = dummy_signal + np.random.randn(*dummy_signal.shape) * 0.5
    sg_op = SavitzkyGolayFilterOp(window_length=51, polyorder=3)
    sg_result = sg_op.execute(noisy_signal)
    print(f"SavitzkyGolayFilterOp output shape: {sg_result.shape}")
    assert sg_result.shape == noisy_signal.shape
    # Check if noise is reduced (variance should be smaller)
    assert np.var(sg_result) < np.var(noisy_signal)

    # 5. Test PrincipalComponentAnalysisOp
    print("\n5. Testing PrincipalComponentAnalysisOp...")
    # Create dummy features
    features = np.random.rand(100, 10) # 100 samples, 10 features
    pca_op = PrincipalComponentAnalysisOp(n_components=3)
    pca_result = pca_op.execute(features)
    print(f"PCA output shape: {pca_result.shape}")
    assert pca_result.shape == (100, 3)

    # 6. Test AdaptiveFilterOp
    print("\n6. Testing AdaptiveFilterOp...")
    # Create a desired signal and a noisy input
    d = np.sin(np.linspace(0, 100, 1000))
    noise = np.random.randn(1000) * 0.1
    x_in = d + noise
    af_op = AdaptiveFilterOp(n=1, mu=0.1)
    af_result = af_op.execute({"d": d, "x": x_in})
    print(f"Adaptive filter output keys: {af_result.keys()}")
    assert "y" in af_result and "e" in af_result
    assert np.isfinite(af_result["e"]).all()

    print("\n--- transform_schemas.py tests passed! ---")
