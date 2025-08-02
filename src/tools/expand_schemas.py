# src/tools/signal_ops.py
from __future__ import annotations
from typing import ClassVar, Literal
import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy import signal
from skimage.util import view_as_windows

from .signal_processing_schemas import ExpandOp, register_op


@register_op
class PatchOp(ExpandOp):
    """
    Splits a long signal into smaller, potentially overlapping patches.
    This is often a preliminary step for applying image-based models (like CNNs) to time-series data.
    """
    op_name: ClassVar[str] = "patch"
    description: ClassVar[str] = "Splits a signal into smaller, potentially overlapping patches."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, N, P, C)"
    
    patch_size: int = Field(..., description="The number of samples in each patch (window size).")
    stride: int = Field(..., description="The number of samples to slide the window forward.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """
        Input shape: (B, L, C)
        Output shape: (B, N, P, C), where N is number of patches, P is patch_size.
        """
        if x.ndim != 3:
            raise ValueError(f"Input for PatchOp must be 3D (B, L, C), but got {x.ndim}D.")
        
        # We need to transpose to (B, C, L) for windowing, then transpose back.
        x_transposed = x.transpose(0, 2, 1) # -> (B, C, L)
        
        # view_as_windows works on the last axis.
        # It creates a view, so we need to copy to get a contiguous array.
        # The window shape needs to match the dimensions of the array being windowed.
        window_shape = (1,) * (x_transposed.ndim - 1) + (self.patch_size,)
        step = (1,) * (x_transposed.ndim - 1) + (self.stride,)

        patches = view_as_windows(x_transposed, window_shape=window_shape, step=step)

        # Remove singleton dimensions introduced by view_as_windows
        patches = patches[..., 0, 0, :]

        # Now shape is (B, C, N, P); transpose to (B, N, P, C)
        return patches.transpose(0, 2, 3, 1)


@register_op
class STFTOp(ExpandOp):
    """
    Computes the Short-Time Fourier Transform (STFT).
    This transforms a 1D signal into a 2D time-frequency representation.
    """
    op_name: ClassVar[str] = "stft"
    description: ClassVar[str] = "Computes the Short-Time Fourier Transform (STFT)."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, F, T, C)"
    
    fs: float = Field(..., description="Sampling frequency of the signal.")
    nperseg: int = Field(256, description="Length of each segment.")
    noverlap: int | None = Field(None, description="Number of points to overlap between segments.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """
        Input shape: (B, L, C)
        Output shape: (B, F, T, C), where F is frequency bins, T is time frames.
        """
        if x.ndim != 3:
            raise ValueError(f"Input for STFTOp must be 3D (B, L, C), but got {x.ndim}D.")

        # Scipy's stft works on the last axis, so we need to move the L-axis to the end.
        x_transposed = x.transpose(0, 2, 1) # -> (B, C, L)
        
        f, t, Zxx = signal.stft(x_transposed, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=-1)
        
        # Zxx shape is (B, C, F, T). We want (B, F, T, C).
        # We also take the absolute value to get the magnitude spectrogram.
        return np.abs(Zxx).transpose(0, 2, 3, 1)


@register_op
class MelSpectrogramOp(ExpandOp):
    """
    Computes the Mel Spectrogram of a signal.
    This is highly effective for audio and vibration analysis as it mimics human auditory response.
    """
    op_name: ClassVar[str] = "mel_spectrogram"
    description: ClassVar[str] = "Computes the Mel Spectrogram of a signal."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, M, T, C)"
    
    fs: float = Field(..., description="Sampling frequency of the signal.")
    n_fft: int = Field(2048, description="Length of the FFT window.")
    hop_length: int = Field(512, description="Number of samples between successive frames.")
    n_mels: int = Field(128, description="Number of Mel bands to generate.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """
        Input shape: (B, L, C)
        Output shape: (B, M, T, C), where M is n_mels, T is time frames.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("Librosa is not installed. Please install it with 'pip install librosa'.")

        if x.ndim != 3:
            raise ValueError(f"Input for MelSpectrogramOp must be 3D (B, L, C), but got {x.ndim}D.")

        # Librosa works with (C, L) or (L,) for batch processing we iterate.
        batch_size, _, channels = x.shape
        mel_specs = []
        for i in range(batch_size):
            channel_specs = []
            for j in range(channels):
                # librosa.feature.melspectrogram expects a 1D array
                S = librosa.feature.melspectrogram(y=x[i, :, j], sr=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
                S_db = librosa.power_to_db(S, ref=np.max)
                channel_specs.append(S_db)
            mel_specs.append(np.stack(channel_specs, axis=-1))
        
        # Output shape from stack is (B, M, T, C)
        return np.stack(mel_specs, axis=0)


@register_op
class ScalogramOp(ExpandOp):
    """Compute scalogram using continuous wavelet transform."""

    op_name: ClassVar[str] = "scalogram"
    description: ClassVar[str] = "Continuous wavelet scalogram using PyWavelets."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, S, L, C)"

    wavelet: str = Field("morl", description="Name of the wavelet to use (e.g., 'morl', 'mexh').")
    scales: list[int] = Field(..., description="List of scales to use for the CWT.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets is not installed. Please install it with 'pip install PyWavelets'.")

        # pywt.cwt expects the axis to be the last one.
        # Input shape: (B, L, C) -> Transpose to (B, C, L)
        x_transposed = x.transpose(0, 2, 1)
        coeffs, _ = pywt.cwt(x_transposed, self.scales, self.wavelet, axis=-1)
        # Coeffs shape: (num_scales, B, C, L) -> Transpose to (B, S, L, C)
        return np.abs(coeffs.transpose(1, 0, 3, 2))

@register_op
class WignerVilleDistributionOp(ExpandOp):
    """
    Computes the Wigner-Ville Distribution (WVD).
    Provides a high-resolution time-frequency representation, but may suffer from cross-term interference.
    """
    op_name: ClassVar[str] = "wigner_ville_distribution"
    description: ClassVar[str] = "Computes the Wigner-Ville Distribution for time-frequency analysis."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, L, F, C)" # Note: F is often L

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """
        This is a simplified implementation. For production, using a dedicated library like `tftb` is recommended.
        """
        if x.ndim != 3:
            raise ValueError(f"Input for WVD must be 3D (B, L, C), but got {x.ndim}D.")

        batch_size, n_samples, n_channels = x.shape
        wvd_results = []

        for i in range(batch_size):
            channel_wvd = []
            for j in range(n_channels):
                analytic_signal = signal.hilbert(x[i, :, j])
                # This is a pseudo-WVD to keep it simple
                tfr = np.zeros((n_samples, n_samples), dtype=complex)
                for n in range(n_samples):
                    taumax = min(n, n_samples - 1 - n)
                    for tau in range(-taumax, taumax + 1):
                        tfr[n, n] += analytic_signal[n + tau] * np.conj(analytic_signal[n - tau])
                channel_wvd.append(np.abs(tfr))
            wvd_results.append(np.stack(channel_wvd, axis=-1))
        
        return np.stack(wvd_results, axis=0)

@register_op
class SpectrogramOp(ExpandOp):
    """
    Computes a spectrogram, which is the squared magnitude of the STFT.
    """
    op_name: ClassVar[str] = "spectrogram"
    description: ClassVar[str] = "Computes the spectrogram (power of STFT) of a signal."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, F, T, C)"
    
    fs: float = Field(..., description="Sampling frequency of the signal.")
    nperseg: int = Field(256, description="Length of each segment.")
    noverlap: int | None = Field(None, description="Number of points to overlap between segments.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        """
        Input shape: (B, L, C)
        Output shape: (B, F, T, C), where F is frequency bins, T is time frames.
        """
        if x.ndim != 3:
            raise ValueError(f"Input for SpectrogramOp must be 3D (B, L, C), but got {x.ndim}D.")

        x_transposed = x.transpose(0, 2, 1) # -> (B, C, L)
        
        f, t, Zxx = signal.stft(x_transposed, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=-1)
        
        # Spectrogram is the squared magnitude
        return (np.abs(Zxx)**2).transpose(0, 2, 3, 1)

@register_op
class VariableQTransformOp(ExpandOp):
    """
    Computes the Variable-Q Transform (VQT), which provides a constant-Q resolution on a logarithmic frequency axis.
    Excellent for analyzing signals where frequency components are logarithmically spaced, like in music or gearboxes.
    """
    op_name: ClassVar[str] = "vqt"
    description: ClassVar[str] = "Computes the Variable-Q Transform (VQT) for logarithmic frequency analysis."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, Q, T, C)" # Q is the number of VQT bins
    
    fs: float = Field(..., description="Sampling frequency of the signal.")
    hop_length: int = Field(512, description="Number of samples between successive frames.")
    fmin: float = Field(..., description="Minimum frequency.")
    n_bins: int = Field(84, description="Number of frequency bins.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        try:
            import librosa
        except ImportError:
            raise ImportError("Librosa is not installed. Please install it with 'pip install librosa'.")

        if x.ndim != 3:
            raise ValueError(f"Input for VQT must be 3D (B, L, C), but got {x.ndim}D.")

        batch_size, _, channels = x.shape
        vqt_results = []
        for i in range(batch_size):
            channel_results = []
            for j in range(channels):
                C = librosa.vqt(y=x[i, :, j], sr=self.fs, hop_length=self.hop_length, fmin=self.fmin, n_bins=self.n_bins)
                channel_results.append(np.abs(C))
            vqt_results.append(np.stack(channel_results, axis=-1))
        
        return np.stack(vqt_results, axis=0)

@register_op
class TimeDelayEmbeddingOp(ExpandOp):
    """
    Performs time-delay embedding to reconstruct the phase space of a dynamical system from a 1D time series.
    """
    op_name: ClassVar[str] = "time_delay_embedding"
    description: ClassVar[str] = "Reconstructs phase space using time-delay embedding."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, L', D, C)" # L' is new length, D is embedding dimension
    
    dimension: int = Field(..., description="Embedding dimension (D).")
    delay: int = Field(..., description="Time delay (tau).")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        if x.ndim != 3:
            raise ValueError(f"Input for TimeDelayEmbeddingOp must be 3D (B, L, C), but got {x.ndim}D.")

        batch_size, n_samples, n_channels = x.shape
        
        # The new length of the time series
        new_length = n_samples - (self.dimension - 1) * self.delay
        if new_length <= 0:
            raise ValueError("Resulting length is non-positive. Decrease dimension or delay.")
            
        embedded_results = []
        for i in range(batch_size):
            channel_results = []
            for j in range(n_channels):
                # Create a new array for the embedded points
                embedded_channel = np.zeros((new_length, self.dimension))
                for k in range(new_length):
                    embedded_channel[k] = [x[i, k + m * self.delay, j] for m in range(self.dimension)]
                channel_results.append(embedded_channel)
            embedded_results.append(np.stack(channel_results, axis=-1))
            
        return np.stack(embedded_results, axis=0)

@register_op
class VariationalModeDecompositionOp(ExpandOp):
    """
    Decomposes a signal into a discrete number of quasi-orthogonal modes using Variational Mode Decomposition (VMD).
    """
    op_name: ClassVar[str] = "vmd"
    description: ClassVar[str] = "Decomposes a signal into modes using Variational Mode Decomposition (VMD)."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, L, K, C)" # K is the number of modes

    alpha: int = Field(2000, description="Balancing parameter of the data-fidelity constraint.")
    tau: float = Field(0., description="Time-step of the dual ascent.")
    K: int = Field(..., description="The number of modes to be recovered.")
    DC: bool = Field(False, description="True if the first mode is put and kept at DC.")
    init: int = Field(1, description="0: all omegas start at 0; 1: all omegas start uniformly distributed; 2: all omegas initialized randomly.")
    tol: float = Field(1e-7, description="Tolerance of convergence criterion.")

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        try:
            from vmdpy import VMD
        except ImportError:
            raise ImportError("vmdpy is not installed. Please install it with 'pip install vmdpy'.")

        if x.ndim != 3:
            raise ValueError(f"Input for VMD must be 3D (B, L, C), but got {x.ndim}D.")

        batch_size, _, channels = x.shape
        vmd_results = []
        for i in range(batch_size):
            channel_results = []
            for j in range(channels):
                u, _, _ = VMD(x[i, :, j], self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
                channel_results.append(u.T) # VMD returns (K, L), transpose to (L, K)
            vmd_results.append(np.stack(channel_results, axis=-1))
        
        return np.stack(vmd_results, axis=0)


@register_op
class EmpiricalModeDecompositionOp(ExpandOp):
    """
    Decomposes a signal into Intrinsic Mode Functions (IMFs) using Empirical Mode Decomposition (EMD).
    """
    op_name: ClassVar[str] = "emd"
    description: ClassVar[str] = "Decomposes a signal into Intrinsic Mode Functions (IMFs)."
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, L, I, C)" # I is the number of IMFs

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        try:
            import emd
        except ImportError:
            raise ImportError("emd is not installed. Please install it with 'pip install emd'.")

        if x.ndim != 3:
            raise ValueError(f"Input for EMD must be 3D (B, L, C), but got {x.ndim}D.")

        batch_size, _, channels = x.shape
        imf_results = []
        for i in range(batch_size):
            channel_results = []
            for j in range(channels):
                imfs = emd.sift.sift(x[i, :, j])
                channel_results.append(imfs)
            # Pad IMFs to the same length
            max_imfs = max(imf.shape[1] for imf in channel_results)
            padded_channels = [np.pad(imf, ((0,0), (0, max_imfs - imf.shape[1])), 'constant') for imf in channel_results]
            imf_results.append(np.stack(padded_channels, axis=-1))
        
        return np.stack(imf_results, axis=0)


if __name__ == "__main__":
    print("--- Testing expand_schemas.py ---")
    
    # Create a dummy signal: Batch=1, Length=8192, Channels=1
    fs = 2048
    L = 8192
    t = np.linspace(0, L/fs, L, endpoint=False)
    dummy_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
    dummy_signal = dummy_signal[np.newaxis, :, np.newaxis] # Shape it to (1, 8192, 1)

    # 1. Test PatchOp
    print("\n1. Testing PatchOp...")
    patch_op = PatchOp(patch_size=1024, stride=512)
    patches = patch_op.execute(dummy_signal)
    print(f"Input shape: {dummy_signal.shape}")
    print(f"PatchOp output shape: {patches.shape}")
    assert patches.shape == (1, 15, 1024, 1)

    # 2. Test STFTOp
    print("\n2. Testing STFTOp...")
    stft_op = STFTOp(fs=fs, nperseg=256, noverlap=128)
    stft_result = stft_op.execute(dummy_signal)
    print(f"STFTOp output shape: {stft_result.shape}")
    assert stft_result.shape[0] == 1 and stft_result.shape[3] == 1

    # 3. Test VariableQTransformOp
    print("\n3. Testing VariableQTransformOp...")
    vqt_op = VariableQTransformOp(fs=fs, hop_length=256, fmin=20, n_bins=48)
    vqt_result = vqt_op.execute(dummy_signal)
    print(f"VQT output shape: {vqt_result.shape}")
    assert vqt_result.shape == (1, 48, 33, 1)

    # 4. Test TimeDelayEmbeddingOp
    print("\n4. Testing TimeDelayEmbeddingOp...")
    tde_op = TimeDelayEmbeddingOp(dimension=3, delay=4)
    tde_result = tde_op.execute(dummy_signal)
    print(f"TimeDelayEmbeddingOp output shape: {tde_result.shape}")
    assert tde_result.shape == (1, 8192 - (3-1)*4, 3, 1)

    # 5. Test EmpiricalModeDecompositionOp
    print("\n5. Testing EmpiricalModeDecompositionOp...")
    emd_op = EmpiricalModeDecompositionOp()
    emd_result = emd_op.execute(dummy_signal)
    print(f"EMD output shape: {emd_result.shape}")
    assert emd_result.shape[0] == 1 and emd_result.shape[1] == dummy_signal.shape[1]

    # 6. Test VariationalModeDecompositionOp
    print("\n6. Testing VariationalModeDecompositionOp...")
    vmd_op = VariationalModeDecompositionOp(K=5)
    vmd_result = vmd_op.execute(dummy_signal)
    print(f"VMD output shape: {vmd_result.shape}")
    assert vmd_result.shape == (1, 8192, 5, 1)

    print("\n--- expand_schemas.py tests passed! ---")
