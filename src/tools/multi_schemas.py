from __future__ import annotations

from typing import ClassVar, Dict, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy import signal

from .signal_processing_schemas import MultiVariableOp, register_op


@register_op
class SubtractOp(MultiVariableOp):
    """
    Subtracts one signal from another, element-wise.
    Requires two inputs named 'minuend' and 'subtrahend'.
    Output = minuend - subtrahend.
    """
    op_name: ClassVar[str] = "subtract"
    description: ClassVar[str] = "Subtracts the 'subtrahend' signal from the 'minuend' signal."
    input_spec: ClassVar[str] = "minuend: (..., L, C), subtrahend: (..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"SubtractOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        minuend, subtrahend = values[0], values[1]
        return minuend - subtrahend


@register_op
class CrossCorrelationOp(MultiVariableOp):
    """
    Computes the cross-correlation between two signals.
    Requires two inputs named 'signal1' and 'signal2'.
    """
    op_name: ClassVar[str] = "cross_correlation"
    description: ClassVar[str] = "Computes the cross-correlation between two signals."
    input_spec: ClassVar[str] = "signal1: (..., L, C), signal2: (..., L, C)"
    output_spec: ClassVar[str] = "(..., L_corr, C)"
    mode: Literal["full", "valid", "same"] = Field("full", description="The mode of the correlation.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"CrossCorrelationOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        sig1, sig2 = values[0], values[1]

        # Assuming batch processing, correlation is applied on the last relevant axis (L)
        # We iterate through batch and channels
        if sig1.ndim == 3 and sig2.ndim == 3: # (B, L, C)
            batch_size, _, channels = sig1.shape
            results = []
            for i in range(batch_size):
                channel_results = []
                for j in range(channels):
                    corr = signal.correlate(sig1[i, :, j], sig2[i, :, j], mode=self.mode)
                    channel_results.append(corr)
                # This part needs careful handling of output shape, for now we stack
                # and assume the user knows the output shape will vary with 'mode'.
                results.append(np.stack(channel_results, axis=-1))
            return np.stack(results, axis=0)
        else: # Fallback for simpler shapes
            return signal.correlate(sig1, sig2, mode=self.mode)

@register_op
class DistanceOp(MultiVariableOp):
    """
    Computes the distance between two feature vectors.
    Requires two inputs named 'vec1' and 'vec2'.
    """
    op_name: ClassVar[str] = "distance"
    description: ClassVar[str] = "Computes the distance between two feature vectors."
    input_spec: ClassVar[str] = "vec1: (..., C'), vec2: (..., C')"
    output_spec: ClassVar[str] = "(...,)"
    metric: Literal["euclidean", "manhattan", "cosine"] = Field("euclidean", description="The distance metric to use.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"DistanceOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        vec1, vec2 = values[0], values[1]

        if self.metric == "euclidean":
            return np.linalg.norm(vec1 - vec2, axis=-1)
        elif self.metric == "manhattan":
            return np.sum(np.abs(vec1 - vec2), axis=-1)
        elif self.metric == "cosine":
            # Returns cosine distance, not similarity
            return 1 - np.sum(vec1 * vec2, axis=-1) / (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

@register_op
class ConcatenateOp(MultiVariableOp):
    """
    Concatenates multiple feature vectors along a specified axis.
    This op is special and will take all values from the input dict.
    """
    op_name: ClassVar[str] = "concatenate"
    description: ClassVar[str] = "Concatenates multiple feature vectors."
    input_spec: ClassVar[str] = "Multiple arrays in a dictionary."
    output_spec: ClassVar[str] = "(..., C_new)"
    axis: int = Field(-1, description="The axis along which the arrays will be joined.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if not x:
            raise ValueError("ConcatenateOp requires at least one input vector.")
        
        return np.concatenate(list(x.values()), axis=self.axis)

@register_op
class ElementWiseProductOp(MultiVariableOp):
    """
    Computes the element-wise product of two signals.
    Requires two inputs named 'signal1' and 'signal2'.
    """
    op_name: ClassVar[str] = "element_wise_product"
    description: ClassVar[str] = "Computes the element-wise product of two signals (Hadamard product)."
    input_spec: ClassVar[str] = "signal1: (..., L, C), signal2: (..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"ElementWiseProductOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        return values[0] * values[1]

@register_op
class CoherenceOp(MultiVariableOp):
    """
    Computes the coherence between two signals, indicating their linear relationship at different frequencies.
    Requires two inputs named 'signal1' and 'signal2'.
    """
    op_name: ClassVar[str] = "coherence"
    description: ClassVar[str] = "Computes the magnitude squared coherence between two signals."
    input_spec: ClassVar[str] = "signal1: (..., L, C), signal2: (..., L, C)"
    output_spec: ClassVar[str] = "(..., F, C)" # F is the number of frequency bins
    
    fs: float = Field(..., description="Sampling frequency of the signals.")
    nperseg: int = Field(256, description="Length of each segment for Welch's method used in coherence calculation.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"CoherenceOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        sig1, sig2 = values[0], values[1]

        if sig1.ndim != 3 or sig2.ndim != 3:
            raise ValueError(f"Inputs for CoherenceOp must be 3D (B, L, C).")

        # Transpose to (B, C, L) for scipy
        sig1_t = sig1.transpose(0, 2, 1)
        sig2_t = sig2.transpose(0, 2, 1)
        
        f, Cxy = signal.coherence(sig1_t, sig2_t, fs=self.fs, nperseg=self.nperseg, axis=-1)
        
        # Cxy shape is (B, C, F). We want (B, F, C).
        return Cxy.transpose(0, 2, 1)

@register_op
class ArithmeticOp(MultiVariableOp):
    """
    Performs a basic arithmetic operation between two signals.
    Requires two inputs named 'signal1' and 'signal2'.
    """
    op_name: ClassVar[str] = "arithmetic"
    description: ClassVar[str] = "Performs basic arithmetic (+, -, *, /) between two signals."
    input_spec: ClassVar[str] = "signal1: (..., L, C), signal2: (..., L, C)"
    output_spec: ClassVar[str] = "(..., L, C)"
    
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(..., description="The arithmetic operation to perform.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"ArithmeticOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        sig1, sig2 = values[0], values[1]
        
        if self.operation == "add":
            return sig1 + sig2
        elif self.operation == "subtract":
            return sig1 - sig2
        elif self.operation == "multiply":
            return sig1 * sig2
        elif self.operation == "divide":
            return sig1 / (sig2 + 1e-9) # Add epsilon for stability
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

@register_op
class PhaseDifferenceOp(MultiVariableOp):
    """
    Computes the phase difference between two signals in the frequency domain.
    Requires two complex-valued inputs named 'fft1' and 'fft2'.
    """
    op_name: ClassVar[str] = "phase_difference"
    description: ClassVar[str] = "Computes the phase difference between two FFTs."
    input_spec: ClassVar[str] = "fft1: (..., F, C), fft2: (..., F, C)"
    output_spec: ClassVar[str] = "(..., F, C)"

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"PhaseDifferenceOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        fft1, fft2 = values[0], values[1]
        
        # Phase difference is the angle of the conjugate product
        phase_diff = np.angle(fft1 * np.conj(fft2))
        return phase_diff

@register_op
class ConvolutionOp(MultiVariableOp):
    """
    Performs 1D convolution of two signals.
    Requires 'signal' and 'kernel' inputs.
    """
    op_name: ClassVar[str] = "convolution"
    description: ClassVar[str] = "Performs 1D convolution of a signal with a kernel."
    input_spec: ClassVar[str] = "signal: (..., L, C), kernel: (K,)"
    output_spec: ClassVar[str] = "(..., L', C)"
    mode: Literal["full", "valid", "same"] = Field("same", description="The mode of the convolution.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        if len(x) != 2:
            raise ValueError(f"ConvolutionOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        sig, kernel = values[0], values[1]

        if sig.ndim != 3 or kernel.ndim != 1:
            raise ValueError("ConvolutionOp requires a 3D signal and a 1D kernel.")

        # Apply convolution to each channel and batch item
        return signal.convolve(sig, kernel[np.newaxis, :, np.newaxis], mode=self.mode)

@register_op
class DynamicTimeWarpingOp(MultiVariableOp):
    """
    Computes the Dynamic Time Warping (DTW) distance between two time series.
    """
    op_name: ClassVar[str] = "dtw_distance"
    description: ClassVar[str] = "Computes Dynamic Time Warping (DTW) distance between two time series."
    input_spec: ClassVar[str] = "signal1: (L, C), signal2: (L, C)"
    output_spec: ClassVar[str] = "Scalar"

    def execute(self, x: Dict[str, npt.NDArray], **_) -> npt.NDArray:
        try:
            from dtaidistance import dtw
        except ImportError:
            raise ImportError("dtaidistance is not installed. Please install it with 'pip install dtaidistance'.")

        if len(x) != 2:
            raise ValueError(f"DTW requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        sig1, sig2 = values[0].squeeze(), values[1].squeeze()

        distance = dtw.distance(sig1, sig2)
        return np.array([distance])

@register_op
class TransferFunctionOp(MultiVariableOp):
    """
    Estimates the transfer function between an input and output signal.
    Requires 'input_signal' and 'output_signal'.
    """
    op_name: ClassVar[str] = "transfer_function"
    description: ClassVar[str] = "Estimates the transfer function H(f) = P_xy(f) / P_xx(f)."
    input_spec: ClassVar[str] = "input_signal: (L,), output_signal: (L,)"
    output_spec: ClassVar[str] = "Dict['frequencies', 'transfer_function']"
    
    fs: float = Field(..., description="Sampling frequency.")

    def execute(self, x: Dict[str, npt.NDArray], **_) -> Dict[str, npt.NDArray]:
        if len(x) != 2:
            raise ValueError(f"TransferFunctionOp requires exactly 2 inputs, but got {len(x)}.")
        
        values = list(x.values())
        input_sig, output_sig = values[0].squeeze(), values[1].squeeze()
        
        f, Pxx = signal.welch(input_sig, fs=self.fs)
        f, Pxy = signal.csd(input_sig, output_sig, fs=self.fs)
        
        H = Pxy / (Pxx + 1e-9)
        return {"frequencies": f, "transfer_function": H}


if __name__ == "__main__":
    print("--- Testing multi_schemas.py ---")

    # Create two dummy signals
    fs = 100
    L = 100
    t = np.linspace(0, L/fs, L, endpoint=False)
    sig1 = np.sin(2 * np.pi * 10 * t)[np.newaxis, :, np.newaxis]
    sig2 = np.cos(2 * np.pi * 10 * t)[np.newaxis, :, np.newaxis] # 90 degree phase shift

    # 1. Test ArithmeticOp
    print("\n1. Testing ArithmeticOp (subtract)...")
    arith_op = ArithmeticOp(operation="subtract")
    arith_result = arith_op.execute({"signal1": sig1, "signal2": sig2})
    print(f"Input shape: {sig1.shape}")
    print(f"ArithmeticOp output shape: {arith_result.shape}")
    assert arith_result.shape == sig1.shape

    # 2. Test PhaseDifferenceOp
    print("\n2. Testing PhaseDifferenceOp...")
    # Get complex FFTs first (not just magnitude)
    fft1 = np.fft.rfft(sig1, axis=-2)
    fft2 = np.fft.rfft(sig2, axis=-2)
    
    phase_op = PhaseDifferenceOp()
    phase_result = phase_op.execute({"fft1": fft1, "fft2": fft2})
    print(f"PhaseDifferenceOp output shape: {phase_result.shape}")
    assert phase_result.shape == fft1.shape
    # The phase difference at the 10Hz bin should be close to -pi/2 (-1.57)
    phase_at_10hz = phase_result[0, 10, 0]
    print(f"Phase difference at 10Hz: {phase_at_10hz:.2f} radians")
    assert np.isclose(phase_at_10hz, -np.pi/2, atol=0.1)

    # 3. Test ConvolutionOp
    print("\n3. Testing ConvolutionOp...")
    # Use a simple moving average kernel
    kernel = np.ones(5) / 5
    conv_op = ConvolutionOp(mode="same")
    conv_result = conv_op.execute({"signal": sig1, "kernel": kernel})
    print(f"ConvolutionOp output shape: {conv_result.shape}")
    assert conv_result.shape == sig1.shape

    # 4. Test DynamicTimeWarpingOp
    print("\n4. Testing DynamicTimeWarpingOp...")
    # Create two slightly different signals
    sig_dtw1 = np.sin(np.linspace(0, 20, 100))
    sig_dtw2 = np.sin(np.linspace(0, 20, 120)) # Different length
    dtw_op = DynamicTimeWarpingOp()
    dtw_result = dtw_op.execute({"signal1": sig_dtw1, "signal2": sig_dtw2})
    print(f"DTW distance: {dtw_result.item():.2f}")
    assert dtw_result.ndim == 1

    # 5. Test TransferFunctionOp
    print("\n5. Testing TransferFunctionOp...")
    # Create a simple system (e.g., a filter)
    system = ([1.0], [1.0, 0.5])
    input_sig = np.random.randn(1000)
    output_sig = signal.lfilter(system[0], system[1], input_sig)
    tf_op = TransferFunctionOp(fs=100)
    tf_result = tf_op.execute({"input_signal": input_sig, "output_signal": output_sig})
    print(f"Transfer function output keys: {tf_result.keys()}")
    assert "frequencies" in tf_result and "transfer_function" in tf_result

    print("\n--- multi_schemas.py tests passed! ---")
