"""
Interactive Operator Playground

Provides hands-on experimentation with PHMGA signal processing operators.
Users can interactively apply operators to signals and see results in real-time.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Dict, List, Any, Tuple, Optional, Callable
import time

# Add src/ to path for production system imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

try:
    from tools.signal_processing_schemas import OP_REGISTRY, list_available_operators
    from states.phm_states import InputData, ProcessedData
except ImportError:
    print("⚠️ Could not import production operators. Using demo mode.")
    OP_REGISTRY = {}


class SignalGenerator:
    """Generate various test signals for experimentation"""
    
    def __init__(self):
        self.signal_types = {
            'sine': self._generate_sine,
            'square': self._generate_square,
            'sawtooth': self._generate_sawtooth,
            'chirp': self._generate_chirp,
            'noise': self._generate_noise,
            'bearing_normal': self._generate_bearing_normal,
            'bearing_fault': self._generate_bearing_fault,
            'composite': self._generate_composite
        }
    
    def generate_signal(self, signal_type: str, duration: float = 1.0, 
                       sampling_rate: float = 10000, **kwargs) -> np.ndarray:
        """Generate a test signal of specified type"""
        
        if signal_type not in self.signal_types:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        t = np.linspace(0, duration, int(sampling_rate * duration))
        return self.signal_types[signal_type](t, **kwargs)
    
    def _generate_sine(self, t: np.ndarray, frequency: float = 60, 
                      amplitude: float = 1.0, phase: float = 0) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    def _generate_square(self, t: np.ndarray, frequency: float = 60, 
                        amplitude: float = 1.0) -> np.ndarray:
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    
    def _generate_sawtooth(self, t: np.ndarray, frequency: float = 60, 
                          amplitude: float = 1.0) -> np.ndarray:
        return amplitude * 2 * (frequency * t - np.floor(frequency * t + 0.5))
    
    def _generate_chirp(self, t: np.ndarray, f0: float = 10, f1: float = 100, 
                       amplitude: float = 1.0) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / t[-1]) * t)
    
    def _generate_noise(self, t: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        return amplitude * np.random.randn(len(t))
    
    def _generate_bearing_normal(self, t: np.ndarray, shaft_freq: float = 60, 
                                amplitude: float = 1.0, noise_level: float = 0.1) -> np.ndarray:
        signal = amplitude * np.sin(2 * np.pi * shaft_freq * t)
        signal += amplitude * 0.3 * np.sin(2 * np.pi * 2 * shaft_freq * t)  # 2nd harmonic
        signal += noise_level * np.random.randn(len(t))
        return signal
    
    def _generate_bearing_fault(self, t: np.ndarray, shaft_freq: float = 60, 
                               fault_freq: float = 157, amplitude: float = 1.0, 
                               noise_level: float = 0.1) -> np.ndarray:
        signal = amplitude * np.sin(2 * np.pi * shaft_freq * t)
        signal += amplitude * 0.5 * np.sin(2 * np.pi * fault_freq * t)  # Fault frequency
        signal += amplitude * 0.2 * np.sin(2 * np.pi * 2 * fault_freq * t)  # 2nd harmonic
        signal += noise_level * np.random.randn(len(t))
        return signal
    
    def _generate_composite(self, t: np.ndarray, **kwargs) -> np.ndarray:
        """Generate composite signal with multiple components"""
        components = kwargs.get('components', [
            {'type': 'sine', 'frequency': 60, 'amplitude': 1.0},
            {'type': 'sine', 'frequency': 157, 'amplitude': 0.5},
            {'type': 'noise', 'amplitude': 0.1}
        ])
        
        signal = np.zeros_like(t)
        for comp in components:
            comp_type = comp.pop('type')
            if comp_type in self.signal_types and comp_type != 'composite':
                signal += self.signal_types[comp_type](t, **comp)
        
        return signal
    
    def get_signal_info(self) -> Dict[str, str]:
        """Get information about available signal types"""
        return {
            'sine': 'Pure sinusoidal signal at specified frequency',
            'square': 'Square wave signal with sharp transitions',
            'sawtooth': 'Sawtooth wave with linear ramps',
            'chirp': 'Frequency sweep from f0 to f1',
            'noise': 'White Gaussian noise',
            'bearing_normal': 'Healthy bearing vibration (shaft frequency + harmonics)',
            'bearing_fault': 'Faulty bearing vibration (shaft + fault frequencies)',
            'composite': 'Multiple signal components combined'
        }


class OperatorPlayground:
    """Interactive playground for signal processing operators"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.signal_generator = SignalGenerator()
        self.current_signal = None
        self.current_result = None
        self.sampling_rate = 10000
        self.duration = 1.0
        
        # Available operators from production system
        self.available_operators = self._get_available_operators()
        
        # Playground state
        self.signal_history = []
        self.operation_log = []
    
    def _get_available_operators(self) -> Dict[str, Any]:
        """Get available operators from production system"""
        
        operators = {}
        
        if OP_REGISTRY:
            # Use real production operators
            operator_info = list_available_operators()
            for category, ops in operator_info.items():
                operators[category] = {}
                for op_name in ops:
                    op_class = OP_REGISTRY.get(op_name)
                    if op_class:
                        operators[category][op_name] = {
                            'class': op_class,
                            'description': getattr(op_class, 'description', 'Signal processing operator')
                        }
        else:
            # Demo operators when production system not available
            operators = {
                'TRANSFORM': {
                    'fft': {'description': 'Fast Fourier Transform'},
                    'filter_lowpass': {'description': 'Low-pass filter'},
                    'filter_highpass': {'description': 'High-pass filter'},
                    'envelope': {'description': 'Signal envelope extraction'}
                },
                'AGGREGATE': {
                    'statistics': {'description': 'Statistical features (mean, std, rms)'},
                    'spectral_features': {'description': 'Spectral domain features'},
                    'time_features': {'description': 'Time domain features'}
                },
                'EXPAND': {
                    'windowing': {'description': 'Signal windowing/segmentation'},
                    'overlapping_windows': {'description': 'Overlapping window analysis'}
                }
            }
        
        return operators
    
    def create_jupyter_interface(self):
        """Create interactive Jupyter notebook interface"""
        
        print("🎮 INTERACTIVE OPERATOR PLAYGROUND")
        print("=" * 45)
        print("Experiment with signal processing operators in real-time!")
        
        # Signal generation controls
        signal_type_widget = widgets.Dropdown(
            options=list(self.signal_generator.signal_types.keys()),
            value='bearing_normal',
            description='Signal Type:'
        )
        
        frequency_widget = widgets.FloatSlider(
            value=60.0, min=1.0, max=200.0, step=1.0,
            description='Frequency (Hz):'
        )
        
        amplitude_widget = widgets.FloatSlider(
            value=1.0, min=0.1, max=2.0, step=0.1,
            description='Amplitude:'
        )
        
        noise_widget = widgets.FloatSlider(
            value=0.1, min=0.0, max=0.5, step=0.05,
            description='Noise Level:'
        )
        
        generate_button = widgets.Button(
            description='Generate Signal',
            button_style='success',
            icon='play'
        )
        
        # Operator selection
        category_widget = widgets.Dropdown(
            options=list(self.available_operators.keys()),
            value=list(self.available_operators.keys())[0] if self.available_operators else 'TRANSFORM',
            description='Operator Category:'
        )
        
        operator_widget = widgets.Dropdown(
            options=[],
            description='Operator:'
        )
        
        apply_button = widgets.Button(
            description='Apply Operator',
            button_style='info',
            icon='cog'
        )
        
        clear_button = widgets.Button(
            description='Clear History',
            button_style='warning',
            icon='trash'
        )
        
        # Output area
        output_area = widgets.Output()
        
        # Event handlers
        def update_operators(*args):
            category = category_widget.value
            if category in self.available_operators:
                operators = list(self.available_operators[category].keys())
                operator_widget.options = operators
                if operators:
                    operator_widget.value = operators[0]
        
        def generate_signal(*args):
            with output_area:
                clear_output(wait=True)
                try:
                    # Generate signal
                    signal = self.signal_generator.generate_signal(
                        signal_type_widget.value,
                        duration=self.duration,
                        sampling_rate=self.sampling_rate,
                        frequency=frequency_widget.value,
                        amplitude=amplitude_widget.value,
                        noise_level=noise_widget.value,
                        shaft_freq=frequency_widget.value,
                        fault_freq=frequency_widget.value * 2.62  # Typical bearing fault ratio
                    )
                    
                    self.current_signal = signal
                    self.signal_history.append({
                        'signal': signal.copy(),
                        'type': signal_type_widget.value,
                        'parameters': {
                            'frequency': frequency_widget.value,
                            'amplitude': amplitude_widget.value,
                            'noise_level': noise_widget.value
                        },
                        'timestamp': time.time()
                    })
                    
                    self._plot_signal(signal, f\"Generated {signal_type_widget.value} signal\")\
                    print(f\"✅ Generated {signal_type_widget.value} signal\")\n",
                    print(f\"   Signal length: {len(signal)} samples\")\n",
                    print(f\"   Duration: {self.duration}s at {self.sampling_rate}Hz\")\n",
                    print(f\"   RMS: {np.sqrt(np.mean(signal**2)):.3f}\")\n",
                    \n",
                except Exception as e:\n",
                    print(f\"❌ Error generating signal: {e}\")\n",
        \n",
        def apply_operator(*args):\n",
            with output_area:\n",
                clear_output(wait=True)\n",
                \n",
                if self.current_signal is None:\n",
                    print(\"⚠️ Please generate a signal first!\")\n",
                    return\n",
                \n",
                try:\n",
                    category = category_widget.value\n",
                    operator_name = operator_widget.value\n",
                    \n",
                    # Apply operator (demo version)\n",
                    result = self._apply_demo_operator(\n",
                        self.current_signal, category, operator_name\n",
                    )\n",
                    \n",
                    if result is not None:\n",
                        self.current_result = result\n",
                        self.operation_log.append({\n",
                            'operator': f\"{category}.{operator_name}\",\n",
                            'input_shape': self.current_signal.shape,\n",
                            'output_shape': result.shape,\n",
                            'timestamp': time.time()\n",
                        })\n",
                        \n",
                        self._plot_operator_result(\n",
                            self.current_signal, result, \n",
                            f\"{category}.{operator_name}\"\n",
                        )\n",
                        \n",
                        print(f\"✅ Applied {category}.{operator_name}\")\n",
                        print(f\"   Input shape: {self.current_signal.shape}\")\n",
                        print(f\"   Output shape: {result.shape}\")\n",
                        \n",
                        # Update current signal to result for chaining\n",
                        self.current_signal = result\n",
                        \n",
                    else:\n",
                        print(f\"❌ Failed to apply {operator_name}\")\n",
                        \n",
                except Exception as e:\n",
                    print(f\"❌ Error applying operator: {e}\")\n",
        \n",
        def clear_history(*args):\n",
            with output_area:\n",
                clear_output(wait=True)\n",
                self.signal_history.clear()\n",
                self.operation_log.clear()\n",
                self.current_signal = None\n",
                self.current_result = None\n",
                print(\"🗑️ History cleared!\")\n",
        \n",
        # Connect event handlers\n",
        category_widget.observe(update_operators, names='value')\n",
        generate_button.on_click(generate_signal)\n",
        apply_button.on_click(apply_operator)\n",
        clear_button.on_click(clear_history)\n",
        \n",
        # Initialize operators dropdown\n",
        update_operators()\n",
        \n",
        # Display interface\n",
        signal_controls = widgets.VBox([\n",
            widgets.HTML(\"<h3>🎵 Signal Generation</h3>\"),\n",
            signal_type_widget,\n",
            frequency_widget,\n",
            amplitude_widget,\n",
            noise_widget,\n",
            generate_button\n",
        ])\n",
        \n",
        operator_controls = widgets.VBox([\n",
            widgets.HTML(\"<h3>⚙️ Operator Application</h3>\"),\n",
            category_widget,\n",
            operator_widget,\n",
            apply_button,\n",
            clear_button\n",
        ])\n",
        \n",
        controls = widgets.HBox([signal_controls, operator_controls])\n",
        \n",
        display(controls)\n",
        display(output_area)\n",
        \n",
        # Show initial help\n",
        with output_area:\n",
            print(\"🎮 Welcome to the Operator Playground!\")\n",
            print(\"\\n📋 Instructions:\")\n",
            print(\"   1. Select signal type and adjust parameters\")\n",
            print(\"   2. Click 'Generate Signal' to create test signal\")\n",
            print(\"   3. Choose operator category and specific operator\")\n",
            print(\"   4. Click 'Apply Operator' to process the signal\")\n",
            print(\"   5. Results become new input for operator chaining\")\n",
            print(\"\\n💡 Available Signal Types:\")\n",
            for sig_type, description in self.signal_generator.get_signal_info().items():\n",
                print(f\"   • {sig_type}: {description}\")\n",
            print(f\"\\n🔧 Available Operator Categories: {len(self.available_operators)}\")\n",
            for category in self.available_operators.keys():\n",
                print(f\"   • {category}\")\n",
    \n",
    def _apply_demo_operator(self, signal: np.ndarray, category: str, operator_name: str) -> Optional[np.ndarray]:\n",
        \"\"\"Apply demo version of operator (when production operators not available)\"\"\"\n",
        \n",
        if OP_REGISTRY and operator_name in OP_REGISTRY:\n",
            # Use real production operator (would need proper implementation)\n",
            try:\n",
                op_class = OP_REGISTRY[operator_name]\n",
                # This would need proper operator instantiation and execution\n",
                # For now, return demo result\n",
                return self._demo_operator_implementation(signal, category, operator_name)\n",
            except Exception as e:\n",
                print(f\"Production operator failed: {e}\")\n",
                return self._demo_operator_implementation(signal, category, operator_name)\n",
        else:\n",
            # Use demo implementation\n",
            return self._demo_operator_implementation(signal, category, operator_name)\n",
    \n",
    def _demo_operator_implementation(self, signal: np.ndarray, category: str, operator_name: str) -> Optional[np.ndarray]:\n",
        \"\"\"Demo implementations of common operators\"\"\"\n",
        \n",
        if category == 'TRANSFORM':\n",
            if operator_name == 'fft':\n",
                # Return magnitude spectrum\n",
                fft_result = np.fft.fft(signal)\n",
                return np.abs(fft_result[:len(fft_result)//2])\n",
            \n",
            elif operator_name in ['filter_lowpass', 'filter_highpass']:\n",
                # Simple filtering using convolution\n",
                from scipy import signal as scipy_signal\n",
                if operator_name == 'filter_lowpass':\n",
                    b, a = scipy_signal.butter(4, 0.1, btype='low')\n",
                else:\n",
                    b, a = scipy_signal.butter(4, 0.1, btype='high')\n",
                try:\n",
                    return scipy_signal.filtfilt(b, a, signal)\n",
                except:\n",
                    # Fallback simple filter\n",
                    kernel = np.ones(10) / 10  # Simple moving average\n",
                    return np.convolve(signal, kernel, mode='same')\n",
            \n",
            elif operator_name == 'envelope':\n",
                # Signal envelope using Hilbert transform\n",
                try:\n",
                    from scipy.signal import hilbert\n",
                    return np.abs(hilbert(signal))\n",
                except:\n",
                    # Fallback envelope\n",
                    return np.abs(signal)\n",
        \n",
        elif category == 'AGGREGATE':\n",
            if operator_name == 'statistics':\n",
                # Return statistical features as array\n",
                stats = np.array([\n",
                    np.mean(signal),\n",
                    np.std(signal),\n",
                    np.sqrt(np.mean(signal**2)),  # RMS\n",
                    np.max(signal),\n",
                    np.min(signal)\n",
                ])\n",
                return stats\n",
            \n",
            elif operator_name == 'spectral_features':\n",
                # Spectral features\n",
                fft_mag = np.abs(np.fft.fft(signal))[:len(signal)//2]\n",
                freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)[:len(signal)//2]\n",
                \n",
                spectral_features = np.array([\n",
                    np.sum(fft_mag),  # Total power\n",
                    freqs[np.argmax(fft_mag)],  # Dominant frequency\n",
                    np.sum(freqs * fft_mag) / np.sum(fft_mag),  # Spectral centroid\n",
                    np.sqrt(np.sum((freqs - np.sum(freqs * fft_mag) / np.sum(fft_mag))**2 * fft_mag) / np.sum(fft_mag))  # Spectral spread\n",
                ])\n",
                return spectral_features\n",
            \n",
            elif operator_name == 'time_features':\n",
                # Time domain features\n",
                time_features = np.array([\n",
                    np.mean(np.abs(signal)),  # Mean absolute value\n",
                    np.sqrt(np.mean(signal**2)),  # RMS\n",
                    np.max(signal) - np.min(signal),  # Peak-to-peak\n",
                    len(signal[1:][np.diff(signal > 0)])  # Zero crossings\n",
                ])\n",
                return time_features\n",
        \n",
        elif category == 'EXPAND':\n",
            if operator_name == 'windowing':\n",
                # Simple windowing - return first half\n",
                window_size = len(signal) // 2\n",
                return signal[:window_size]\n",
            \n",
            elif operator_name == 'overlapping_windows':\n",
                # Return overlapping windows as 2D array\n",
                window_size = min(256, len(signal) // 4)\n",
                step_size = window_size // 2\n",
                windows = []\n",
                for i in range(0, len(signal) - window_size, step_size):\n",
                    windows.append(signal[i:i + window_size])\n",
                return np.array(windows) if windows else signal.reshape(1, -1)\n",
        \n",
        # Default: return original signal\n",
        print(f\"⚠️ Demo implementation not available for {category}.{operator_name}\")\n",
        return signal\n",
    \n",
    def _plot_signal(self, signal: np.ndarray, title: str):\n",
        \"\"\"Plot signal in time and frequency domain\"\"\"\n",
        \n",
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)\n",
        \n",
        # Time domain\n",
        t = np.linspace(0, len(signal) / self.sampling_rate, len(signal))\n",
        ax1.plot(t, signal, 'b-', linewidth=1)\n",
        ax1.set_xlabel('Time (s)')\n",
        ax1.set_ylabel('Amplitude')\n",
        ax1.set_title(f'{title} - Time Domain')\n",
        ax1.grid(True, alpha=0.3)\n",
        \n",
        # Show only first 0.2 seconds for clarity\n",
        if len(t) > self.sampling_rate * 0.2:\n",
            ax1.set_xlim(0, 0.2)\n",
        \n",
        # Frequency domain\n",
        f = np.fft.fftfreq(len(signal), 1/self.sampling_rate)[:len(signal)//2]\n",
        fft_signal = np.fft.fft(signal)\n",
        magnitude = np.abs(fft_signal[:len(signal)//2])\n",
        \n",
        ax2.plot(f, magnitude, 'r-', linewidth=1)\n",
        ax2.set_xlabel('Frequency (Hz)')\n",
        ax2.set_ylabel('Magnitude')\n",
        ax2.set_title(f'{title} - Frequency Domain')\n",
        ax2.grid(True, alpha=0.3)\n",
        ax2.set_xlim(0, min(500, max(f)))\n",
        \n",
        plt.tight_layout()\n",
        plt.show()\n",
    \n",
    def _plot_operator_result(self, input_signal: np.ndarray, output_signal: np.ndarray, operator_name: str):\n",
        \"\"\"Plot operator input and output\"\"\"\n",
        \n",
        if len(output_signal.shape) > 1:\n",
            # 2D output (e.g., windowed signal)\n",
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)\n",
            \n",
            # Input signal time domain\n",
            t_in = np.linspace(0, len(input_signal) / self.sampling_rate, len(input_signal))\n",
            axes[0, 0].plot(t_in, input_signal, 'b-', linewidth=1)\n",
            axes[0, 0].set_title('Input Signal - Time Domain')\n",
            axes[0, 0].set_xlabel('Time (s)')\n",
            axes[0, 0].set_ylabel('Amplitude')\n",
            axes[0, 0].grid(True, alpha=0.3)\n",
            \n",
            # Input signal frequency domain\n",
            f_in = np.fft.fftfreq(len(input_signal), 1/self.sampling_rate)[:len(input_signal)//2]\n",
            fft_in = np.abs(np.fft.fft(input_signal)[:len(input_signal)//2])\n",
            axes[0, 1].plot(f_in, fft_in, 'b-', linewidth=1)\n",
            axes[0, 1].set_title('Input Signal - Frequency Domain')\n",
            axes[0, 1].set_xlabel('Frequency (Hz)')\n",
            axes[0, 1].set_ylabel('Magnitude')\n",
            axes[0, 1].grid(True, alpha=0.3)\n",
            axes[0, 1].set_xlim(0, min(500, max(f_in)))\n",
            \n",
            # Output as 2D plot (spectrogram-like)\n",
            im = axes[1, 0].imshow(output_signal, aspect='auto', cmap='viridis')\n",
            axes[1, 0].set_title(f'{operator_name} Output (2D)')\n",
            axes[1, 0].set_xlabel('Sample')\n",
            axes[1, 0].set_ylabel('Window/Feature')\n",
            plt.colorbar(im, ax=axes[1, 0])\n",
            \n",
            # Output summary statistics\n",
            axes[1, 1].bar(range(min(10, output_signal.shape[1])), \n",
                          np.mean(output_signal, axis=0)[:10])\n",
            axes[1, 1].set_title('Output Features (Mean across windows)')\n",
            axes[1, 1].set_xlabel('Feature Index')\n",
            axes[1, 1].set_ylabel('Mean Value')\n",
            axes[1, 1].grid(True, alpha=0.3)\n",
            \n",
        else:\n",
            # 1D output\n",
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)\n",
            \n",
            # Input signal\n",
            t_in = np.linspace(0, len(input_signal) / self.sampling_rate, len(input_signal))\n",
            axes[0, 0].plot(t_in, input_signal, 'b-', linewidth=1, label='Input')\n",
            axes[0, 0].set_title('Input Signal')\n",
            axes[0, 0].set_xlabel('Time (s)')\n",
            axes[0, 0].set_ylabel('Amplitude')\n",
            axes[0, 0].grid(True, alpha=0.3)\n",
            axes[0, 0].legend()\n",
            \n",
            # Output signal\n",
            if len(output_signal) == len(input_signal):\n",
                # Same length - time domain plot\n",
                t_out = t_in\n",
                axes[0, 1].plot(t_out, output_signal, 'r-', linewidth=1, label='Output')\n",
                axes[0, 1].set_xlabel('Time (s)')\n",
            else:\n",
                # Different length - could be frequency domain or features\n",
                axes[0, 1].plot(output_signal, 'r-', linewidth=1, label='Output')\n",
                axes[0, 1].set_xlabel('Index')\n",
                \n",
            axes[0, 1].set_title(f'{operator_name} Output')\n",
            axes[0, 1].set_ylabel('Value')\n",
            axes[0, 1].grid(True, alpha=0.3)\n",
            axes[0, 1].legend()\n",
            \n",
            # Comparison (overlay if same length)\n",
            if len(output_signal) == len(input_signal):\n",
                axes[1, 0].plot(t_in, input_signal, 'b-', alpha=0.7, label='Input', linewidth=1)\n",
                axes[1, 0].plot(t_in, output_signal, 'r-', alpha=0.9, label='Output', linewidth=2)\n",
                axes[1, 0].set_title('Input vs Output Comparison')\n",
                axes[1, 0].set_xlabel('Time (s)')\n",
                axes[1, 0].set_ylabel('Amplitude')\n",
                axes[1, 0].legend()\n",
                axes[1, 0].grid(True, alpha=0.3)\n",
            else:\n",
                # Show histograms for different lengths\n",
                axes[1, 0].hist(input_signal, bins=30, alpha=0.7, label='Input', density=True)\n",
                if len(output_signal) > 1:\n",
                    axes[1, 0].hist(output_signal, bins=min(30, len(output_signal)), alpha=0.7, label='Output', density=True)\n",
                axes[1, 0].set_title('Distribution Comparison')\n",
                axes[1, 0].set_xlabel('Value')\n",
                axes[1, 0].set_ylabel('Density')\n",
                axes[1, 0].legend()\n",
                axes[1, 0].grid(True, alpha=0.3)\n",
            \n",
            # Statistics comparison\n",
            input_stats = [np.mean(input_signal), np.std(input_signal), np.sqrt(np.mean(input_signal**2))]\n",
            output_stats = [np.mean(output_signal), np.std(output_signal), np.sqrt(np.mean(output_signal**2))]\n",
            \n",
            x = np.arange(3)\n",
            width = 0.35\n",
            axes[1, 1].bar(x - width/2, input_stats, width, label='Input', alpha=0.7)\n",
            axes[1, 1].bar(x + width/2, output_stats, width, label='Output', alpha=0.7)\n",
            axes[1, 1].set_title('Statistics Comparison')\n",
            axes[1, 1].set_xlabel('Metric')\n",
            axes[1, 1].set_ylabel('Value')\n",
            axes[1, 1].set_xticks(x)\n",
            axes[1, 1].set_xticklabels(['Mean', 'Std', 'RMS'])\n",
            axes[1, 1].legend()\n",
            axes[1, 1].grid(True, alpha=0.3)\n",
        \n",
        plt.suptitle(f'Operator: {operator_name}', fontsize=16, fontweight='bold')\n",
        plt.tight_layout()\n",
        plt.show()\n",
    \n",
    def create_standalone_demo(self):\n",
        \"\"\"Create standalone demo for non-Jupyter environments\"\"\"\n",
        \n",
        print(\"🎮 PHMGA OPERATOR PLAYGROUND - STANDALONE DEMO\")\n",
        print(\"=\" * 55)\n",
        \n",
        # Generate demo signals\n",
        print(\"\\n📡 Generating demo signals...\")\n",
        signals = {}\n",
        for signal_type in ['bearing_normal', 'bearing_fault', 'sine', 'noise']:\n",
            signals[signal_type] = self.signal_generator.generate_signal(\n",
                signal_type, duration=1.0, sampling_rate=10000,\n",
                frequency=60, amplitude=1.0, noise_level=0.1\n",
            )\n",
        \n",
        # Show signal info\n",
        for name, signal in signals.items():\n",
            print(f\"   • {name}: {len(signal)} samples, RMS: {np.sqrt(np.mean(signal**2)):.3f}\")\n",
        \n",
        # Demonstrate operators\n",
        print(\"\\n⚙️ Demonstrating operators...\")\n",
        \n",
        test_signal = signals['bearing_fault']\n",
        \n",
        operators_to_demo = [\n",
            ('TRANSFORM', 'fft'),\n",
            ('AGGREGATE', 'statistics'),\n",
            ('AGGREGATE', 'spectral_features')\n",
        ]\n",
        \n",
        for category, op_name in operators_to_demo:\n",
            print(f\"\\n   Applying {category}.{op_name}...\")\n",
            result = self._apply_demo_operator(test_signal, category, op_name)\n",
            \n",
            if result is not None:\n",
                print(f\"      Input shape: {test_signal.shape}\")\n",
                print(f\"      Output shape: {result.shape}\")\n",
                if len(result) <= 10:\n",
                    print(f\"      Result: {result}\")\n",
                else:\n",
                    print(f\"      Result (first 5): {result[:5]}...\")\n",
        \n",
        print(\"\\n🎓 Demo completed! Use create_jupyter_interface() for interactive experience.\")\n",
        return signals\n",
\n",
\n",
def create_operator_playground(interface_type: str = 'jupyter') -> OperatorPlayground:\n",
    \"\"\"Create operator playground with specified interface type\"\"\"\n",
    \n",
    playground = OperatorPlayground()\n",
    \n",
    if interface_type == 'jupyter':\n",
        try:\n",
            playground.create_jupyter_interface()\n",
        except ImportError:\n",
            print(\"⚠️ Jupyter widgets not available, falling back to standalone demo\")\n",
            playground.create_standalone_demo()\n",
    else:\n",
        playground.create_standalone_demo()\n",
    \n",
    return playground\n",
\n",
\n",
if __name__ == \"__main__\":\n",
    # Create and run playground demo\n",
    playground = create_operator_playground('standalone')\n",
    \n",
    print(\"\\n💡 To use in Jupyter notebook:\")\n",
    print(\"   from operator_playground import create_operator_playground\")\n",
    print(\"   playground = create_operator_playground('jupyter')\")\n",