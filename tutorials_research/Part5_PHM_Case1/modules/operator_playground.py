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
                    
                    self._plot_signal(signal, f"Generated {signal_type_widget.value} signal")
                    print(f"✅ Generated {signal_type_widget.value} signal")
                    print(f"   Signal length: {len(signal)} samples")
                    print(f"   Duration: {self.duration}s at {self.sampling_rate}Hz")
                    print(f"   RMS: {np.sqrt(np.mean(signal**2)):.3f}")
                    
                except Exception as e:
                    print(f"❌ Error generating signal: {e}")
        
        def apply_operator(*args):
            with output_area:
                clear_output(wait=True)
                
                if self.current_signal is None:
                    print("⚠️ Please generate a signal first!")
                    return
                
                try:
                    category = category_widget.value
                    operator_name = operator_widget.value
                    
                    # Apply operator (demo version)
                    result = self._apply_demo_operator(
                        self.current_signal, category, operator_name
                    )
                    
                    if result is not None:
                        self.current_result = result
                        self.operation_log.append({
                            'operator': f"{category}.{operator_name}",
                            'input_shape': self.current_signal.shape,
                            'output_shape': result.shape,
                            'timestamp': time.time()
                        })
                        
                        self._plot_operator_result(
                            self.current_signal, result, 
                            f"{category}.{operator_name}"
                        )
                        
                        print(f"✅ Applied {category}.{operator_name}")
                        print(f"   Input shape: {self.current_signal.shape}")
                        print(f"   Output shape: {result.shape}")
                        
                        # Update current signal to result for chaining
                        self.current_signal = result
                        
                    else:
                        print(f"❌ Failed to apply {operator_name}")
                        
                except Exception as e:
                    print(f"❌ Error applying operator: {e}")
        
        def clear_history(*args):
            with output_area:
                clear_output(wait=True)
                self.signal_history.clear()
                self.operation_log.clear()
                self.current_signal = None
                self.current_result = None
                print("🗑️ History cleared!")
        
        # Connect event handlers
        category_widget.observe(update_operators, names='value')
        generate_button.on_click(generate_signal)
        apply_button.on_click(apply_operator)
        clear_button.on_click(clear_history)
        
        # Initialize operators dropdown
        update_operators()
        
        # Display interface
        signal_controls = widgets.VBox([
            widgets.HTML("<h3>🎵 Signal Generation</h3>"),
            signal_type_widget,
            frequency_widget,
            amplitude_widget,
            noise_widget,
            generate_button
        ])
        
        operator_controls = widgets.VBox([
            widgets.HTML("<h3>⚙️ Operator Application</h3>"),
            category_widget,
            operator_widget,
            apply_button,
            clear_button
        ])
        
        controls = widgets.HBox([signal_controls, operator_controls])
        
        display(controls)
        display(output_area)
        
        # Show initial help
        with output_area:
            print("🎮 Welcome to the Operator Playground!")
            print("\n📋 Instructions:")
            print("   1. Select signal type and adjust parameters")
            print("   2. Click 'Generate Signal' to create test signal")
            print("   3. Choose operator category and specific operator")
            print("   4. Click 'Apply Operator' to process the signal")
            print("   5. Results become new input for operator chaining")
            print("\n💡 Available Signal Types:")
            for sig_type, description in self.signal_generator.get_signal_info().items():
                print(f"   • {sig_type}: {description}")
            print(f"\n🔧 Available Operator Categories: {len(self.available_operators)}")
            for category in self.available_operators.keys():
                print(f"   • {category}")
    
    def _apply_demo_operator(self, signal: np.ndarray, category: str, operator_name: str) -> Optional[np.ndarray]:
        """Apply demo version of operator (when production operators not available)"""
        
        if OP_REGISTRY and operator_name in OP_REGISTRY:
            # Use real production operator (would need proper implementation)
            try:
                op_class = OP_REGISTRY[operator_name]
                # This would need proper operator instantiation and execution
                # For now, return demo result
                return self._demo_operator_implementation(signal, category, operator_name)
            except Exception as e:
                print(f"Production operator failed: {e}")
                return self._demo_operator_implementation(signal, category, operator_name)
        else:
            # Use demo implementation
            return self._demo_operator_implementation(signal, category, operator_name)
    
    def _demo_operator_implementation(self, signal: np.ndarray, category: str, operator_name: str) -> Optional[np.ndarray]:
        """Demo implementations of common operators"""
        
        if category == 'TRANSFORM':
            if operator_name == 'fft':
                # Return magnitude spectrum
                fft_result = np.fft.fft(signal)
                return np.abs(fft_result[:len(fft_result)//2])
            
            elif operator_name in ['filter_lowpass', 'filter_highpass']:
                # Simple filtering using convolution
                from scipy import signal as scipy_signal
                if operator_name == 'filter_lowpass':
                    b, a = scipy_signal.butter(4, 0.1, btype='low')
                else:
                    b, a = scipy_signal.butter(4, 0.1, btype='high')
                try:
                    return scipy_signal.filtfilt(b, a, signal)
                except:
                    # Fallback simple filter
                    kernel = np.ones(10) / 10  # Simple moving average
                    return np.convolve(signal, kernel, mode='same')
            
            elif operator_name == 'envelope':
                # Signal envelope using Hilbert transform
                try:
                    from scipy.signal import hilbert
                    return np.abs(hilbert(signal))
                except:
                    # Fallback envelope
                    return np.abs(signal)
        
        elif category == 'AGGREGATE':
            if operator_name == 'statistics':
                # Return statistical features as array
                stats = np.array([
                    np.mean(signal),
                    np.std(signal),
                    np.sqrt(np.mean(signal**2)),  # RMS
                    np.max(signal),
                    np.min(signal)
                ])
                return stats
            
            elif operator_name == 'spectral_features':
                # Spectral features
                fft_mag = np.abs(np.fft.fft(signal))[:len(signal)//2]
                freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)[:len(signal)//2]
                
                spectral_features = np.array([
                    np.sum(fft_mag),  # Total power
                    freqs[np.argmax(fft_mag)],  # Dominant frequency
                    np.sum(freqs * fft_mag) / np.sum(fft_mag),  # Spectral centroid
                    np.sqrt(np.sum((freqs - np.sum(freqs * fft_mag) / np.sum(fft_mag))**2 * fft_mag) / np.sum(fft_mag))  # Spectral spread
                ])
                return spectral_features
            
            elif operator_name == 'time_features':
                # Time domain features
                time_features = np.array([
                    np.mean(np.abs(signal)),  # Mean absolute value
                    np.sqrt(np.mean(signal**2)),  # RMS
                    np.max(signal) - np.min(signal),  # Peak-to-peak
                    len(signal[1:][np.diff(signal > 0)])  # Zero crossings
                ])
                return time_features
        
        elif category == 'EXPAND':
            if operator_name == 'windowing':
                # Simple windowing - return first half
                window_size = len(signal) // 2
                return signal[:window_size]
            
            elif operator_name == 'overlapping_windows':
                # Return overlapping windows as 2D array
                window_size = min(256, len(signal) // 4)
                step_size = window_size // 2
                windows = []
                for i in range(0, len(signal) - window_size, step_size):
                    windows.append(signal[i:i + window_size])
                return np.array(windows) if windows else signal.reshape(1, -1)
        
        # Default: return original signal
        print(f"⚠️ Demo implementation not available for {category}.{operator_name}")
        return signal
    
    def _plot_signal(self, signal: np.ndarray, title: str):
        """Plot signal in time and frequency domain"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Time domain
        t = np.linspace(0, len(signal) / self.sampling_rate, len(signal))
        ax1.plot(t, signal, 'b-', linewidth=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'{title} - Time Domain')
        ax1.grid(True, alpha=0.3)
        
        # Show only first 0.2 seconds for clarity
        if len(t) > self.sampling_rate * 0.2:
            ax1.set_xlim(0, 0.2)
        
        # Frequency domain
        f = np.fft.fftfreq(len(signal), 1/self.sampling_rate)[:len(signal)//2]
        fft_signal = np.fft.fft(signal)
        magnitude = np.abs(fft_signal[:len(signal)//2])
        
        ax2.plot(f, magnitude, 'r-', linewidth=1)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title(f'{title} - Frequency Domain')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(500, max(f)))
        
        plt.tight_layout()
        plt.show()
    
    def _plot_operator_result(self, input_signal: np.ndarray, output_signal: np.ndarray, operator_name: str):
        """Plot operator input and output"""
        
        if len(output_signal.shape) > 1:
            # 2D output (e.g., windowed signal)
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            
            # Input signal time domain
            t_in = np.linspace(0, len(input_signal) / self.sampling_rate, len(input_signal))
            axes[0, 0].plot(t_in, input_signal, 'b-', linewidth=1)
            axes[0, 0].set_title('Input Signal - Time Domain')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Input signal frequency domain
            f_in = np.fft.fftfreq(len(input_signal), 1/self.sampling_rate)[:len(input_signal)//2]
            fft_in = np.abs(np.fft.fft(input_signal)[:len(input_signal)//2])
            axes[0, 1].plot(f_in, fft_in, 'b-', linewidth=1)
            axes[0, 1].set_title('Input Signal - Frequency Domain')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Magnitude')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(0, min(500, max(f_in)))
            
            # Output as 2D plot (spectrogram-like)
            im = axes[1, 0].imshow(output_signal, aspect='auto', cmap='viridis')
            axes[1, 0].set_title(f'{operator_name} Output (2D)')
            axes[1, 0].set_xlabel('Sample')
            axes[1, 0].set_ylabel('Window/Feature')
            plt.colorbar(im, ax=axes[1, 0])
            
            # Output summary statistics
            axes[1, 1].bar(range(min(10, output_signal.shape[1])), 
                          np.mean(output_signal, axis=0)[:10])
            axes[1, 1].set_title('Output Features (Mean across windows)')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Mean Value')
            axes[1, 1].grid(True, alpha=0.3)
            
        else:
            # 1D output
            fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            
            # Input signal
            t_in = np.linspace(0, len(input_signal) / self.sampling_rate, len(input_signal))
            axes[0, 0].plot(t_in, input_signal, 'b-', linewidth=1, label='Input')
            axes[0, 0].set_title('Input Signal')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Output signal
            if len(output_signal) == len(input_signal):
                # Same length - time domain plot
                t_out = t_in
                axes[0, 1].plot(t_out, output_signal, 'r-', linewidth=1, label='Output')
                axes[0, 1].set_xlabel('Time (s)')
            else:
                # Different length - could be frequency domain or features
                axes[0, 1].plot(output_signal, 'r-', linewidth=1, label='Output')
                axes[0, 1].set_xlabel('Index')
                
            axes[0, 1].set_title(f'{operator_name} Output')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Comparison (overlay if same length)
            if len(output_signal) == len(input_signal):
                axes[1, 0].plot(t_in, input_signal, 'b-', alpha=0.7, label='Input', linewidth=1)
                axes[1, 0].plot(t_in, output_signal, 'r-', alpha=0.9, label='Output', linewidth=2)
                axes[1, 0].set_title('Input vs Output Comparison')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Amplitude')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                # Show histograms for different lengths
                axes[1, 0].hist(input_signal, bins=30, alpha=0.7, label='Input', density=True)
                if len(output_signal) > 1:
                    axes[1, 0].hist(output_signal, bins=min(30, len(output_signal)), alpha=0.7, label='Output', density=True)
                axes[1, 0].set_title('Distribution Comparison')
                axes[1, 0].set_xlabel('Value')
                axes[1, 0].set_ylabel('Density')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Statistics comparison
            input_stats = [np.mean(input_signal), np.std(input_signal), np.sqrt(np.mean(input_signal**2))]
            output_stats = [np.mean(output_signal), np.std(output_signal), np.sqrt(np.mean(output_signal**2))]
            
            x = np.arange(3)
            width = 0.35
            axes[1, 1].bar(x - width/2, input_stats, width, label='Input', alpha=0.7)
            axes[1, 1].bar(x + width/2, output_stats, width, label='Output', alpha=0.7)
            axes[1, 1].set_title('Statistics Comparison')
            axes[1, 1].set_xlabel('Metric')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(['Mean', 'Std', 'RMS'])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Operator: {operator_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_standalone_demo(self):
        """Create standalone demo for non-Jupyter environments"""
        
        print("🎮 PHMGA OPERATOR PLAYGROUND - STANDALONE DEMO")
        print("=" * 55)
        
        # Generate demo signals
        print("\n📡 Generating demo signals...")
        signals = {}
        for signal_type in ['bearing_normal', 'bearing_fault', 'sine', 'noise']:
            signals[signal_type] = self.signal_generator.generate_signal(
                signal_type, duration=1.0, sampling_rate=10000,
                frequency=60, amplitude=1.0, noise_level=0.1
            )
        
        # Show signal info
        for name, signal in signals.items():
            print(f"   • {name}: {len(signal)} samples, RMS: {np.sqrt(np.mean(signal**2)):.3f}")
        
        # Demonstrate operators
        print("\n⚙️ Demonstrating operators...")
        
        test_signal = signals['bearing_fault']
        
        operators_to_demo = [
            ('TRANSFORM', 'fft'),
            ('AGGREGATE', 'statistics'),
            ('AGGREGATE', 'spectral_features')
        ]
        
        for category, op_name in operators_to_demo:
            print(f"\n   Applying {category}.{op_name}...")
            result = self._apply_demo_operator(test_signal, category, op_name)
            
            if result is not None:
                print(f"      Input shape: {test_signal.shape}")
                print(f"      Output shape: {result.shape}")
                if len(result) <= 10:
                    print(f"      Result: {result}")
                else:
                    print(f"      Result (first 5): {result[:5]}...")
        
        print("\n🎓 Demo completed! Use create_jupyter_interface() for interactive experience.")
        return signals


def create_operator_playground(interface_type: str = 'jupyter') -> OperatorPlayground:
    """Create operator playground with specified interface type"""
    
    playground = OperatorPlayground()
    
    if interface_type == 'jupyter':
        try:
            playground.create_jupyter_interface()
        except ImportError:
            print("⚠️ Jupyter widgets not available, falling back to standalone demo")
            playground.create_standalone_demo()
    else:
        playground.create_standalone_demo()
    
    return playground


if __name__ == "__main__":
    # Create and run playground demo
    playground = create_operator_playground('standalone')
    
    print("\n💡 To use in Jupyter notebook:")
    print("   from operator_playground import create_operator_playground")
    print("   playground = create_operator_playground('jupyter')")