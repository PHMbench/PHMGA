"""
Visualization Tools for PHMGA Tutorial

Provides comprehensive visualization capabilities for DAG evolution,
signal processing, and system performance monitoring.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import networkx as nx
from datetime import datetime
import json

# Add src/ to path for production system imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

try:
    from states.phm_states import PHMState, DAGState, InputData, ProcessedData
    from agents.reflect_agent import get_dag_depth
except ImportError:
    print("⚠️ Could not import production components. Some features may be limited.")


class DAGVisualizer:
    """Visualize DAG structure and evolution"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.node_colors = {
            'InputData': '#E8F4FD',
            'ProcessedData': '#B8E6B8', 
            'SimilarityNode': '#FFE6CC',
            'DataSetNode': '#E6E6FA',
            'OutputNode': '#FFB6C1'
        }
        self.edge_colors = {
            'data_flow': '#2E86AB',
            'dependency': '#A23B72'
        }
    
    def visualize_dag_structure(self, dag_state: DAGState, title: str = "DAG Structure") -> plt.Figure:
        """Visualize the current DAG structure"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in dag_state.nodes.items():
            node_type = type(node).__name__
            G.add_node(node_id, node_type=node_type)
        
        # Add edges based on parent relationships
        for node_id, node in dag_state.nodes.items():
            parents = node.parents if isinstance(node.parents, list) else [node.parents] if node.parents else []
            for parent_id in parents:
                if parent_id in dag_state.nodes:
                    G.add_edge(parent_id, node_id)
        
        # Use hierarchical layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = {node: (i % 5, i // 5) for i, node in enumerate(G.nodes())}
        
        # Draw nodes by type
        for node_type, color in self.node_colors.items():
            node_list = [node for node, data in G.nodes(data=True) if data.get('node_type') == node_type]
            if node_list:
                nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                                     node_color=color, node_size=1000, 
                                     alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=self.edge_colors['data_flow'],
                             arrows=True, arrowsize=20, alpha=0.6, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Add legend
        legend_elements = [
            patches.Patch(color=color, label=node_type) 
            for node_type, color in self.node_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        ax.set_title(f"{title}\nNodes: {len(G.nodes())}, Depth: {get_dag_depth(dag_state)}", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_dag_evolution(self, evolution_history: List[Dict], title: str = "DAG Evolution") -> plt.Figure:
        """Visualize how the DAG evolves over iterations"""
        
        if not evolution_history:
            print("No evolution history to visualize")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract evolution data
        iterations = [item['iteration'] for item in evolution_history]
        node_counts = [item.get('nodes_after', item.get('nodes', 0)) for item in evolution_history]
        depths = [item['depth'] for item in evolution_history]
        nodes_added = [item.get('nodes_added', 0) for item in evolution_history]
        
        # Plot 1: Node count evolution
        ax1.plot(iterations, node_counts, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Nodes')
        ax1.set_title('DAG Size Growth')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(iterations)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(iterations, node_counts)):
            ax1.annotate(f'{y}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # Plot 2: Depth evolution
        ax2.plot(iterations, depths, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('DAG Depth')
        ax2.set_title('DAG Depth Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(iterations)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(iterations, depths)):
            ax2.annotate(f'{y}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # Plot 3: Nodes added per iteration
        colors = ['green' if x > 0 else 'orange' for x in nodes_added]
        bars = ax3.bar(iterations, nodes_added, color=colors, alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Nodes Added')
        ax3.set_title('Nodes Added per Iteration')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(iterations)
        
        # Add value labels on bars
        for bar, value in zip(bars, nodes_added):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{value}', ha='center', va='bottom')
        
        # Plot 4: Cumulative progress
        ax4.fill_between(iterations, node_counts, alpha=0.3, color='blue', label='Total Nodes')
        ax4.plot(iterations, depths, 'r-', linewidth=3, label='Depth')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Count/Depth')
        ax4.set_title('Cumulative Progress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(iterations)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_dag_animation(self, evolution_history: List[Dict], interval: int = 1000) -> FuncAnimation:
        """Create animated visualization of DAG evolution"""
        
        if not evolution_history:
            print("No evolution history for animation")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def animate(frame):
            ax.clear()
            
            # Get data for current frame
            if frame < len(evolution_history):
                data = evolution_history[frame]
                iteration = data['iteration']
                depth = data['depth']
                nodes = data.get('nodes_after', data.get('nodes', 0))
                
                # Create simple visualization for animation
                # This would need actual DAG structure data for full implementation
                ax.text(0.5, 0.7, f"Iteration {iteration}", ha='center', fontsize=20, fontweight='bold')
                ax.text(0.5, 0.5, f"Nodes: {nodes}", ha='center', fontsize=16)
                ax.text(0.5, 0.3, f"Depth: {depth}", ha='center', fontsize=16)
                
                # Simple progress bar
                progress = frame / len(evolution_history)
                ax.barh(0.1, progress, height=0.05, color='blue', alpha=0.7)
                ax.text(0.5, 0.05, f"Progress: {progress:.1%}", ha='center')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title("DAG Evolution Animation", fontsize=16, fontweight='bold')
        
        anim = FuncAnimation(fig, animate, frames=len(evolution_history), 
                           interval=interval, repeat=True)
        return anim


class SignalVisualizer:
    """Visualize signal processing and analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        self.figsize = figsize
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_signal_comparison(self, signals: Dict[str, np.ndarray], 
                                  sampling_rate: float = 10000,
                                  title: str = "Signal Comparison") -> plt.Figure:
        """Compare multiple signals in time and frequency domain"""
        
        if not signals:
            print("No signals provided for visualization")
            return None
        
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 2, figsize=(self.figsize[0], 3*n_signals))
        
        if n_signals == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_signals))
        
        for i, (signal_name, signal_data) in enumerate(signals.items()):
            if len(signal_data) == 0:
                continue
                
            # Time domain
            t = np.linspace(0, len(signal_data) / sampling_rate, len(signal_data))
            axes[i, 0].plot(t, signal_data, color=colors[i], linewidth=1)
            axes[i, 0].set_title(f"{signal_name.title()} - Time Domain")
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Show only first 0.2 seconds for clarity
            if len(t) > sampling_rate * 0.2:
                axes[i, 0].set_xlim(0, 0.2)
            
            # Frequency domain
            f = np.fft.fftfreq(len(signal_data), 1/sampling_rate)[:len(signal_data)//2]
            fft_signal = np.fft.fft(signal_data)
            magnitude = np.abs(fft_signal[:len(signal_data)//2])
            
            axes[i, 1].plot(f, magnitude, color=colors[i], linewidth=1)
            axes[i, 1].set_title(f"{signal_name.title()} - Frequency Domain")
            axes[i, 1].set_xlabel('Frequency (Hz)')
            axes[i, 1].set_ylabel('Magnitude')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_xlim(0, min(500, max(f)))  # Show up to 500 Hz
            
            # Add statistics
            stats_text = f"Mean: {np.mean(signal_data):.3f}\nStd: {np.std(signal_data):.3f}\nRMS: {np.sqrt(np.mean(signal_data**2)):.3f}"
            axes[i, 0].text(0.02, 0.98, stats_text, transform=axes[i, 0].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_signal_processing_pipeline(self, processing_steps: List[Dict]) -> plt.Figure:
        """Visualize signal processing pipeline steps"""
        
        if not processing_steps:
            print("No processing steps provided")
            return None
        
        n_steps = len(processing_steps)
        fig, axes = plt.subplots(n_steps, 1, figsize=(self.figsize[0], 3*n_steps))
        
        if n_steps == 1:
            axes = [axes]
        
        for i, step in enumerate(processing_steps):
            step_name = step.get('name', f'Step {i+1}')
            input_signal = step.get('input', np.array([]))
            output_signal = step.get('output', np.array([]))
            
            if len(input_signal) > 0 and len(output_signal) > 0:
                # Plot input and output
                x_in = np.arange(len(input_signal))
                x_out = np.arange(len(output_signal))
                
                axes[i].plot(x_in, input_signal, 'b-', alpha=0.7, label='Input', linewidth=1)
                axes[i].plot(x_out, output_signal, 'r-', alpha=0.9, label='Output', linewidth=2)
                
                axes[i].set_title(f"{step_name}: {step.get('description', '')}")
                axes[i].set_xlabel('Sample')
                axes[i].set_ylabel('Amplitude')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Show only first 1000 samples for clarity
                if max(len(input_signal), len(output_signal)) > 1000:
                    axes[i].set_xlim(0, 1000)
        
        plt.suptitle("Signal Processing Pipeline", fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_feature_extraction(self, features: Dict[str, np.ndarray], 
                                   feature_names: List[str] = None) -> plt.Figure:
        """Visualize extracted features"""
        
        if not features:
            print("No features provided for visualization")
            return None
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        feature_list = list(features.items())
        
        for i, (feature_name, feature_data) in enumerate(feature_list):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes[0]
            else:
                ax = axes[row, col] if n_cols > 1 else axes[row]
            
            if len(feature_data.shape) == 1:
                # 1D feature - plot as time series or histogram
                if len(feature_data) > 10:
                    ax.plot(feature_data, 'b-', linewidth=2)
                    ax.set_ylabel('Feature Value')
                    ax.set_xlabel('Index')
                else:
                    ax.bar(range(len(feature_data)), feature_data, alpha=0.7)
                    ax.set_ylabel('Feature Value')
                    ax.set_xlabel('Feature Index')
            elif len(feature_data.shape) == 2:
                # 2D feature - plot as heatmap
                im = ax.imshow(feature_data, aspect='auto', cmap='viridis')
                plt.colorbar(im, ax=ax)
            
            ax.set_title(feature_name)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes[0]
            else:
                ax = axes[row, col] if n_cols > 1 else axes[row]
            ax.set_visible(False)
        
        plt.suptitle("Extracted Features", fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


class PerformanceVisualizer:
    """Visualize system performance metrics"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        self.figsize = figsize
    
    def visualize_processing_times(self, processing_times: Dict[str, float],
                                 title: str = "Processing Times by Component") -> plt.Figure:
        """Visualize processing times for different components"""
        
        if not processing_times:
            print("No processing times provided")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        components = list(processing_times.keys())
        times = list(processing_times.values())
        
        # Bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        bars = ax1.bar(components, times, color=colors, alpha=0.8)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Processing Times')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time:.3f}s', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(times, labels=components, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Time Distribution')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_accuracy_metrics(self, confusion_matrix: np.ndarray, 
                                 class_names: List[str],
                                 accuracy: float = None,
                                 title: str = "Classification Performance") -> plt.Figure:
        """Visualize classification performance metrics"""
        
        if confusion_matrix is None or confusion_matrix.size == 0:
            print("No confusion matrix provided")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Confusion matrix heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title('Confusion Matrix')
        
        # Performance metrics
        if len(class_names) == confusion_matrix.shape[0]:
            # Calculate per-class metrics
            precision = []
            recall = []
            f1_score = []
            
            for i in range(len(class_names)):
                tp = confusion_matrix[i, i]
                fp = np.sum(confusion_matrix[:, i]) - tp
                fn = np.sum(confusion_matrix[i, :]) - tp
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                
                precision.append(prec)
                recall.append(rec)
                f1_score.append(f1)
            
            # Bar chart of metrics
            x = np.arange(len(class_names))
            width = 0.25
            
            ax2.bar(x - width, precision, width, label='Precision', alpha=0.8)
            ax2.bar(x, recall, width, label='Recall', alpha=0.8)
            ax2.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
            
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Score')
            ax2.set_title('Per-Class Metrics')
            ax2.set_xticks(x)
            ax2.set_xticklabels(class_names, rotation=45)
            ax2.legend()
            ax2.set_ylim(0, 1)
            
            # Add overall accuracy text
            if accuracy is not None:
                ax2.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.1%}',
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


class IntegratedVisualizer:
    """Main visualization interface combining all visualizers"""
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8), style: str = 'default'):
        self.figsize = figsize
        plt.style.use(style)
        
        self.dag_viz = DAGVisualizer(figsize)
        self.signal_viz = SignalVisualizer(figsize)
        self.perf_viz = PerformanceVisualizer(figsize)
    
    def create_tutorial_dashboard(self, tutorial_data: Dict[str, Any]) -> List[plt.Figure]:
        """Create comprehensive tutorial dashboard"""
        
        figures = []
        
        # DAG visualization
        if 'dag_state' in tutorial_data:
            fig = self.dag_viz.visualize_dag_structure(
                tutorial_data['dag_state'], 
                "Tutorial DAG Structure"
            )
            figures.append(("DAG Structure", fig))
        
        # DAG evolution
        if 'dag_evolution' in tutorial_data:
            fig = self.dag_viz.visualize_dag_evolution(
                tutorial_data['dag_evolution'],
                "Tutorial DAG Evolution"
            )
            figures.append(("DAG Evolution", fig))
        
        # Signal comparison
        if 'signals' in tutorial_data:
            fig = self.signal_viz.visualize_signal_comparison(
                tutorial_data['signals'],
                tutorial_data.get('sampling_rate', 10000),
                "Tutorial Signal Analysis"
            )
            figures.append(("Signal Analysis", fig))
        
        # Processing times
        if 'processing_times' in tutorial_data:
            fig = self.perf_viz.visualize_processing_times(
                tutorial_data['processing_times'],
                "Tutorial Processing Performance"
            )
            figures.append(("Processing Performance", fig))
        
        return figures
    
    def save_dashboard(self, figures: List[Tuple[str, plt.Figure]], 
                      output_dir: str = "tutorial_plots"):
        """Save all dashboard figures to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures:
            filename = f"{name.lower().replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 Saved {name} to {filepath}")
        
        print(f"📁 All plots saved to {output_dir}/")


def demo_visualization_tools():
    """Demonstrate the visualization tools with synthetic data"""
    
    print("🎨 PHMGA VISUALIZATION TOOLS DEMO")
    print("=" * 45)
    
    # Create integrated visualizer
    viz = IntegratedVisualizer()
    
    # Generate demo data
    print("\n📊 Generating demo data...")
    
    # Synthetic signals
    fs = 10000
    t = np.linspace(0, 1, fs)
    signals = {
        'normal': np.sin(2*np.pi*60*t) + 0.1*np.random.randn(len(t)),
        'inner_fault': np.sin(2*np.pi*60*t) + 0.5*np.sin(2*np.pi*157*t) + 0.1*np.random.randn(len(t)),
        'outer_fault': np.sin(2*np.pi*60*t) + 0.4*np.sin(2*np.pi*236*t) + 0.1*np.random.randn(len(t))
    }
    
    # Synthetic DAG evolution
    dag_evolution = [
        {'iteration': 1, 'nodes': 3, 'depth': 1, 'nodes_added': 3},
        {'iteration': 2, 'nodes': 5, 'depth': 2, 'nodes_added': 2},
        {'iteration': 3, 'nodes': 8, 'depth': 3, 'nodes_added': 3},
        {'iteration': 4, 'nodes': 10, 'depth': 3, 'nodes_added': 2}
    ]
    
    # Processing times
    processing_times = {
        'Plan Agent': 0.245,
        'Execute Agent': 1.832,
        'Reflect Agent': 0.156,
        'Inquirer Agent': 2.145,
        'Dataset Preparer': 0.678,
        'ML Agent': 3.421,
        'Report Agent': 0.334
    }
    
    # Create demo dashboard
    tutorial_data = {
        'signals': signals,
        'sampling_rate': fs,
        'dag_evolution': dag_evolution,
        'processing_times': processing_times
    }
    
    print("🎨 Creating visualization dashboard...")
    figures = viz.create_tutorial_dashboard(tutorial_data)
    
    print(f"\n✅ Generated {len(figures)} visualization panels:")
    for name, fig in figures:
        print(f"   📊 {name}")
    
    # Optionally save figures
    save_plots = input("\n💾 Save plots to files? (y/n): ").lower().strip()
    if save_plots == 'y':
        viz.save_dashboard(figures)
    
    # Show plots
    plt.show()
    
    print("\n🎓 Visualization Tools Ready!")
    print("   Use IntegratedVisualizer class for tutorial dashboards")
    print("   Individual visualizers available for specific needs")
    print("   All tools integrate with production PHMGA data structures")
    
    return viz, figures


if __name__ == "__main__":
    demo_visualization_tools()