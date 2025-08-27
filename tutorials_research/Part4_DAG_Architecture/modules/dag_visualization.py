"""
DAG Visualization Tools

Advanced visualization capabilities for DAGs using NetworkX and matplotlib.
Includes static plots, animations, and interactive features.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import time

from dag_fundamentals import ResearchDAG, NodeType, ExecutionStatus


class LayoutType(Enum):
    """Different layout algorithms for DAG visualization"""
    SPRING = "spring"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    SHELL = "shell"
    KAMADA_KAWAI = "kamada_kawai"


class DAGVisualizer:
    """
    Advanced DAG visualization with multiple layout options and features.
    """
    
    def __init__(self):
        self.color_schemes = {
            'default': {
                NodeType.INPUT: '#e8f5e8',
                NodeType.PROCESSING: '#fff2cc', 
                NodeType.DECISION: '#ffe6cc',
                NodeType.AGGREGATION: '#f0e6ff',
                NodeType.OUTPUT: '#ffe6e6',
                NodeType.VALIDATION: '#e6f3ff'
            },
            'vibrant': {
                NodeType.INPUT: '#4CAF50',
                NodeType.PROCESSING: '#2196F3', 
                NodeType.DECISION: '#FF9800',
                NodeType.AGGREGATION: '#9C27B0',
                NodeType.OUTPUT: '#F44336',
                NodeType.VALIDATION: '#00BCD4'
            },
            'status': {
                ExecutionStatus.PENDING: '#CCCCCC',
                ExecutionStatus.RUNNING: '#FFEB3B',
                ExecutionStatus.COMPLETED: '#4CAF50',
                ExecutionStatus.FAILED: '#F44336',
                ExecutionStatus.SKIPPED: '#9E9E9E'
            }
        }
    
    def plot_dag(self, dag: ResearchDAG, 
                 layout: LayoutType = LayoutType.SPRING,
                 color_scheme: str = 'default',
                 figsize: Tuple[int, int] = (12, 8),
                 show_labels: bool = True,
                 show_execution_times: bool = False,
                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive DAG visualization.
        
        Args:
            dag: The ResearchDAG to visualize
            layout: Layout algorithm to use
            color_scheme: Color scheme ('default', 'vibrant', 'status')
            figsize: Figure size as (width, height)
            show_labels: Whether to show node labels
            show_execution_times: Whether to show execution times
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout algorithm
        pos = self._get_layout(dag.graph, layout)
        
        # Get node colors based on scheme
        if color_scheme == 'status':
            colors = [self.color_schemes['status'].get(dag.nodes[node_id].status, '#CCCCCC')
                     for node_id in dag.graph.nodes()]
        else:
            colors = [self.color_schemes[color_scheme].get(dag.nodes[node_id].node_type, '#f0f0f0')
                     for node_id in dag.graph.nodes()]
        
        # Calculate node sizes based on execution time if available
        if show_execution_times:
            sizes = []
            for node_id in dag.graph.nodes():
                exec_time = dag.nodes[node_id].execution_time
                base_size = 2000
                time_factor = min(exec_time * 500, 2000) if exec_time > 0 else 0
                sizes.append(base_size + time_factor)
        else:
            sizes = [2500] * len(dag.graph.nodes())
        
        # Draw nodes
        nx.draw_networkx_nodes(dag.graph, pos,
                              node_color=colors,
                              node_size=sizes,
                              alpha=0.8,
                              linewidths=2,
                              edgecolors='black',
                              ax=ax)
        
        # Draw edges with different styles based on relationships
        nx.draw_networkx_edges(dag.graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              alpha=0.6,
                              width=2,
                              ax=ax)
        
        # Add labels
        if show_labels:
            labels = {}
            for node_id in dag.graph.nodes():
                node = dag.nodes[node_id]
                label = node.name
                if show_execution_times and node.execution_time > 0:
                    label += f"\\n({node.execution_time:.2f}s)"
                labels[node_id] = label
                
            nx.draw_networkx_labels(dag.graph, pos, labels,
                                   font_size=9,
                                   font_weight='bold',
                                   ax=ax)
        
        # Add title and formatting
        title = f"DAG Visualization: {dag.dag_id}"
        if color_scheme == 'status':
            title += " (Execution Status)"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        self._add_legend(ax, dag, color_scheme)
        
        # Add statistics text
        stats = dag.get_statistics()
        stats_text = f"Nodes: {stats['total_nodes']} | Edges: {stats['total_edges']} | Density: {stats['graph_metrics']['density']:.2f}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_execution_timeline(self, dag: ResearchDAG, 
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot execution timeline showing when each node executed"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get execution order and times
        execution_order = dag.execution_order or list(dag.graph.nodes())
        node_times = []
        node_names = []
        
        cumulative_time = 0
        for node_id in execution_order:
            node = dag.nodes[node_id]
            node_times.append(node.execution_time)
            node_names.append(node.name)
        
        # Create timeline chart
        y_positions = range(len(node_names))
        colors = [self.color_schemes['default'].get(dag.nodes[execution_order[i]].node_type, '#f0f0f0')
                 for i in range(len(execution_order))]
        
        bars = ax.barh(y_positions, node_times, color=colors, alpha=0.8, edgecolor='black')
        
        # Customize chart
        ax.set_yticks(y_positions)
        ax.set_yticklabels(node_names)
        ax.set_xlabel('Execution Time (seconds)')
        ax.set_title(f'DAG Execution Timeline: {dag.dag_id}', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add time labels on bars
        for bar, time_val in zip(bars, node_times):
            if time_val > 0:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{time_val:.2f}s', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_parallel_analysis(self, dag: ResearchDAG,
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Analyze and visualize parallelization opportunities"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: DAG with parallel groups highlighted
        pos = self._get_layout(dag.graph, LayoutType.HIERARCHICAL)
        
        # Identify parallel groups (nodes with same distance from source)
        try:
            # Find source nodes (no predecessors)
            sources = [n for n in dag.graph.nodes() if dag.graph.in_degree(n) == 0]
            if not sources:
                sources = list(dag.graph.nodes())[:1]
            
            # Calculate shortest path lengths from first source
            source = sources[0]
            distances = nx.single_source_shortest_path_length(dag.graph, source)
            
            # Group nodes by distance (parallel groups)
            parallel_groups = {}
            for node, dist in distances.items():
                if dist not in parallel_groups:
                    parallel_groups[dist] = []
                parallel_groups[dist].append(node)
            
            # Create colors for parallel groups
            colors = plt.cm.Set3(np.linspace(0, 1, len(parallel_groups)))
            node_colors = []
            for node_id in dag.graph.nodes():
                dist = distances.get(node_id, 0)
                group_idx = list(parallel_groups.keys()).index(dist)
                node_colors.append(colors[group_idx])
            
            # Draw the graph
            nx.draw_networkx_nodes(dag.graph, pos, node_color=node_colors,
                                  node_size=2000, alpha=0.8, ax=ax1)
            nx.draw_networkx_edges(dag.graph, pos, alpha=0.6, arrows=True,
                                  arrowsize=15, ax=ax1)
            
            labels = {node_id: dag.nodes[node_id].name for node_id in dag.graph.nodes()}
            nx.draw_networkx_labels(dag.graph, pos, labels, font_size=8, ax=ax1)
            
            ax1.set_title('Parallel Groups Analysis', fontweight='bold')
            ax1.axis('off')
            
            # Right plot: Execution time comparison
            group_labels = []
            sequential_times = []
            parallel_times = []
            
            for dist, nodes in parallel_groups.items():
                group_labels.append(f'Group {dist}')
                # Sequential time: sum of all execution times in group
                seq_time = sum(dag.nodes[node_id].execution_time for node_id in nodes)
                # Parallel time: max execution time in group
                par_time = max([dag.nodes[node_id].execution_time for node_id in nodes] + [0])
                
                sequential_times.append(seq_time)
                parallel_times.append(par_time)
            
            x = np.arange(len(group_labels))
            width = 0.35
            
            ax2.bar(x - width/2, sequential_times, width, label='Sequential', alpha=0.8)
            ax2.bar(x + width/2, parallel_times, width, label='Parallel', alpha=0.8)
            
            ax2.set_xlabel('Parallel Groups')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Sequential vs Parallel Execution')
            ax2.set_xticks(x)
            ax2.set_xticklabels(group_labels)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            
            # Calculate speedup
            total_sequential = sum(sequential_times)
            total_parallel = max(parallel_times) * len(parallel_groups)  # Simplified
            if total_parallel > 0:
                speedup = total_sequential / total_parallel
                ax2.text(0.02, 0.98, f'Theoretical Speedup: {speedup:.2f}x',
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Analysis failed: {str(e)}', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax2.text(0.5, 0.5, 'No parallel analysis available',
                    transform=ax2.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, dags: List[ResearchDAG], 
                              labels: List[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Compare multiple DAGs side by side"""
        n_dags = len(dags)
        fig, axes = plt.subplots(1, n_dags, figsize=figsize)
        
        if n_dags == 1:
            axes = [axes]
        if labels is None:
            labels = [f'DAG {i+1}' for i in range(n_dags)]
        
        for i, (dag, label) in enumerate(zip(dags, labels)):
            ax = axes[i]
            
            # Use hierarchical layout for comparison
            pos = self._get_layout(dag.graph, LayoutType.HIERARCHICAL)
            
            # Color by node type
            colors = [self.color_schemes['default'].get(dag.nodes[node_id].node_type, '#f0f0f0')
                     for node_id in dag.graph.nodes()]
            
            # Draw graph
            nx.draw_networkx_nodes(dag.graph, pos, node_color=colors,
                                  node_size=1500, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(dag.graph, pos, alpha=0.6, arrows=True,
                                  arrowsize=12, ax=ax)
            
            # Add labels
            node_labels = {node_id: dag.nodes[node_id].name[:8] + ('...' if len(dag.nodes[node_id].name) > 8 else '')
                          for node_id in dag.graph.nodes()}
            nx.draw_networkx_labels(dag.graph, pos, node_labels, font_size=8, ax=ax)
            
            # Add title and stats
            stats = dag.get_statistics()
            ax.set_title(f'{label}\\nNodes: {stats["total_nodes"]} | Edges: {stats["total_edges"]}',
                        fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('DAG Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _get_layout(self, graph: nx.DiGraph, layout_type: LayoutType) -> Dict:
        """Get node positions based on layout algorithm"""
        if layout_type == LayoutType.SPRING:
            return nx.spring_layout(graph, k=2, iterations=50)
        elif layout_type == LayoutType.HIERARCHICAL:
            return self._hierarchical_layout(graph)
        elif layout_type == LayoutType.CIRCULAR:
            return nx.circular_layout(graph)
        elif layout_type == LayoutType.SHELL:
            return nx.shell_layout(graph)
        elif layout_type == LayoutType.KAMADA_KAWAI:
            return nx.kamada_kawai_layout(graph)
        else:
            return nx.spring_layout(graph)
    
    def _hierarchical_layout(self, graph: nx.DiGraph) -> Dict:
        """Create hierarchical layout showing DAG levels"""
        try:
            # Use topological sort to get levels
            topo_order = list(nx.topological_sort(graph))
            
            # Calculate levels based on longest path from source
            levels = {}
            for node in topo_order:
                if graph.in_degree(node) == 0:
                    levels[node] = 0
                else:
                    levels[node] = max([levels[pred] for pred in graph.predecessors(node)]) + 1
            
            # Group nodes by level
            level_groups = {}
            for node, level in levels.items():
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(node)
            
            # Position nodes
            pos = {}
            for level, nodes in level_groups.items():
                y_pos = -level  # Higher levels at top
                for i, node in enumerate(nodes):
                    x_pos = (i - len(nodes)/2) * 2  # Center nodes horizontally
                    pos[node] = (x_pos, y_pos)
            
            return pos
            
        except nx.NetworkXError:
            # Fallback to spring layout if hierarchical fails
            return nx.spring_layout(graph)
    
    def _add_legend(self, ax: plt.Axes, dag: ResearchDAG, color_scheme: str):
        """Add legend to the plot"""
        if color_scheme == 'status':
            # Status legend
            unique_statuses = set(dag.nodes[node_id].status for node_id in dag.graph.nodes())
            legend_elements = [
                patches.Patch(color=self.color_schemes['status'].get(status, '#CCCCCC'),
                             label=status.value.title())
                for status in unique_statuses
            ]
        else:
            # Node type legend
            unique_types = set(dag.nodes[node_id].node_type for node_id in dag.graph.nodes())
            legend_elements = [
                patches.Patch(color=self.color_schemes[color_scheme].get(node_type, '#f0f0f0'),
                             label=node_type.value.title())
                for node_type in unique_types
            ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))


# Convenience functions
def quick_plot(dag: ResearchDAG, layout: str = 'spring', **kwargs) -> plt.Figure:
    """Quick plot function for easy DAG visualization"""
    visualizer = DAGVisualizer()
    layout_enum = LayoutType(layout) if isinstance(layout, str) else layout
    return visualizer.plot_dag(dag, layout_enum, **kwargs)


def compare_execution_modes(dag: ResearchDAG, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Compare sequential vs parallel execution visualization"""
    visualizer = DAGVisualizer()
    return visualizer.plot_parallel_analysis(dag, figsize)


def execution_timeline(dag: ResearchDAG, **kwargs) -> plt.Figure:
    """Create execution timeline plot"""
    visualizer = DAGVisualizer()
    return visualizer.plot_execution_timeline(dag, **kwargs)