

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.patches import Circle
import os


class Visualizer:
    
    
    def __init__(self, save_dir: Optional[str] = './results/figures'):
        
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 100
    
    def plot_eigenvalues_with_unit_circle(self,
                                          eigenvalues: np.ndarray,
                                          title: str = "Koopman Eigenvalues",
                                          save_name: Optional[str] = None,
                                          show: bool = True) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', 
                       linewidth=2, label='Unit Circle')
        ax.add_patch(circle)
        
        ax.scatter(eigenvalues.real, eigenvalues.imag, 
                  c='blue', marker='o', s=50, alpha=0.7, label='Eigenvalues')
        
        unstable_mask = np.abs(eigenvalues) > 1.0
        if np.any(unstable_mask):
            unstable_eigs = eigenvalues[unstable_mask]
            ax.scatter(unstable_eigs.real, unstable_eigs.imag,
                      c='red', marker='x', s=100, linewidths=2,
                      label=f'Unstable ({np.sum(unstable_mask)})')
        
        max_abs = np.max(np.abs(eigenvalues))
        lim = max(1.2, max_abs * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        if show:
            plt.show()
        
        return fig

    def plot_eigenvalues_combined(self,
                                  eigenvalues_dict: Dict[str, np.ndarray],
                                  colors: Optional[Dict[str, str]] = None,
                                  title: str = "Combined Koopman Eigenvalues",
                                  save_name: Optional[str] = None,
                                  show: bool = True,
                                  max_radius: float = 3.0) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=(6.8, 6.5))

        ax.set_xlabel('Real Part', fontsize=22)
        ax.set_ylabel('Imaginary Part', fontsize=22)
        ax.tick_params(axis='both', labelsize=18)

        circle = Circle((0, 0), 1, fill=False, color='black', linestyle='--',
                        linewidth=1.5, label='Unit Circle')
        ax.add_patch(circle)

        if colors is None:
            colors = {}
        default_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

        summary_lines = []
        for idx, (method, eigs) in enumerate(eigenvalues_dict.items()):
            color = colors.get(method, default_colors[idx % len(default_colors)])
            mags = np.abs(eigs)
            total = len(eigs)
            unstable_mask = mags > 1.0
            unstable = int(np.sum(unstable_mask))

            in_range = mags <= max_radius
            eigs_plot = eigs[in_range]
            mags_plot = mags[in_range]

            stable_mask = mags_plot <= 1.0
            unstable_plot_mask = mags_plot > 1.0

            if np.any(stable_mask):
                ax.scatter(
                    eigs_plot[stable_mask].real,
                    eigs_plot[stable_mask].imag,
                    c=color,
                    marker='o',
                    s=60,
                    alpha=0.75,
                    label=method
                )
            if np.any(unstable_plot_mask):
                ax.scatter(
                    eigs_plot[unstable_plot_mask].real,
                    eigs_plot[unstable_plot_mask].imag,
                    c=color,
                    marker='x',
                    s=90,
                    linewidths=2.0,
                    alpha=0.85
                )

            summary_lines.append(f"{method}: total={total}, unstable={unstable}")

        max_abs = 1.2
        if eigenvalues_dict:
            all_eigs = np.concatenate(list(eigenvalues_dict.values()))
            max_abs = max(max_abs, np.max(np.abs(all_eigs[~np.isnan(all_eigs)])))
        lim = 2.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        summary_text = "\n".join(summary_lines)
        fig.text(
            0.5, 1.02, summary_text,
            ha='center',
            va='top',
            fontsize=16
        )

        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
            fontsize=18,
            ncol=max(1, len(eigenvalues_dict)),
            frameon=False
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.9])

        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")

        if show:
            plt.show()

        return fig
    
    def plot_trajectory_comparison(self,
                                   t: np.ndarray,
                                   true_states: np.ndarray,
                                   reconstructed_states: np.ndarray,
                                   dim_names: Optional[List[str]] = None,
                                   title: str = "Trajectory Reconstruction",
                                   save_name: Optional[str] = None,
                                   show: bool = True,
                                   valid_range: Optional[Tuple[int, int]] = None) -> plt.Figure:
        
        n_dims = true_states.shape[1]
        
        if dim_names is None:
            dim_names = [f"$x_{i+1}$" for i in range(n_dims)]
        
        fig, axes = plt.subplots(n_dims, 1, figsize=(7.0, 2.4 * n_dims), sharex=True)
        
        if n_dims == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.plot(t, true_states[:, i], 'b-', linewidth=1.5, 
                   label='True', alpha=0.8)
            
            if valid_range is not None:
                start_idx, end_idx = valid_range
                ax.plot(t[start_idx:end_idx], reconstructed_states[start_idx:end_idx, i], 
                       'r--', linewidth=1.5,
                       label='Reconstructed', alpha=0.8)
            else:
                ax.plot(t, reconstructed_states[:, i], 'r--', linewidth=1.5,
                       label='Reconstructed', alpha=0.8)
            
            ax.set_ylabel(dim_names[i], fontsize=18)
            ax.legend(loc='upper right', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=14)
        
        axes[-1].set_xlabel('Time', fontsize=18)
        fig.suptitle(title, fontsize=18)
        plt.tight_layout()
        
        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_prediction_error(self,
                             t: np.ndarray,
                             errors: np.ndarray,
                             error_type: str = "NRMSE",
                             title: str = "Prediction Error",
                             save_name: Optional[str] = None,
                             show: bool = True) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(t, errors, 'b-', linewidth=2)
        ax.set_xlabel('Time / Step', fontsize=12)
        ax.set_ylabel(error_type, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        
        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_reconstruction_error_comparison(self,
                                            t: np.ndarray,
                                            true_states: np.ndarray,
                                            reconstructed_states_dict: Dict[str, np.ndarray],
                                            title: str = "Reconstruction Error Comparison",
                                            save_name: Optional[str] = None,
                                            show: bool = True,
                                            valid_ranges_dict: Optional[Dict[str, Tuple[int, int]]] = None) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        linestyles = ['-', '--', '-.', ':', '-', '--']
        
        for i, (method_name, reconstructed) in enumerate(reconstructed_states_dict.items()):
            errors = np.linalg.norm(true_states - reconstructed, axis=1)
            
            valid_range = None
            if valid_ranges_dict is not None and method_name in valid_ranges_dict:
                valid_range = valid_ranges_dict[method_name]
            
            if valid_range is not None:
                start_idx, end_idx = valid_range
                ax.plot(t[start_idx:end_idx], errors[start_idx:end_idx], 
                       color=colors[i % len(colors)],
                       linestyle=linestyles[i % len(linestyles)],
                       linewidth=2,
                       label=f"{method_name} (predicted)",
                       alpha=0.8)
                if start_idx > 0:
                    ax.axvspan(t[0], t[start_idx], alpha=0.1, color='gray', 
                              label=f'{method_name} original' if i == 0 else '')
            else:
                ax.plot(t, errors, 
                       color=colors[i % len(colors)],
                       linestyle=linestyles[i % len(linestyles)],
                       linewidth=2,
                       label=method_name,
                       alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Reconstruction Error (L2 Norm)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        
        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_method_comparison_bar(self,
                                   method_names: List[str],
                                   nrmse_values: List[float],
                                   title: str = "Method Comparison (NRMSE)",
                                   save_name: Optional[str] = None,
                                   show: bool = True) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(method_names))
        bars = ax.bar(x, nrmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        for bar, value in zip(bars, nrmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2e}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.set_ylabel('NRMSE', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_method_comparison_table(self,
                                     comparison_dict: Dict,
                                     title: str = "Method Comparison Table",
                                     save_name: Optional[str] = None,
                                     show: bool = True) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=(12, len(comparison_dict['method_names']) * 0.8 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        method_names = comparison_dict['method_names']
        n_methods = len(method_names)
        
        table_data = []
        for i in range(n_methods):
            row = [
                method_names[i],
                f"{comparison_dict['nrmse'][i]:.2e}",
                f"{comparison_dict['spectral_radius'][i]:.4f}",
                f"{comparison_dict['a_condition_number'][i]:.2e}",
                f"{comparison_dict['gram_condition_number'][i]:.2e}" 
                    if not np.isnan(comparison_dict['gram_condition_number'][i]) else "N/A"
            ]
            table_data.append(row)
        
        col_labels = ['Method', 'NRMSE', 'Spectral\nRadius', 
                     'A Cond.\nNumber', 'G Cond.\nNumber']
        
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(col_labels)):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title(title, fontsize=14, pad=20)
        plt.tight_layout()
        
        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_phase_portrait_2d(self,
                              states: np.ndarray,
                              reconstructed_states: Optional[np.ndarray] = None,
                              title: str = "Phase Portrait",
                              xlabel: str = "$x_1$",
                              ylabel: str = "$x_2$",
                              save_name: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5,
               label='True', alpha=0.7)
        
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
        ax.plot(states[-1, 0], states[-1, 1], 'rs', markersize=10, label='End')
        
        if reconstructed_states is not None:
            ax.plot(reconstructed_states[:, 0], reconstructed_states[:, 1], 
                   'r--', linewidth=1.5, label='Reconstructed', alpha=0.7)
        
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        
        if save_name is not None and self.save_dir is not None:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {filepath}")
        
        if show:
            plt.show()
        
        return fig


__all__ = ['Visualizer']
