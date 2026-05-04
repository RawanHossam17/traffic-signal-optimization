import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.simple_runner import run_comparison


def plot_results(pso_fits, hybrid_fits):
    """Plot comparison between PSO and Hybrid"""
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Line graph
    x = range(1, len(pso_fits) + 1)
    ax1.plot(x, pso_fits, 'b-o', alpha=0.7, markersize=4, label='PSO', linewidth=1)
    ax1.plot(x, hybrid_fits, 'r-s', alpha=0.7, markersize=4, label='Hybrid', linewidth=1)
    ax1.axhline(np.mean(pso_fits), color='b', linestyle='--', alpha=0.5, label=f'PSO Mean: {np.mean(pso_fits):.2f}')
    ax1.axhline(np.mean(hybrid_fits), color='r', linestyle='--', alpha=0.5, label=f'Hybrid Mean: {np.mean(hybrid_fits):.2f}')
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Fitness (lower is better)')
    ax1.set_title('PSO vs Hybrid Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    data = [pso_fits, hybrid_fits]
    bp = ax2.boxplot(data, labels=['PSO', 'Hybrid'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Fitness (lower is better)')
    ax2.set_title('Distribution Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiments/comparison_plots.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # Run comparison
    print("Running 30 comparisons...")
    pso_fits, hybrid_fits = run_comparison(30)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(pso_fits, hybrid_fits)
    
    print("\nPlots saved to 'experiments/' folder")