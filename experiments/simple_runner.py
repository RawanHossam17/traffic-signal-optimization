import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid


def run_comparison(num_runs=30):
    """Compare PSO vs Hybrid (30 runs)"""
    
    print("\n" + "="*60)
    print("PSO vs HYBRID (PSO + GA) - 30 RUNS")
    print("="*60)
    
    pso_fits = []
    hybrid_fits = []
    
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")
        
        # Run PSO
        print("  PSO...")
        _, pso_f = run_pso(6, 200, "2")
        pso_fits.append(pso_f)
        print(f"    Fitness: {pso_f:.2f}")
        
        # Run Hybrid
        print("  Hybrid...")
        _, hybrid_f = run_hybrid(6, 200, "2", 
                                 "tournament", "uniform", "gaussian", "elitism")
        hybrid_fits.append(hybrid_f)
        print(f"    Fitness: {hybrid_f:.2f}")
    
    # Results
    print("\n" + "="*60)
    print("FINAL RESULTS (30 runs)")
    print("="*60)
    
    print(f"\nPSO:")
    print(f"  Mean: {np.mean(pso_fits):.2f} ± {np.std(pso_fits):.2f}")
    print(f"  Best: {min(pso_fits):.2f}")
    print(f"  Worst: {max(pso_fits):.2f}")
    
    print(f"\nHybrid (PSO+GA):")
    print(f"  Mean: {np.mean(hybrid_fits):.2f} ± {np.std(hybrid_fits):.2f}")
    print(f"  Best: {min(hybrid_fits):.2f}")
    print(f"  Worst: {max(hybrid_fits):.2f}")
    
    # Improvement
    improvement = ((np.mean(pso_fits) - np.mean(hybrid_fits)) / np.mean(pso_fits)) * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    if np.mean(hybrid_fits) < np.mean(pso_fits):
        print("Winner: HYBRID")
    else:
        print("Winner: PSO")
    
    return pso_fits, hybrid_fits


if __name__ == "__main__":
    pso, hybrid = run_comparison(30)