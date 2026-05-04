import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid


def get_user_inputs():
    print("\n" + "=" * 60)
    print("TRAFFIC SIGNAL OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    num_intersections = int(input("\nEnter number of intersections (e.g., 3, 6): "))
    
    steps = int(input("Enter simulation steps (e.g., 100, 200): "))
    
    print("\nTraffic Level:")
    print("  1) Low Traffic (vehicles arrive every 8 steps)")
    print("  2) Medium Traffic (vehicles arrive every 5 steps)")
    print("  3) High Traffic (vehicles arrive every 3 steps)")
    
    level = input("Choose (1/2/3): ")
    
    # Map level to string
    level_map = {"1": "1", "2": "2", "3": "3"}
    traffic_level = level_map.get(level, "2")
    
    # Optional: Choose hybrid configuration
    print("\n" + "-" * 40)
    print("HYBRID ALGORITHM CONFIGURATION")
    print("-" * 40)
    print("Choose variation operators for the Hybrid GA:")
    print("\nSelection Methods:")
    print("  1) Tournament Selection (default)")
    print("  2) Roulette Wheel Selection")
    
    sel_choice = input("Choose (1/2) [default=1]: ") or "1"
    selection_method = "tournament" if sel_choice == "1" else "roulette"
    
    print("\nRecombination Methods:")
    print("  1) Uniform Crossover (default)")
    print("  2) One-Point Crossover")
    
    rec_choice = input("Choose (1/2) [default=1]: ") or "1"
    recombination_method = "uniform" if rec_choice == "1" else "one_point"
    
    print("\nMutation Methods:")
    print("  1) Uniform Mutation (default)")
    print("  2) Gaussian Mutation")
    
    mut_choice = input("Choose (1/2) [default=1]: ") or "1"
    mutation_method = "uniform" if mut_choice == "1" else "gaussian"
    
    print("\nSurvivor Selection Methods:")
    print("  1) Elitism Replacement (default)")
    print("  2) Round-Robin Tournament")
    
    rep_choice = input("Choose (1/2) [default=1]: ") or "1"
    replacement_method = "elitism" if rep_choice == "1" else "round_robin"
    
    return {
        "num_intersections": num_intersections,
        "steps": steps,
        "traffic_level": traffic_level,
        "selection_method": selection_method,
        "recombination_method": recombination_method,
        "mutation_method": mutation_method,
        "replacement_method": replacement_method
    }


def print_solution_details(algorithm_name, solution, fitness, exec_time):
    """Print detailed solution information"""
    print(f"\n--- {algorithm_name} Results ---")
    print(f"Best Fitness (lower is better): {fitness:.4f}")
    print(f"Execution Time: {exec_time:.4f} seconds")
    
    # Show first few signal timings
    print(f"Signal Timings (green times in seconds):")
    for i, timing in enumerate(solution[:min(6, len(solution))]):
        print(f"  Intersection {i+1}: {timing:.0f}s")
    
    if len(solution) > 6:
        print(f"  ... and {len(solution)-6} more intersections")


def run_comparison_experiment(params):
    """Run comparison between PSO and Hybrid with user-selected operators"""
    
    print("\n" + "=" * 70)
    print("TRAFFIC SIGNAL OPTIMIZATION: PSO vs HYBRID (PSO + GA)")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  - Number of intersections: {params['num_intersections']}")
    print(f"  - Simulation steps: {params['steps']}")
    print(f"  - Traffic level: {params['traffic_level']} ({'Low' if params['traffic_level']=='1' else 'Medium' if params['traffic_level']=='2' else 'High'})")
    print(f"  - GA Selection: {params['selection_method']}")
    print(f"  - GA Recombination: {params['recombination_method']}")
    print(f"  - GA Mutation: {params['mutation_method']}")
    print(f"  - GA Survivor Selection: {params['replacement_method']}")
    
    # =========================
    # 1) Run PSO
    # =========================
    print("\n" + "-" * 70)
    print("[1] Running Particle Swarm Optimization (PSO)...")
    print("-" * 70)
    
    start_time = time.time()
    pso_solution, pso_fitness = run_pso(
        num_intersections=params['num_intersections'],
        steps=params['steps'],
        traffic_level=params['traffic_level']
    )
    pso_time = time.time() - start_time
    
    print_solution_details("PSO", pso_solution, pso_fitness, pso_time)
    
    # =========================
    # 2) Run Hybrid
    # =========================
    print("\n" + "-" * 70)
    print("[2] Running Hybrid Algorithm (PSO Warm-start + GA)...")
    print("-" * 70)
    
    start_time = time.time()
    hybrid_solution, hybrid_fitness = run_hybrid(
        num_intersections=params['num_intersections'],
        steps=params['steps'],
        traffic_level=params['traffic_level'],
        selection_method=params['selection_method'],
        recombination_method=params['recombination_method'],
        mutation_method=params['mutation_method'],
        replacement_method=params['replacement_method']
    )
    hybrid_time = time.time() - start_time
    
    print_solution_details("Hybrid", hybrid_solution, hybrid_fitness, hybrid_time)
    
    # =========================
    # 3) Comparison Summary
    # =========================
    print("\n" + "=" * 70)
    print(" COMPARISON SUMMARY")
    print("=" * 70)
    
    # Determine winner
    if hybrid_fitness < pso_fitness:
        winner = "HYBRID"
        improvement = ((pso_fitness - hybrid_fitness) / pso_fitness) * 100
        print(f"\n✓ {winner} algorithm performed better!")
        print(f"  Improvement: {improvement:.2f}% lower fitness")
    elif hybrid_fitness > pso_fitness:
        winner = "PSO"
        improvement = ((hybrid_fitness - pso_fitness) / hybrid_fitness) * 100
        print(f"\n✓ {winner} algorithm performed better!")
        print(f"  Improvement: {improvement:.2f}% lower fitness")
    else:
        winner = "TIE"
        print(f"\n• Both algorithms achieved the same fitness")
    
    # Detailed comparison table
    print("\n" + "-" * 70)
    print(f"{'Metric':<25}{'PSO':<20}{'Hybrid (PSO+GA)':<20}")
    print("-" * 70)
    print(f"{'Best Fitness':<25}{pso_fitness:<20.4f}{hybrid_fitness:<20.4f}")
    print(f"{'Execution Time (s)':<25}{pso_time:<20.4f}{hybrid_time:<20.4f}")
    
    # Speed comparison
    time_diff = abs(pso_time - hybrid_time)
    if hybrid_time < pso_time:
        print(f"{'Speed':<25}{'Slower':<20}{f'Faster by {time_diff:.2f}s':<20}")
    else:
        print(f"{'Speed':<25}{f'Faster by {time_diff:.2f}s':<20}{'Slower':<20}")
    
    # Sample solution comparison
    print("\n" + "-" * 70)
    print("SAMPLE SIGNAL TIMINGS (first 5 intersections)")
    print("-" * 70)
    print(f"{'Intersection':<15}{'PSO (s)':<15}{'Hybrid (s)':<15}")
    for i in range(min(5, params['num_intersections'])):
        pso_val = pso_solution[i] if i < len(pso_solution) else "N/A"
        hybrid_val = hybrid_solution[i] if i < len(hybrid_solution) else "N/A"
        print(f"{i+1:<15}{pso_val:<15}{hybrid_val:<15}")
    
    return {
        "pso": {"fitness": pso_fitness, "time": pso_time, "solution": pso_solution},
        "hybrid": {"fitness": hybrid_fitness, "time": hybrid_time, "solution": hybrid_solution},
        "winner": winner
    }


def run_multiple_comparisons(params, num_runs=5):
    """Run multiple comparisons to get statistical significance"""
    
    print("\n" + "=" * 70)
    print(f"STATISTICAL COMPARISON ({num_runs} runs)")
    print("=" * 70)
    
    pso_fitnesses = []
    hybrid_fitnesses = []
    pso_times = []
    hybrid_times = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}...")
        
        # Run PSO
        start = time.time()
        _, pso_fit = run_pso(
            num_intersections=params['num_intersections'],
            steps=params['steps'],
            traffic_level=params['traffic_level']
        )
        pso_times.append(time.time() - start)
        pso_fitnesses.append(pso_fit)
        
        # Run Hybrid
        start = time.time()
        _, hybrid_fit = run_hybrid(
            num_intersections=params['num_intersections'],
            steps=params['steps'],
            traffic_level=params['traffic_level'],
            selection_method=params['selection_method'],
            recombination_method=params['recombination_method'],
            mutation_method=params['mutation_method'],
            replacement_method=params['replacement_method']
        )
        hybrid_times.append(time.time() - start)
        hybrid_fitnesses.append(hybrid_fit)
    
    # Statistical summary
    print("\n" + "-" * 70)
    print("STATISTICAL RESULTS")
    print("-" * 70)
    
    import numpy as np
    
    print(f"\n{'Metric':<25}{'PSO':<20}{'Hybrid':<20}")
    print("-" * 70)
    print(f"{'Mean Fitness':<25}{np.mean(pso_fitnesses):<20.4f}{np.mean(hybrid_fitnesses):<20.4f}")
    print(f"{'Std Fitness':<25}{np.std(pso_fitnesses):<20.4f}{np.std(hybrid_fitnesses):<20.4f}")
    print(f"{'Best Fitness':<25}{min(pso_fitnesses):<20.4f}{min(hybrid_fitnesses):<20.4f}")
    print(f"{'Worst Fitness':<25}{max(pso_fitnesses):<20.4f}{max(hybrid_fitnesses):<20.4f}")
    print(f"{'Mean Time (s)':<25}{np.mean(pso_times):<20.4f}{np.mean(hybrid_times):<20.4f}")
    
    # Wins count
    hybrid_wins = sum(1 for p, h in zip(pso_fitnesses, hybrid_fitnesses) if h < p)
    pso_wins = sum(1 for p, h in zip(pso_fitnesses, hybrid_fitnesses) if p < h)
    ties = num_runs - hybrid_wins - pso_wins
    
    print(f"\n{'Wins':<25}{pso_wins:<20}{hybrid_wins:<20}")
    print(f"{'Ties':<25}{ties:<20}{'-':<20}")
    
    return {
        "pso": {"fitnesses": pso_fitnesses, "times": pso_times},
        "hybrid": {"fitnesses": hybrid_fitnesses, "times": hybrid_times},
        "hybrid_wins": hybrid_wins,
        "pso_wins": pso_wins,
        "ties": ties
    }


def main():
    """Main execution function"""
    
    # Get user inputs
    params = get_user_inputs()
    
    results = run_comparison_experiment(params)
    
    # Final message
    print("\n" + "=" * 70)
    print(" IMPLEMENTATION COMPLETED")
    print("=" * 70)
    print("\nVariation operators used:")
    print(f"  • Selection: {params['selection_method']}")
    print(f"  • Recombination: {params['recombination_method']}")
    print(f"  • Mutation: {params['mutation_method']}")
    print(f"  • Survivor Selection: {params['replacement_method']}")


if __name__ == "__main__":
    main()