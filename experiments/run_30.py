import numpy as np
import random
import time

from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid
from algorithms.ga import run_ga


def run_experiments():
    start_total_time = time.time()
    
    print("\n" + "="*50)
    print("     PHASE 1: PSO vs GA vs HYBRID")
    print("="*50)

    pso_results = []
    ga_results = []
    hybrid_results = []

    for i in range(30):
        run_start_time = time.time()

        random.seed(i)
        np.random.seed(i)

        print(f"[Run {i+1}/30] ", end="")

        # === Normal run ===
        _, pso_fit, _ = run_pso()
        _, ga_fit = run_ga(mutation_type=1, crossover_type=1, seed=i)
        _, hybrid_fit = run_hybrid(
            mutation_type=1,
            crossover_type=1,
            selection_type=1
        )

        # === DEBUG  ===
        if i == 0:
            sol_pso, _, _ = run_pso()
            sol_ga, _ = run_ga(mutation_type=1, crossover_type=1, seed=i)
            sol_hybrid, _ = run_hybrid(
                mutation_type=1,
                crossover_type=1,
                selection_type=1
            )

            print("\n--- Sample Solutions ---")
            print("PSO:", sol_pso)
            print("GA:", sol_ga)
            print("Hybrid:", sol_hybrid)
            print("------------------------\n")

        
        pso_results.append(pso_fit)
        ga_results.append(ga_fit)
        hybrid_results.append(hybrid_fit)

        print(f"PSO={pso_fit:.2f} | GA={ga_fit:.2f} | Hybrid={hybrid_fit:.2f} | {time.time()-run_start_time:.1f}s")

    print("\n" + "-"*40)

    pso_mean = np.mean(pso_results)
    ga_mean = np.mean(ga_results)
    hybrid_mean = np.mean(hybrid_results)

    pso_std = np.std(pso_results)
    ga_std = np.std(ga_results)
    hybrid_std = np.std(hybrid_results)

    print(f"PSO    => Mean: {pso_mean:.2f} | Std: {pso_std:.2f}")
    print(f"GA     => Mean: {ga_mean:.2f} | Std: {ga_std:.2f}")
    print(f"Hybrid => Mean: {hybrid_mean:.2f} | Std: {hybrid_std:.2f}")

    results = [
        ("PSO", pso_mean),
        ("GA", ga_mean),
        ("Hybrid", hybrid_mean)
    ]

    results.sort(key=lambda x: x[1])
    best_name, best_value = results[0]
    second_name, second_value = results[1]

    print("\n" + "-"*40)

    if abs(best_value - second_value) < 0.1:
        print(">>> No clear winner (results are very close)")
    else:
        print(f">>> BEST ALGORITHM: {best_name}")

    print("-"*40)

    # =========================
    # PHASE 2 (Hybrid variations)
    # =========================
    print("\n" + "="*50)
    print("     PHASE 2: HYBRID VARIATIONS")
    print("="*50)

    all_results = []

    for m in [1, 2]:
        for c in [1, 2]:
            for s in [1, 2]:

                results = []

                print(f"\n[Testing M{m}, C{c}, S{s}]")

                for i in range(30):
                    random.seed(i)
                    np.random.seed(i)

                    _, fit = run_hybrid(
                        mutation_type=m,
                        crossover_type=c,
                        selection_type=s
                    )

                    results.append(fit)

                mean_val = np.mean(results)
                std_val = np.std(results)

                all_results.append((f"M{m}_C{c}_S{s}", mean_val, std_val))

                print(f"Result => Mean: {mean_val:.2f} | Std: {std_val:.2f}")

    print("\n" + "="*50)
    print("     FINAL COMPARISON")
    print("="*50)

    all_results.sort(key=lambda x: x[1])

    for name, mean, std in all_results:
        print(f"{name:12} | Mean={mean:.2f} | Std={std:.2f}")

    print("\n>>> BEST CONFIGURATION:", all_results[0][0])

    total_time = (time.time() - start_total_time) / 60
    print("\n" + "="*50)
    print(f"TOTAL TIME: {total_time:.1f} minutes")
    print("="*50)


if __name__ == "__main__":
    run_experiments()