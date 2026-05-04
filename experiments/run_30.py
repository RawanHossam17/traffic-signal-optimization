import numpy as np
import random
import time
from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid

def run_experiments():
    start_total_time = time.time()
    
    print("\n" + "="*50)
    print("         PHASE 1: PSO vs HYBRID")
    print("="*50)

    pso_results = []
    hybrid_results = []

    for i in range(30):
        run_start_time = time.time()
        random.seed(i)
        np.random.seed(i)

        print(f"[Run {i+1}/30] ", end="")

        _, pso_fit = run_pso()
        _, hybrid_fit = run_hybrid(mutation_type=1, crossover_type=1)

        pso_results.append(pso_fit)
        hybrid_results.append(hybrid_fit)

        print(f"PSO={pso_fit:.2f} | Hybrid={hybrid_fit:.2f} | {time.time()-run_start_time:.1f}s")

    print("\n" + "-"*40)
    pso_mean = np.mean(pso_results)
    hybrid_mean = np.mean(hybrid_results)

    print(f"PSO    => Mean: {pso_mean:.2f} | Std: {np.std(pso_results):.2f}")
    print(f"Hybrid => Mean: {hybrid_mean:.2f} | Std: {np.std(hybrid_results):.2f}")

  
    if hybrid_mean < pso_mean:
        print(">>> Hybrid is BETTER than PSO")
    else:
        print(">>> PSO is BETTER than Hybrid")

    print("-"*40)

    # =========================
    # PHASE 2
    # =========================
    print("\n" + "="*50)
    print("         PHASE 2: HYBRID VARIATIONS")
    print("="*50)

    all_results = []

    for m in [1, 2]:
        for c in [1, 2]:

            results = []

            print(f"\n[Testing M{m}, C{c}]")

            for i in range(30):
                random.seed(i)
                np.random.seed(i)

                _, fit = run_hybrid(mutation_type=m, crossover_type=c)
                results.append(fit)

            mean_val = np.mean(results)
            std_val = np.std(results)

            all_results.append((f"M{m}_C{c}", mean_val, std_val))

            print(f"Result => Mean: {mean_val:.2f} | Std: {std_val:.2f}")

   
    print("\n" + "="*50)
    print("         FINAL COMPARISON")
    print("="*50)


    all_results.sort(key=lambda x: x[1])

    for name, mean, std in all_results:
        print(f"{name:8} | Mean={mean:.2f} | Std={std:.2f}")

    print("\n>>> BEST CONFIGURATION:", all_results[0][0])

    total_time = (time.time() - start_total_time) / 60
    print("\n" + "="*50)
    print(f"TOTAL TIME: {total_time:.1f} minutes")
    print("="*50)


if __name__ == "__main__":
    run_experiments()