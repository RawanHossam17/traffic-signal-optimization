import numpy as np
from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid


def run_experiments():

    pso_results = []
    hybrid_results = []

    for i in range(30):
        print(f"\nRun {i+1}")

        _, pso_fit = run_pso()
        _, hybrid_fit = run_hybrid()

        pso_results.append(pso_fit)
        hybrid_results.append(hybrid_fit)

    print("\n===== RESULTS =====")

    print("\nPSO:")
    print("Mean:", np.mean(pso_results))
    print("Std:", np.std(pso_results))

    print("\nHybrid:")
    print("Mean:", np.mean(hybrid_results))
    print("Std:", np.std(hybrid_results))


if __name__ == "__main__":
    run_experiments()