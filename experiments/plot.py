import matplotlib.pyplot as plt
import numpy as np
from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid


def collect_results(runs=30):
    pso_results = []
    hybrid_results = []

    for _ in range(runs):
        _, pso_fit = run_pso()
        _, hybrid_fit = run_hybrid()

        pso_results.append(pso_fit)
        hybrid_results.append(hybrid_fit)

    return pso_results, hybrid_results



def plot_results(pso, hybrid):
    x = np.arange(1, len(pso) + 1)

    pso_mean = np.mean(pso)
    hybrid_mean = np.mean(hybrid)

    plt.figure()

    # main lines
    plt.plot(x, pso, label="PSO")
    plt.plot(x, hybrid, label="Hybrid")

    # mean lines
    plt.axhline(pso_mean, linestyle='--', linewidth=2, label=f"PSO Mean = {pso_mean:.2f}")
    plt.axhline(hybrid_mean, linestyle='-.', linewidth=2, label=f"Hybrid Mean = {hybrid_mean:.2f}")

    plt.xlabel("Run")
    plt.ylabel("Fitness")
    plt.title("PSO vs Hybrid Performance")

    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    pso, hybrid = collect_results(30)
    plot_results(pso, hybrid)