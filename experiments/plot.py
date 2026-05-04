import matplotlib.pyplot as plt
import numpy as np
from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid


def collect_results(runs=30):
    pso_results = []
    hybrid_results = []

    for i in range(runs):
        import random
        import numpy as np

        random.seed(i)
        np.random.seed(i)

        _, pso_fit = run_pso()
        _, hybrid_fit = run_hybrid()

        pso_results.append(pso_fit)
        hybrid_results.append(hybrid_fit)

    return pso_results, hybrid_results


def plot_results(pso, hybrid):
    x = np.arange(1, len(pso) + 1)

    pso_mean = np.mean(pso)
    hybrid_mean = np.mean(hybrid)

    # =========================
    # 1. Line Plot
    # =========================
    plt.figure()

    plt.plot(x, pso, marker='o', linewidth=2, label="PSO")
    plt.plot(x, hybrid, marker='s', linewidth=2, label="Hybrid")

    plt.axhline(pso_mean, linestyle='--', linewidth=2, label=f"PSO Mean = {pso_mean:.2f}")
    plt.axhline(hybrid_mean, linestyle='-.', linewidth=2, label=f"Hybrid Mean = {hybrid_mean:.2f}")

    plt.xlabel("Run", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title("PSO vs Hybrid (Line Plot)", fontsize=14)

    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig("line_plot.png", dpi=300)

    # =========================
    # 2. Box Plot
    # =========================
    plt.figure()

    plt.boxplot([pso, hybrid], labels=["PSO", "Hybrid"])

    plt.title("Fitness Distribution (Box Plot)", fontsize=14)
    plt.ylabel("Fitness", fontsize=12)

    plt.grid()

    plt.savefig("box_plot.png", dpi=300)

    # =========================
    # 3. Bar Chart
    # =========================
    plt.figure()

    labels = ["PSO", "Hybrid"]
    means = [pso_mean, hybrid_mean]

    plt.bar(labels, means)

    plt.title("Mean Fitness Comparison", fontsize=14)
    plt.ylabel("Mean Fitness", fontsize=12)

    plt.grid(axis='y')

    plt.savefig("bar_chart.png", dpi=300)

 
    plt.show()


if __name__ == "__main__":
    pso, hybrid = collect_results(30)
    plot_results(pso, hybrid)