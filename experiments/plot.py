import matplotlib.pyplot as plt
import numpy as np
import random

from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid
from algorithms.ga import run_ga


# =========================
# Collect Results
# =========================
def collect_results(runs=30):
    pso_results = []
    ga_results = []
    hybrid_results = []

    for i in range(runs):
        random.seed(i)
        np.random.seed(i)

        # PSO
        _, pso_fit = run_pso()

        # GA (نفس الإعدادات)
        _, ga_fit = run_ga(mutation_type=1, crossover_type=1, seed=i)

        # Hybrid (نفس الإعدادات المهمة)
        _, hybrid_fit = run_hybrid(
            mutation_type=1,
            crossover_type=1,
            selection_type=1
        )

        pso_results.append(pso_fit)
        ga_results.append(ga_fit)
        hybrid_results.append(hybrid_fit)

    return pso_results, ga_results, hybrid_results


# =========================
# Plot Results
# =========================
def plot_results(pso, ga, hybrid):

    plt.style.use('seaborn-v0_8')

    x = np.arange(1, len(pso) + 1)

    pso_mean = np.mean(pso)
    ga_mean = np.mean(ga)
    hybrid_mean = np.mean(hybrid)

    # =========================
    # 1. Line Plot (3 algorithms)
    # =========================
    plt.figure(figsize=(10, 6))

    plt.plot(x, pso, marker='o', linewidth=2, label="PSO")
    plt.plot(x, ga, marker='s', linewidth=2, label="GA")
    plt.plot(x, hybrid, marker='^', linewidth=2, label="Hybrid")

    plt.axhline(pso_mean, linestyle='--', linewidth=1.5, label=f"PSO Mean = {pso_mean:.2f}")
    plt.axhline(ga_mean, linestyle='--', linewidth=1.5, label=f"GA Mean = {ga_mean:.2f}")
    plt.axhline(hybrid_mean, linestyle='--', linewidth=1.5, label=f"Hybrid Mean = {hybrid_mean:.2f}")

    plt.title("PSO vs GA vs Hybrid (Across Runs)", weight='bold')
    plt.xlabel("Run Number")
    plt.ylabel("Fitness")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("line_all.png", dpi=300)

    # =========================
    # 2. Box Plot
    # =========================
    plt.figure(figsize=(8, 5))

    box = plt.boxplot(
        [pso, ga, hybrid],
        labels=["PSO", "GA", "Hybrid"],
        patch_artist=True
    )

    colors = ['skyblue', 'orange', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title("Fitness Distribution (All Algorithms)", weight='bold')
    plt.ylabel("Fitness")

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig("box_all.png", dpi=300)

    # =========================
    # 3. Bar Chart
    # =========================
    plt.figure(figsize=(8, 5))

    labels = ["PSO", "GA", "Hybrid"]
    means = [pso_mean, ga_mean, hybrid_mean]

    bars = plt.bar(labels, means)

    # show values
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y,
                 f"{y:.2f}", ha='center', va='bottom')

    plt.title("Mean Fitness Comparison", weight='bold')
    plt.ylabel("Mean Fitness")

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig("bar_all.png", dpi=300)

    plt.show()


# =========================
# Run
# =========================
if __name__ == "__main__":
    pso, ga, hybrid = collect_results(30)
    plot_results(pso, ga, hybrid)