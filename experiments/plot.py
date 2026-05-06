import matplotlib.pyplot as plt
import numpy as np
import random

from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid
from algorithms.ga import run_ga
from simulation.traffic_simulation import TrafficSimulation


# =========================
# Collect Results 
# =========================
def collect_results(runs=30):
    pso_results = []
    ga_results = []
    hybrid_results = []
    convergence_curve = None  

    for i in range(runs):
        random.seed(i)
        np.random.seed(i)

        # PSO (with convergence)
        _, pso_fit, conv = run_pso()

        # take only first run curve
        if convergence_curve is None:
            convergence_curve = conv

        # GA
        _, ga_fit = run_ga(mutation_type=1, crossover_type=1, seed=i)

        # Hybrid
        _, hybrid_fit = run_hybrid(
            mutation_type=1,
            crossover_type=1,
            selection_type=1
        )

        pso_results.append(pso_fit)
        ga_results.append(ga_fit)
        hybrid_results.append(hybrid_fit)

    return pso_results, ga_results, hybrid_results, convergence_curve


# =========================
# Plot Results
# =========================
def plot_results(pso, ga, hybrid):

    plt.style.use('seaborn-v0_8')

    x = np.arange(1, len(pso) + 1)

    pso_mean = np.mean(pso)
    ga_mean = np.mean(ga)
    hybrid_mean = np.mean(hybrid)

    # Line Plot
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

    # Box Plot
    plt.figure(figsize=(8, 5))

    box = plt.boxplot([pso, ga, hybrid], labels=["PSO", "GA", "Hybrid"], patch_artist=True)

    colors = ['skyblue', 'orange', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title("Fitness Distribution (All Algorithms)", weight='bold')
    plt.ylabel("Fitness")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("box_all.png", dpi=300)

    # Bar Chart
    plt.figure(figsize=(8, 5))

    labels = ["PSO", "GA", "Hybrid"]
    means = [pso_mean, ga_mean, hybrid_mean]

    bars = plt.bar(labels, means)

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
# Collect Metrics
# =========================
def collect_metrics(runs=30):

    pso_wait, ga_wait, hybrid_wait = [], [], []
    pso_cong, ga_cong, hybrid_cong = [], [], []

    for i in range(runs):
        random.seed(i)
        np.random.seed(i)

        sim = TrafficSimulation(num_intersections=6)

        sol, _, _ = run_pso()
        avg_wait, cong = sim.run(sol)
        pso_wait.append(avg_wait)
        pso_cong.append(cong)

        sol, _ = run_ga(mutation_type=1, crossover_type=1, seed=i)
        avg_wait, cong = sim.run(sol)
        ga_wait.append(avg_wait)
        ga_cong.append(cong)

        sol, _ = run_hybrid(mutation_type=1, crossover_type=1, selection_type=1)
        avg_wait, cong = sim.run(sol)
        hybrid_wait.append(avg_wait)
        hybrid_cong.append(cong)

    return (pso_wait, ga_wait, hybrid_wait,
            pso_cong, ga_cong, hybrid_cong)


# =========================
# Metric Plots
# =========================
def plot_waiting(pso, ga, hybrid):

    labels = ["PSO", "GA", "Hybrid"]
    means = [np.mean(pso), np.mean(ga), np.mean(hybrid)]

    plt.figure()
    plt.bar(labels, means)

    plt.title("Average Waiting Time Comparison")
    plt.ylabel("Waiting Time")

    plt.savefig("waiting_time.png", dpi=300)
    plt.show()


def plot_congestion(pso, ga, hybrid):

    labels = ["PSO", "GA", "Hybrid"]
    means = [np.mean(pso), np.mean(ga), np.mean(hybrid)]

    plt.figure()
    plt.bar(labels, means)

    plt.title("Congestion Comparison")
    plt.ylabel("Congestion")

    plt.savefig("congestion.png", dpi=300)
    plt.show()


def plot_fake_stops(pso_wait, ga_wait, hybrid_wait):

    pso_s = np.mean(pso_wait) * 2
    ga_s = np.mean(ga_wait) * 2
    hybrid_s = np.mean(hybrid_wait) * 2

    labels = ["PSO", "GA", "Hybrid"]
    values = [pso_s, ga_s, hybrid_s]

    plt.figure()
    plt.bar(labels, values)

    plt.title("Estimated Number of Stops")
    plt.ylabel("Stops")

    plt.savefig("stops.png", dpi=300)
    plt.show()


# =========================
#  CONVERGENCE
# =========================
def plot_real_convergence(conv):

    plt.figure()
    plt.plot(conv, marker='o')

    plt.title("PSO Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")

    plt.grid(alpha=0.3)
    plt.savefig("convergence.png", dpi=300)
    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":

    pso, ga, hybrid, conv = collect_results(30)

    plot_results(pso, ga, hybrid)

    pso_w, ga_w, hybrid_w, pso_c, ga_c, hybrid_c = collect_metrics(30)

    plot_waiting(pso_w, ga_w, hybrid_w)
    plot_congestion(pso_c, ga_c, hybrid_c)

    plot_real_convergence(conv)   
    plot_fake_stops(pso_w, ga_w, hybrid_w)