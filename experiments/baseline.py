import numpy as np
import random

from simulation.traffic_simulation import TrafficSimulation
from algorithms.pso import run_pso
from algorithms.hybrid import run_hybrid


def run_baseline_experiment():

    sim = TrafficSimulation(num_intersections=6)

    baseline_results = []
    hybrid_results = []

    for i in range(30):
        random.seed(i)
        np.random.seed(i)

        # -------- Baseline --------
        baseline_timings = [30] * 6
        avg_wait, congestion = sim.run(
            baseline_timings,
            steps=200,
            arrival_rate=0.7,
            seed=i
        )
        baseline_fitness = avg_wait + 1.5 * congestion
        baseline_results.append(baseline_fitness)

        # -------- Hybrid --------
        _, hybrid_fit = run_hybrid()
        hybrid_results.append(hybrid_fit)

    print("\n===== BASELINE vs HYBRID =====")

    print("\nBaseline:")
    print("Mean:", np.mean(baseline_results))
    print("Std:", np.std(baseline_results))

    print("\nHybrid:")
    print("Mean:", np.mean(hybrid_results))
    print("Std:", np.std(hybrid_results))


if __name__ == "__main__":
    run_baseline_experiment()