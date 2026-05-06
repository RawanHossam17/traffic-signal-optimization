import random
import numpy as np

from algorithms.pso import run_pso
from core.initialization import Initialization
from core.selection import Selection
from core.Replacement import Replacement
from core.fitness import calculate_fitness
from simulation.traffic_simulation import TrafficSimulation
from core.individual import Individual


def run_hybrid(mutation_type=1, crossover_type=1, selection_type=1, seed=None):

    # =========================
    #  Fix randomness
    # =========================
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # =========================
    # Step 1: PSO 
    # =========================
    best_solution, best_fitness, _ = run_pso()

    sim = TrafficSimulation(num_intersections=6)
    init = Initialization()

    population = init.initialize_population(
        method="random",
        population_size=10,
        num_intersections=6
    )

    # inject PSO solution 
    population[0].genome = best_solution.copy()
    population[1].genome = best_solution.copy()

    # evaluate
    for ind in population:
        calculate_fitness(ind, sim)

    selector = Selection()
    replacer = Replacement()

    # =========================
    # GA loop 
    # =========================
    for _ in range(12):   

        new_population = []

        for _ in range(len(population)):

            # Selection
            if selection_type == 1:
                p1 = selector.tournament_selection(population, k=5)
                p2 = selector.tournament_selection(population, k=5)
            else:
                p1 = random.choice(population)
                p2 = random.choice(population)

            # Crossover
            if crossover_type == 1:
                child_genome = [
                    int((g1 + g2) / 2) for g1, g2 in zip(p1.genome, p2.genome)
                ]
            else:
                child_genome = [
                    random.choice([g1, g2]) for g1, g2 in zip(p1.genome, p2.genome)
                ]

            # Mutation
            if mutation_type == 1:
                child_genome = [
                    max(10, min(60, g + random.randint(-2, 2)))
                    for g in child_genome
                ]
            else:
                child_genome = [
                    random.randint(10, 60) for _ in child_genome
                ]

            child = Individual(num_intersections=6, genome=child_genome)

            calculate_fitness(child, sim)
            new_population.append(child)

        # Replacement 
        population = replacer.elitism_replacement(
            population,
            new_population,
            elite_size=2   
        )

    best = min(population, key=lambda x: x.fitness)

    return best.genome, best.fitness