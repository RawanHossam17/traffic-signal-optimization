import random
from algorithms.pso import run_pso
from core.initialization import Initialization
from core.selection import Selection
from core.Replacement import Replacement
from core.fitness import calculate_fitness
from simulation.traffic_simulation import TrafficSimulation
from core.individual import Individual


def run_hybrid():

    best_solution, best_fitness = run_pso()
    print("\nPSO Best:", best_solution, best_fitness)

    sim = TrafficSimulation(num_intersections=6)
    init = Initialization()

    population = init.initialize_population(
        method="random",
        population_size=10,
        num_intersections=6
    )

    # inject PSO solution into half the population
    for i in range(5):
        population[i].genome = best_solution.copy()

    # evaluate initial population
    for ind in population:
        calculate_fitness(ind, sim)

    selector = Selection()
    replacer = Replacement()

    # GA loop
    for _ in range(10):

        new_population = []

        for _ in range(len(population)):

            p1 = selector.tournament_selection(population, k=3)
            p2 = selector.tournament_selection(population, k=3)

            # crossover
            child_genome = [
                int((g1 + g2) / 2) for g1, g2 in zip(p1.genome, p2.genome)
            ]

            # mutation
            child_genome = [
                max(10, min(60, g + random.randint(-3, 3)))
                for g in child_genome
            ]

            child = Individual(num_intersections=4, genome=child_genome)

            calculate_fitness(child, sim)
            new_population.append(child)

        population = replacer.elitism_replacement(
            population,
            new_population,
            elite_size=2
        )

    best = min(population, key=lambda x: x.fitness)

    print("\nHybrid Best:", best.genome, best.fitness)

    return best.genome, best.fitness