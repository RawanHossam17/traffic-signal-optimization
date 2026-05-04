import random
from algorithms.pso import run_pso
from core.initialization import Initialization
from core.selection import Selection
from core.Replacement import Replacement
from core.fitness import calculate_fitness
from simulation.traffic_simulation import TrafficSimulation
from core.individual import Individual


def run_hybrid(mutation_type=1, crossover_type=1):

    # Run PSO first
    best_solution, best_fitness = run_pso()
    print("\nPSO Best:", best_solution, best_fitness)

    sim = TrafficSimulation(num_intersections=6)
    init = Initialization()

    # Initialize population
    population = init.initialize_population(
        method="random",
        population_size=10,
        num_intersections=6
    )

    # Inject PSO solution into half population
    for i in range(5):
        population[i].genome = best_solution.copy()

    # Evaluate initial population
    for ind in population:
        calculate_fitness(ind, sim)

    selector = Selection()
    replacer = Replacement()

    # GA loop
    for _ in range(10):

        new_population = []

        for _ in range(len(population)):

            # Select parents
            p1 = selector.tournament_selection(population, k=3)
            p2 = selector.tournament_selection(population, k=3)

            # -------- Crossover --------
            if crossover_type == 1:
                # Average crossover (original)
                child_genome = [
                    int((g1 + g2) / 2) for g1, g2 in zip(p1.genome, p2.genome)
                ]
            else:
                # Random gene crossover
                child_genome = [
                    random.choice([g1, g2]) for g1, g2 in zip(p1.genome, p2.genome)
                ]

            # -------- Mutation --------
            if mutation_type == 1:
                # Small safe mutation
                child_genome = [
                    max(10, min(60, g + random.randint(-1, 1)))
                    for g in child_genome
                ]
            else:
                # Partial random reset (20% chance)
                child_genome = [
                    random.randint(10, 60) if random.random() < 0.2 else g
                    for g in child_genome
                ]

            # Create child (fix important bug هنا)
            child = Individual(num_intersections=6, genome=child_genome)

            # Evaluate child
            calculate_fitness(child, sim)
            new_population.append(child)

        # Replacement (elitism)
        population = replacer.elitism_replacement(
            population,
            new_population,
            elite_size=2
        )

    # Get best solution
    best = min(population, key=lambda x: x.fitness)

    print("\nHybrid Best:", best.genome, best.fitness)

    return best.genome, best.fitness