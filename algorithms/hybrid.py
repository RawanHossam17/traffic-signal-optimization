import random
import numpy as np
from algorithms.pso import run_pso
from core.initialization import Initialization
from core.selection import Selection
from core.recombination import Recombination
from core.mutation import Mutation
from core.Replacement import Replacement
from core.fitness import calculate_fitness
from simulation.traffic_simulation import TrafficSimulation
from core.individual import Individual


def run_hybrid(num_intersections, steps, traffic_level, 
               selection_method="tournament",
               recombination_method="uniform",
               mutation_method="gaussian",
               replacement_method="elitism"):
    
    # Step 1: Run PSO to get a good starting solution
    print("  Running PSO phase...")
    pso_solution, pso_fitness = run_pso(
        num_intersections=num_intersections,
        steps=steps,
        traffic_level=traffic_level
    )
    print(f"    PSO best fitness: {pso_fitness:.2f}")
    
    # Step 2: Initialize GA population with PSO solution
    init = Initialization()
    
    population = []
    
    # Add PSO solution (elite)
    elite = Individual(
        num_intersections=num_intersections,
        genome=pso_solution.copy()
    )
    population.append(elite)
    
    # Add random individuals for diversity
    random_pop = init.initialize_population(
        method="random",
        population_size=24,
        num_intersections=num_intersections
    )
    population.extend(random_pop)
    
    # Evaluate all individuals
    for ind in population:
        sim = TrafficSimulation(
            num_intersections=num_intersections,
            traffic_level=traffic_level
        )
        calculate_fitness(ind, sim)
    
    selector = Selection()
    recombiner = Recombination()
    mutator = Mutation()
    replacer = Replacement()
    
    crossover_rate = 0.8
    mutation_rate = 0.15
    
    # Step 3: GA evolution
    print("  Running GA phase...")
    for generation in range(20):
        new_population = []
        
        # Keep best 2 solutions
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        elites = sorted_pop[:2]
        
        while len(new_population) < len(population) - 2:
            # Select parents
            if selection_method == "tournament":
                p1 = selector.select_parent("tournament", population, k=3)
                p2 = selector.select_parent("tournament", population, k=3)
            else:  # roulette
                # Roulette returns a list of parents
                parents = selector.roulette_wheel_selection(population)
                if len(parents) >= 2:
                    p1, p2 = parents[0], parents[1]
                else:
                    # Fallback to tournament if not enough parents
                    p1 = selector.select_parent("tournament", population, k=3)
                    p2 = selector.select_parent("tournament", population, k=3)
            
            # Crossover
            if random.random() < crossover_rate:
                if recombination_method == "uniform":
                    child_genome = recombiner.uniform_crossover(p1.genome, p2.genome)
                else:
                    child_genome = recombiner.one_point_crossover(p1.genome, p2.genome)
            else:
                child_genome = p1.genome.copy()
            
            # Mutation
            if mutation_method == "uniform":
                child_genome = mutator.uniform_mutation(child_genome, mutation_rate)
            else:
                child_genome = mutator.gaussian_mutation(child_genome, mutation_rate, 3)
            
            # Create and evaluate child
            child = Individual(num_intersections=num_intersections, genome=child_genome)
            sim = TrafficSimulation(num_intersections=num_intersections, traffic_level=traffic_level)
            calculate_fitness(child, sim)
            
            new_population.append(child)
        
        # Replacement
        if replacement_method == "elitism":
            population = replacer.elitism_replacement(population, new_population, 2)
        else:
            population = replacer.round_robin_tournament(elites + new_population, 3)
        
        best = min(population, key=lambda x: x.fitness)
        
        if generation % 5 == 0:
            avg = np.mean([ind.fitness for ind in population])
            print(f"    Gen {generation}: Best={best.fitness:.2f}, Avg={avg:.2f}")
    
    # Final result
    best = min(population, key=lambda x: x.fitness)
    print(f"    Final fitness: {best.fitness:.2f}")
    
    return best.genome, best.fitness