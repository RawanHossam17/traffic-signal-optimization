import random
import numpy as np
from deap import base, creator, tools, algorithms
from simulation.traffic_simulation import TrafficSimulation
from core.fitness import calculate_fitness

# Avoid duplicate creator error
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Generate gene (10–60)
def random_green():
    return random.randint(10, 60)

toolbox.register("attr_int", random_green)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation using your fitness
def evaluate(individual, seed=None):
    sim = TrafficSimulation(num_intersections=6)
    class Dummy: pass
    dummy = Dummy()
    dummy.genome = list(individual)
    return (calculate_fitness(dummy, sim),)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def run_ga(mutation_type=1, crossover_type=1, seed=None):

    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)

    # Choose crossover
    if crossover_type == 1:
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    else:
        toolbox.register("mate", tools.cxTwoPoint)

    # Choose mutation
    if mutation_type == 1:
        toolbox.register("mutate", tools.mutUniformInt, low=10, up=60, indpb=0.2)
    else:
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

    # Run GA
    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=15,
        halloffame=hof,
        verbose=False
    )

    best = hof[0]
    return list(best), best.fitness.values[0]