import random
import numpy as np
from deap import base, creator, tools, algorithms
from core.individual import Individual
from core.fitness import calculate_fitness
from simulation.traffic_simulation import TrafficSimulation

# 1. Define Fitness and Individual in DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # minimize fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 2. Attribute generator: random green time (10–50, rounded to 5)
def random_green():
    g = random.randint(10, 50)
    return round(g / 5) * 5

# 3. Structure initializers
toolbox.register("attr_int", random_green)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=6)  # 6 intersections
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 4. Evaluation function using your simulation
sim = TrafficSimulation(num_intersections=6)

def eval_individual(ind):
    avg_wait, congestion = sim.run(ind, steps=200)
    fitness = avg_wait + 2 * congestion
    return (fitness,)  # DEAP expects a tuple

toolbox.register("evaluate", eval_individual)

# 5. Genetic operators
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # uniform crossover
toolbox.register("mutate", tools.mutUniformInt, low=10, up=50, indpb=0.2)  # mutation
toolbox.register("select", tools.selTournament, tournsize=3)

# 6. GA Execution
def run_ga(pop_size=20, generations=15):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # best individual tracker
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    best_solution = hof[0]
    best_fitness = hof[0].fitness.values[0]
    return best_solution, best_fitness

if __name__ == "__main__":
    best_solution, best_fitness = run_ga()
    print("\n===== FINAL RESULT =====")
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
