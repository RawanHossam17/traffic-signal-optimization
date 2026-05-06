import numpy as np
from deap import base, creator, tools
from simulation.traffic_simulation import TrafficSimulation
from core.fitness import calculate_fitness


def repair_solution(particle, num_intersections, phases_per_intersection=1, cycle_length=60):
    solution = particle.reshape(num_intersections, phases_per_intersection)

    repaired = np.zeros_like(solution)

    for i in range(num_intersections):
        phases = np.clip(solution[i], 10, 60)


        repaired[i] = phases

    return repaired.flatten()


def setup_pso(num_intersections, phases_per_intersection=1):

    DIMENSIONS = num_intersections * phases_per_intersection

    POP_SIZE = 20
    MAX_GEN = 15

    MIN_POS, MAX_POS = 10, 60
    MIN_SPEED, MAX_SPEED = -5, 5

    C1 = 1.5
    C2 = 1.5

    np.random.seed(42)

    toolbox = base.Toolbox()

    # prevent duplicate creator errors
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, "Particle"):
        creator.create(
            "Particle",
            np.ndarray,
            fitness=creator.FitnessMin,
            speed=None,
            best=None
        )

    def create_particle():
        pos = np.random.randint(10, 61, DIMENSIONS).astype(float)
        pos = repair_solution(pos, num_intersections, phases_per_intersection)

        particle = creator.Particle(pos)
        particle.speed = np.random.uniform(MIN_SPEED, MAX_SPEED, DIMENSIONS)
        particle.best = None
        return particle

    toolbox.register("particle", create_particle)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)

    def update_particle(particle, gbest, W):

        r1 = np.random.rand(particle.size)
        r2 = np.random.rand(particle.size)

        cognitive = C1 * r1 * (particle.best - particle)
        social = C2 * r2 * (gbest - particle)

        particle.speed = W * particle.speed + cognitive + social
        particle.speed = np.clip(particle.speed, MIN_SPEED, MAX_SPEED)

        particle[:] = particle + particle.speed
        particle[:] = np.round(particle)

        # enforce constraints
        particle[:] = repair_solution(particle, num_intersections, phases_per_intersection)

    toolbox.register("update", update_particle)

    return toolbox, POP_SIZE, MAX_GEN


def evaluate_particle(particle, simulation):
    class Dummy:
        pass

    dummy = Dummy()
    dummy.genome = list(particle)

    return calculate_fitness(dummy, simulation)


def run_pso():

    num_intersections = 6
    phases_per_intersection = 1

    simulation = TrafficSimulation(num_intersections=num_intersections)

    toolbox, POP_SIZE, MAX_GEN = setup_pso(num_intersections, phases_per_intersection)

    population = toolbox.population(n=POP_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "min", "avg"]

    gbest = None

    convergence = []   
    for generation in range(MAX_GEN):

        #  Dynamic Inertia Weight
        W = 0.9 - (generation / MAX_GEN) * 0.5

        #  Evaluate
        for particle in population:

            fitness = evaluate_particle(particle, simulation)
            particle.fitness.values = (fitness,)

            # personal best
            if particle.best is None or fitness < particle.best.fitness.values[0]:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = (fitness,)

            # global best
            if gbest is None or fitness < gbest.fitness.values[0]:
                gbest = creator.Particle(particle)
                gbest.fitness.values = (fitness,)

      
        convergence.append(gbest.fitness.values[0])

        #  Update swarm
        for particle in population:
            toolbox.update(particle, gbest, W)

        record = stats.compile(population)
        logbook.record(gen=generation, evals=len(population), **record)

    best_matrix = gbest.reshape(num_intersections, phases_per_intersection)

  
    return best_matrix.flatten().tolist(), gbest.fitness.values[0], convergence

if __name__ == "__main__":
    run_pso()