def calculate_fitness(individual, simulation):
    """
    Evaluate the fitness of an individual using the traffic simulation.
    Objective = min(Wait Time + Congestion Penalty)
    """
    timings = individual.genome
    avg_wait, congestion = simulation.run(timings, steps=200)
    fitness = avg_wait + 2 * congestion
    individual.fitness = fitness
    return fitness