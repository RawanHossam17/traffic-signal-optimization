def calculate_fitness(individual, simulation):
    """
    Evaluate the fitness of an individual using the traffic simulation.
    Objective = min(Wait Time + Congestion Penalty)
    """
    timings = individual.genome
    avg_wait, congestion = simulation.run(timings, steps=200)
    
    # Balanced fitness function - lower is better
    congestion_penalty = congestion * 0.3
    fitness = avg_wait + congestion_penalty
    
    individual.fitness = fitness
    return fitness