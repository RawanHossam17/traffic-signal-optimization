import numpy as np
from deap import base, creator, tools
from simulation.traffic_simulation import TrafficSimulation
from core.fitness import calculate_fitness


def repair_solution(particle, num_intersections, phases_per_intersection=1, cycle_length=60):
    """Repair invalid signal timings"""
    if isinstance(particle, list):
        particle = np.array(particle)
    
    particle = np.clip(particle, 10, 50)
    particle = np.round(particle / 5) * 5
    
    return particle


def setup_pso(num_intersections, phases_per_intersection=1):
    DIMENSIONS = num_intersections * phases_per_intersection
    POP_SIZE = 30
    MAX_GEN = 25
    
    MIN_SPEED, MAX_SPEED = -3, 3
    C1 = 1.8
    C2 = 1.8
    
    # REMOVED: np.random.seed(42)  <-- THIS WAS THE PROBLEM!
    
    toolbox = base.Toolbox()
    
    if creator.__dict__.get('FitnessMin'):
        del creator.FitnessMin
    if creator.__dict__.get('Particle'):
        del creator.Particle
    
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, best=None)
    
    def create_particle():
        pos = np.random.uniform(15, 45, DIMENSIONS)
        pos = repair_solution(pos, num_intersections, phases_per_intersection)
        particle = creator.Particle(pos)
        particle.speed = np.random.uniform(MIN_SPEED, MAX_SPEED, DIMENSIONS)
        particle.best = None
        return particle
    
    toolbox.register("particle", create_particle)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    
    def update_particle(particle, gbest, W):
        r1 = np.random.rand(particle.size)  # Random each time
        r2 = np.random.rand(particle.size)  # Random each time
        
        cognitive = C1 * r1 * (particle.best - particle)
        social = C2 * r2 * (gbest - particle)
        
        particle.speed = W * particle.speed + cognitive + social
        particle.speed = np.clip(particle.speed, MIN_SPEED, MAX_SPEED)
        
        particle[:] = particle + particle.speed
        particle[:] = np.round(particle)
        particle[:] = repair_solution(particle, num_intersections, phases_per_intersection)
    
    toolbox.register("update", update_particle)
    
    return toolbox, POP_SIZE, MAX_GEN


def evaluate_particle(particle, num_intersections, steps, traffic_level):
    simulation = TrafficSimulation(
        num_intersections=num_intersections,
        traffic_level=traffic_level
    )
    
    class Dummy:
        pass
    
    dummy = Dummy()
    dummy.genome = list(particle)
    
    fitness = calculate_fitness(dummy, simulation)
    return fitness


def run_pso(num_intersections, steps, traffic_level):
    phases_per_intersection = 1
    toolbox, POP_SIZE, MAX_GEN = setup_pso(num_intersections, phases_per_intersection)
    
    population = toolbox.population(n=POP_SIZE)
    gbest = None
    gbest_fitness = float('inf')
    
    print(f"Running PSO with {POP_SIZE} particles for {MAX_GEN} generations...")
    
    for generation in range(MAX_GEN):
        W = 0.9 - (generation / MAX_GEN) * 0.4
        
        for particle in population:
            fitness = evaluate_particle(particle, num_intersections, steps, traffic_level)
            particle.fitness.values = (fitness,)
            
            if particle.best is None or fitness < particle.best.fitness.values[0]:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = (fitness,)
            
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest = creator.Particle(particle)
                gbest.fitness.values = (fitness,)
        
        for particle in population:
            toolbox.update(particle, gbest, W)
        
        if generation % 5 == 0:
            avg_fitness = np.mean([p.fitness.values[0] for p in population])
            print(f"  Gen {generation:2d}: Best={gbest_fitness:.2f}, Avg={avg_fitness:.2f}")
    
    return gbest.tolist(), gbest_fitness