import numpy as np
from core.individual import Individual


class Initialization:
    def __init__(self, C=60):
        self.C = C

    def random_initialization(self, population_size, num_intersections):
        population = []
        for _ in range(population_size):
            ind = Individual(num_intersections=num_intersections, C=self.C)
            population.append(ind)
        return population

    def heuristic_initialization(self, population_size, num_intersections):
        """Heuristic: green times between 30-45 seconds"""
        population = []
        for _ in range(population_size):
            genome = []
            for _ in range(num_intersections):
                g1 = np.random.randint(30, 45)
                g1 = round(g1 / 5) * 5
                genome.append(g1)
            ind = Individual(num_intersections=num_intersections, C=self.C, genome=genome)
            population.append(ind)
        return population

    def initialize_population(self, method, population_size, num_intersections):
        if method == "random":
            return self.random_initialization(population_size, num_intersections)
        elif method == "heuristic":
            return self.heuristic_initialization(population_size, num_intersections)
        else:
            raise ValueError("Method must be 'random' or 'heuristic'")