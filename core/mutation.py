import random
import numpy as np


class Mutation:
    def __init__(self):
        pass

    def uniform_mutation(self, genome, rate=0.15):
        """Uniform mutation - random value from valid range"""
        mutated = []
        for g in genome:
            if random.random() < rate:
                # New random value between 10 and 50, rounded to nearest 5
                g = random.randint(10, 50)
                g = round(g / 5) * 5
            mutated.append(g)
        return mutated

    def gaussian_mutation(self, genome, rate=0.15, sigma=3):
        """Gaussian mutation - add small Gaussian noise"""
        mutated = []
        for g in genome:
            if random.random() < rate:
                # Add Gaussian noise
                g = g + np.random.normal(0, sigma)
                # Clip and round
                g = max(10, min(50, g))
                g = round(g / 5) * 5
            mutated.append(g)
        return mutated