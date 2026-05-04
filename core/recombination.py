import random


class Recombination:
    def __init__(self):
        pass

    def uniform_crossover(self, genome1, genome2):
        """Uniform crossover - each gene chosen randomly from either parent"""
        child = []
        for g1, g2 in zip(genome1, genome2):
            if random.random() < 0.5:
                child.append(g1)
            else:
                child.append(g2)
        return child

    def one_point_crossover(self, genome1, genome2):
        """One-point crossover - split at random point and swap segments"""
        if len(genome1) < 2:
            return genome1.copy()
        
        point = random.randint(1, len(genome1) - 1)
        child = genome1[:point] + genome2[point:]
        return child