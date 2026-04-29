import random
import numpy as np
from .initialization import Initialization as init


class Selection:

    def __init__(self):
        pass

    def tournament_selection(self, population, k):
        best = None

        for _ in range(k):
            x = random.randint(0, len(population) - 1)

            if (best is None) or (population[x].fitness < population[best].fitness):
                best = x

        return population[best]

    def roulette_wheel_selection(self, population):
        parents = []

        scores = [1 / (1 + ind.fitness) for ind in population]

        total = sum(scores)
        probs = [s / total for s in scores]

        cumulative = []
        current = 0
        for p in probs:
            current += p
            cumulative.append(current)

        N = len(population)

        for _ in range(N):
            r = random.uniform(0, 1)

            for i in range(N):
                if r <= cumulative[i]:
                    parents.append(population[i])
                    break

        return parents

    def select_parent(self, method, population, k=3):

        if method == "tournament":
            return self.tournament_selection(population, k)

        elif method == "roulette":
            return self.roulette_wheel_selection(population)