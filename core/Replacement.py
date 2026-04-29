import random
import numpy as np


class Replacement:

    def __init__(self):
        pass

    def elitism_replacement(self, old_population, new_population, elite_size):

        sorted_old = sorted(old_population, key=lambda ind: ind.fitness)
        sorted_new = sorted(new_population, key=lambda ind: ind.fitness)

        elites = sorted_old[:elite_size]
        remaining = sorted_new[:len(new_population) - elite_size]

        return elites + remaining

    def round_robin_tournament(self, population, q=3):
        N = len(population)
        scores = [0] * N

        for i in range(N):
            opponents = random.sample(range(N), q)

            for j in opponents:
                if population[i].fitness < population[j].fitness:
                    scores[i] += 1

        sorted_population = [ind for _, ind in sorted(zip(scores, population), reverse=True)]

        return sorted_population

    def replace(self, method,
                old_population=None,
                new_population=None,
                population=None,
                elite_size=2,
                q=3):

        if method == "elitism":
            return self.elitism_replacement(
                old_population,
                new_population,
                elite_size
            )

        elif method == "round_robin":
            return self.round_robin_tournament(
                population,
                q
            )

        else:
            raise ValueError("Method must be 'elitism' or 'round_robin'")