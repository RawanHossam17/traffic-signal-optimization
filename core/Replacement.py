import random


class Replacement:
    def __init__(self):
        pass

    def elitism_replacement(self, old_population, new_population, elite_size=2):
        """Keep best individuals from old population"""
        # Sort by fitness (lower is better)
        sorted_old = sorted(old_population, key=lambda ind: ind.fitness)
        elites = sorted_old[:elite_size]
        remaining = new_population[:len(new_population) - elite_size]
        return elites + remaining

    def round_robin_tournament(self, population, q=3):
        """Round-robin tournament selection"""
        N = len(population)
        scores = [0] * N

        for i in range(N):
            # Select q random opponents
            indices = list(range(N))
            indices.remove(i)
            opponents = random.sample(indices, min(q, N-1))
            
            for j in opponents:
                if population[i].fitness < population[j].fitness:
                    scores[i] += 1

        # Sort by score (higher is better)
        sorted_indices = sorted(range(N), key=lambda x: scores[x], reverse=True)
        return [population[i] for i in sorted_indices]

    def replace(self, method, old_population=None, new_population=None, 
                population=None, elite_size=2, q=3):
        if method == "elitism":
            return self.elitism_replacement(old_population, new_population, elite_size)
        elif method == "round_robin":
            return self.round_robin_tournament(population, q)
        else:
            raise ValueError("Method must be 'elitism' or 'round_robin'")