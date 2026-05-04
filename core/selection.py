import random


class Selection:
    def __init__(self):
        pass

    def tournament_selection(self, population, k=3):
        """Tournament selection - returns one parent"""
        best = None
        for _ in range(k):
            x = random.randint(0, len(population) - 1)
            if (best is None) or (population[x].fitness < population[best].fitness):
                best = x
        return population[best]

    def roulette_wheel_selection(self, population):
        """Roulette wheel selection - returns list of 2 parents"""
        # Convert fitness (lower is better) to score (higher is better)
        scores = [1.0 / (1.0 + ind.fitness) for ind in population]
        
        total = sum(scores)
        probs = [s / total for s in scores]
        
        # Select 2 parents
        selected = []
        for _ in range(2):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probs):
                cumulative += prob
                if r <= cumulative:
                    selected.append(population[i])
                    break
        
        return selected

    def select_parent(self, method, population, k=3):
        """Unified parent selection method"""
        if method == "tournament":
            return self.tournament_selection(population, k)
        elif method == "roulette":
            parents = self.roulette_wheel_selection(population)
            return parents[0] if parents else None
        else:
            raise ValueError("Method must be 'tournament' or 'roulette'")