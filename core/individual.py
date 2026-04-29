import numpy as np


class Individual:
    def __init__(self, num_intersections, C=60, genome=None):
        self.num_intersections = num_intersections
        self.C = C

        if genome is None:
            self.genome = self.random_genome()
        else:
            self.genome = genome

        self.phenotype = self.decode()

    def random_genome(self):
        genome = []

        for _ in range(self.num_intersections):
            g1 = np.random.randint(10, 50)
            g1 = round(g1 / 5) * 5

            genome.append(g1)

        return genome

    def decode(self):
        phenotype = []

        for i, g1 in enumerate(self.genome):
            g2 = self.C - g1

            phenotype.append({
                "intersection": i,
                "green north": g1,
                "green south": g2
            })

        return phenotype

    def __repr__(self):
        return f"Individual(genome={self.genome})"