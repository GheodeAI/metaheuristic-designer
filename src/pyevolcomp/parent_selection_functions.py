import numpy as np
import random

def select_best(population, amount):
    """
    Selects the best parent of the population as parents
    """

    # Get the fitness of all the individuals
    fitness_list = np.fromiter(map(lambda x: x.fitness, population), dtype=float)

    # Get the index of the individuals sorted by fitness
    order = np.argsort(fitness_list)[::-1][:amount]

    # Select the 'amount' best individuals
    parents = [population[i] for i in order]

    return parents, order


def prob_tournament(population, tourn_size, prob):
    """
    Selects the parents for the next generation by tournament
    """

    parent_pool = []
    order = []

    for _ in population:

        # Choose 'tourn_size' individuals for the torunament
        parent_idxs = random.sample(range(len(population)), tourn_size)
        parents = [population[i] for i in parent_idxs]
        fits = [i.fitness for i in parents]

        # Choose one of the individuals
        if random.random() < prob:
            idx = random.randint(0, tourn_size - 1)
        else:
            idx = fits.index(max(fits))

        # Add the individuals to the list
        order.append(parent_idxs[idx])
        parent = parents[idx]
        parent_pool.append(parent)

    return parent_pool, order
