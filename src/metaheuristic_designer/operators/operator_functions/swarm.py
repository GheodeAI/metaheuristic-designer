import numpy as np
import scipy as sp
import scipy.spatial.distance as sp_dist
from ...utils import RAND_GEN


def pso_operator(population, population_speed, historical_best, global_best, w, c1, c2):
    """
    Performs a step of the Particle Swarm algorithm
    """

    global_best = global_best[None, :]

    c1 = c1 * RAND_GEN.random(population.shape)
    c2 = c2 * RAND_GEN.random(population.shape)

    speed = w * population_speed + c1 * (historical_best - population) + c2 * (global_best - population)

    return population + speed, speed


def firefly(population, fitness, objfunc, alpha_0, beta_0, delta, gamma):
    """
    Performs a step of the Firefly algorithm
    """

    raise Exception("Not implemented")

    sol_range = objfunc.up_lim - objfunc.low_lim

    fit_order = np.argsort(fitness)
    sorted_population = population[fit_order]
    sorted_fitness = fitness[fit_order]

    distances = sp_dist.squareform(sp_dist.pdist())[:, :, None]
    beta = beta_0 * np.exp(-gamma * distances**2)

    # sol_range = objfunc.up_lim - objfunc.low_lim
    # n_dim = solution.genotype.size
    # new_vector = solution.genotype.copy()
    # for idx, ind in enumerate(population):
    #     if solution.fitness < ind.fitness:
    #         r = np.linalg.norm(solution.genotype - ind.genotype)
    #         alpha = alpha_0 * delta**idx
    #         beta = beta_0 * np.exp(-gamma * (r / (sol_range * np.sqrt(n_dim))) ** 2)
    #         new_vector = new_vector + beta * (ind.genotype - new_vector) + alpha * sol_range * RAND_GEN.random() - 0.5
    #         new_vector = objfunc.repair_solution(new_vector)

    # return new_vector
