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

    raise NotImplementedError

    # sol_range = objfunc.up_lim - objfunc.low_lim

    # fit_order = np.argsort(fitness)
    # sorted_population = population[fit_order]
    # sorted_fitness = fitness[fit_order]

    # distances = sp_dist.squareform(sp_dist.pdist())[:, :, None]
    # beta = beta_0 * np.exp(-gamma * distances**2)

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

    sol_range = objfunc.up_lim - objfunc.low_lim
    dist_matrix = sp_dist.pdist(population)
    fit_order = np.argsort(fitness)
    for idx_i, i in enumerate(fit_order):
        for idx_j, j in enumerate(fit_order[:i]):
            if fit_j > fit_i:
                alpha = alpha_0 * delta**idx_j
                beta = beta_0*np.exp(-gamma * (dist_matrix[idx_i, idx_j]/sol_range)**2 / population.shape[1])
                population[idx_i,:] = solution[idx_i,:] + beta * (population[idx_j, :] - population[idx_i, :]) + alpha * RAND_GEN.random()
    
    return population


def glowworm(population, luciferin, fitness, rho, gamma, radial_range, step_size):
    if np.asarray(radial_range).ndim == 0:
        radial_range = np.full(population.shape[0], radial_range)

    # Update luciferin
    luciferin_next = (1-rho)*luciferin + gamma * fitness

    # Update movement
    dist_matrix = sp_dist.pdist(population)

    for idx, ind in enumerate(population):
        # Select neighborhood
        neighbor_idx, *_ = np.where(dist_matrix[idx,:] < radial_range[idx])
        sum_dist = dist_matrix[idx,neighbor_idx].sum()
        for idx_n in neighbor_idx:
            p = (luciferin[idx_n] - luciferin[idx])/sum_dist
            if RAND_GEN.random() < p:
                population[idx] += step_size * (population[idx_n] - population[idx]) / sp_dist.euclidean(population[idx_n], population[idx])
    
    return population, luciferin_next

