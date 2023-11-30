import math
import random
import numpy as np
import scipy as sp
import scipy.stats
import enum
from enum import Enum
from ..utils import RAND_GEN


class ProbDist(Enum):
    UNIFORM = enum.auto()
    GAUSS = enum.auto()
    CAUCHY = enum.auto()
    LAPLACE = enum.auto()
    GAMMA = enum.auto()
    EXPON = enum.auto()
    LEVYSTABLE = enum.auto()
    POISSON = enum.auto()
    BERNOULLI = enum.auto()
    CUSTOM = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in prob_dist_map:
            raise ValueError(f'Probability distribution "{str_input}" not defined')

        return prob_dist_map[str_input]


prob_dist_map = {
    "uniform": ProbDist.UNIFORM,
    "gauss": ProbDist.GAUSS,
    "normal": ProbDist.GAUSS,
    "cauchy": ProbDist.CAUCHY,
    "laplace": ProbDist.LAPLACE,
    "gamma": ProbDist.GAMMA,
    "exp": ProbDist.EXPON,
    "expon": ProbDist.EXPON,
    "exponential": ProbDist.EXPON,
    "levystable": ProbDist.LEVYSTABLE,
    "levy_stable": ProbDist.LEVYSTABLE,
    "poisson": ProbDist.POISSON,
    "bernoulli": ProbDist.BERNOULLI,
    "custom": ProbDist.CUSTOM,
}


def mutate_sample(vector, population, params):
    """
    Replaces 'n' components of the input vector with a random value sampled from a given probability distribution.
    """

    n = round(params["N"])
    distrib = params["distrib"]

    loc = params.get("loc")
    scale = params.get("scale")
    if distrib == ProbDist.UNIFORM and "max" in params and "min" in params:
        minim = params["min"]
        maxim = params["max"]
        loc = minim
        scale = maxim - minim
    strength = params.get("F", 1)

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    RAND_GEN.shuffle(mask_pos)

    popul_matrix = np.vstack([i.genotype for i in population])
    if loc is None:
        loc = popul_matrix.mean(axis=0)[mask_pos]
    if scale is None:
        scale = popul_matrix.std(axis=0)[mask_pos]

    rand_vec = sample_distribution(distrib, n, loc, scale, params)

    vector[mask_pos] = rand_vec
    return vector


def mutate_noise(vector, population, params):
    """
    Adds random noise with a given probability distribution to 'n' components of the input vector.
    """

    n = round(params["N"])
    distrib = params["distrib"]

    loc = params.get("loc")
    scale = params.get("scale")
    if distrib == ProbDist.UNIFORM and "max" in params and "min" in params:
        minim = params["min"]
        maxim = params["max"]
        loc = minim
        scale = maxim - minim
    strength = params.get("F", 1)

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    RAND_GEN.shuffle(mask_pos)

    rand_vec = sample_distribution(distrib, n, loc, scale, params)

    vector[mask_pos] = vector[mask_pos] + strength * rand_vec
    return vector


def rand_sample(vector, population, params):
    """
    Picks a vector with components sampled from a probability distribution.
    """

    distrib = params["distrib"]

    loc = params.get("loc")
    scale = params.get("scale")
    if distrib == ProbDist.UNIFORM and "max" in params and "min" in params:
        minim = params["min"]
        maxim = params["max"]
        loc = minim
        scale = maxim - minim
    strength = params.get("F", 1)

    popul_matrix = np.vstack([i.genotype for i in population])
    if loc is None:
        loc = popul_matrix.mean(axis=0)
    if scale is None:
        scale = popul_matrix.std(axis=0)

    rand_vec = sample_distribution(distrib, vector.shape, loc, scale, params)

    return rand_vec


def rand_noise(vector, params):
    """
    Adds random noise with a given probability distribution to all components of the input vector.
    """

    distrib = params["distrib"]

    loc = params.get("loc")
    scale = params.get("scale")
    if distrib == ProbDist.UNIFORM and "max" in params and "min" in params:
        minim = params["min"]
        maxim = params["max"]
        loc = minim
        scale = maxim - minim
    strength = params.get("F", 1)

    noise = sample_distribution(distrib, vector.shape, loc, scale, params)

    return vector + strength * noise


def sample_distribution(distrib, n, loc=None, scale=None, params={}):
    """
    Takes 'n' samples from a given probablility distribution and returns them as a vector.
    """

    loc = 0 if loc is None else loc
    scale = 1 if scale is None else scale

    if distrib == ProbDist.GAUSS:
        prob_distrib = sp.stats.norm(loc=loc, scale=scale)
    elif distrib == ProbDist.UNIFORM:
        prob_distrib = sp.stats.uniform(loc=loc, scale=scale)
    elif distrib == ProbDist.CAUCHY:
        prob_distrib = sp.stats.cauchy(loc=loc, scale=scale)
    elif distrib == ProbDist.LAPLACE:
        prob_distrib = sp.stats.laplace(loc=loc, scale=scale)
    elif distrib == ProbDist.GAMMA:
        a = params.get("a", 1)
        prob_distrib = sp.stats.gamma(a, loc=loc, scale=scale)
    elif distrib == ProbDist.EXPON:
        prob_distrib = sp.stats.expon(loc=loc, scale=scale)
    elif distrib == ProbDist.LEVYSTABLE:
        a = params.get("a", 2)
        b = params.get("b", 0)
        prob_distrib = sp.stats.levy_stable(a, b, loc=loc, scale=scale)
    elif distrib == ProbDist.POISSON:
        mu = params.get("mu", 0)
        prob_distrib = sp.stats.poisson(mu, loc=loc)
    elif distrib == ProbDist.BERNOULLI:
        p = params.get("p", 0)
        prob_distrib = sp.stats.bernoulli(p, loc=loc)
    elif distrib == ProbDist.CUSTOM:
        if "distrib_class" not in params:
            raise Exception("To use a custom probability distribution you must specify it with the 'distrib_class' parameter.")
        prob_dist = params["distrib_class"]

    return prob_distrib.rvs(size=n, random_state=RAND_GEN)


def gaussian(vector, strength):
    """
    Adds random noise following a Gaussian distribution to the vector.
    """

    return rand_noise(vector, {"distrib": ProbDist.GAUSS, "F": strength})


def cauchy(vector, strength):
    """
    Adds random noise following a Cauchy distribution to the vector.
    """

    return rand_noise(vector, {"distrib": ProbDist.CAUCHY, "F": strength})


def laplace(vector, strength):
    """
    Adds random noise following a Laplace distribution to the vector.
    """

    return rand_noise(vector, {"distrib": ProbDist.LAPLACE, "F": strength})


def uniform(vector, minim, maxim):
    """
    Adds random noise following an Uniform distribution to the vector.
    """

    return rand_noise(vector, {"distrib": ProbDist.UNIFORM, "min": minim, "max": maxim})


def poisson(vector, mu):
    """
    Adds random noise following a Poisson distribution to the vector.
    """

    return rand_noise(vector, {"distrib": ProbDist.POISSON, "F": mu})


def sample_1_sigma(vector, n, epsilon, tau):
    """
    Replaces 'n' components of the input vector with a value sampled from the mutate 1 sigma function.

    In future, it should be integrated in mutate_sample and sample_distribution functions, considering
    np.exp(tau * N(0,1)) as a distribution function with a minimum value of epsilon.
    """

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    RAND_GEN.shuffle(mask_pos)

    sampled = np.array([mutate_1_sigma(vector[pos], epsilon, tau) for pos in mask_pos])
    vector[mask_pos] = sampled[mask_pos]
    return vector


def mutate_1_sigma(sigma, epsilon, tau):
    """
    Mutate a sigma value in base of tau param, where epsilon is de minimum value that a sigma can have.
    """

    return np.maximum(epsilon, np.exp(tau * RAND_GEN.normal()))


def mutate_n_sigmas(sigmas, epsilon, tau, tau_multiple):
    """
    Mutate a list of sigmas values in base of tau and tau_multiple params, where epsilon is de minimum value that a sigma can have.
    """

    return np.maximum(epsilon, sigmas * np.exp(tau * RAND_GEN.normal() + tau_multiple * RAND_GEN.normal()))


def xor_mask(vector, n, mode="byte"):
    """
    Applies an XOR operation between a random number and the input vector.
    """

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    RAND_GEN.shuffle(mask_pos)

    if mode == "bin":
        mask = mask_pos
    elif mode == "byte":
        mask = RAND_GEN.integers(1, 0xFF, size=vector.shape) * mask_pos
    elif mode == "int":
        mask = RAND_GEN.integers(1, 0xFFFF, size=vector.shape) * mask_pos

    return vector ^ mask


def permutation(vector, n):
    """
    Randomly permutes 'n' of the components of the input vector.
    """

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    RAND_GEN.shuffle(mask_pos)

    if np.count_nonzero(mask_pos == 1) < 2:
        mask_pos[random.sample(range(mask_pos.size), 2)] = 1

    shuffled_vec = vector[mask_pos]
    RAND_GEN.shuffle(shuffled_vec)
    vector[mask_pos] = shuffled_vec

    return vector


def roll(vector, n):
    """
    Rolls a selection of components of the vector.
    """

    roll_start = random.randrange(0, vector.size - 2)
    roll_end = random.randrange(roll_start, vector.size)

    vector[roll_start:roll_end] = np.roll(vector[roll_start:roll_end], n)
    return vector


def invert_mutation(vector):
    """
    Inverts the order a selection of components of the vector.
    """

    seg_start = random.randrange(0, vector.size - 2)
    seg_end = random.randrange(seg_start, vector.size)

    vector[seg_start:seg_end] = np.flip(vector[seg_start:seg_end])
    return vector


def cross_1p(vector1, vector2):
    """
    Performs a 1 point cross between two vectors.
    """

    cross_point = random.randrange(0, vector1.size)
    return np.hstack([vector1[:cross_point], vector2[cross_point:]])


def cross_2p(vector1, vector2):
    """
    Performs a 2 point cross between two vectors.
    """

    cross_point1 = random.randrange(0, vector1.size - 2)
    cross_point2 = random.randrange(cross_point1, vector1.size)
    return np.hstack(
        [
            vector1[:cross_point1],
            vector2[cross_point1:cross_point2],
            vector1[cross_point2:],
        ]
    )


def cross_mp(vector1, vector2):
    """
    Performs a multipoint cross between two vectors.
    """

    mask_pos = 1 * (RAND_GEN.random(vector1.size) > 0.5)
    aux = np.copy(vector1)
    aux[mask_pos == 1] = vector2[mask_pos == 1]
    return aux


def multi_cross(vector, population, n_ind):
    """
    Performs a multipoint cross between the vector and 'n-1' individuals of the population
    """

    if n_ind >= len(population):
        n_ind = len(population)

    other_parents = random.sample(population, n_ind - 1)
    mask_pos = RAND_GEN.integers(n_ind, size=vector.size) - 1
    for i in range(0, n_ind - 1):
        vector[mask_pos == i] = other_parents[i].genotype[mask_pos == i]
    return vector


def xor_cross(vector1, vector2):
    """
    Applies the XOR operation between each component of the input vectors.
    """

    return vector1 ^ vector2


def pmx(vector1, vector2):
    """
    Partially mapped crossover.

    Taken from https://github.com/cosminmarina/A1_ComputacionEvolutiva
    """

    cross_point1 = random.randrange(0, vector1.size - 2)
    cross_point2 = random.randrange(cross_point1, vector1.size)

    # Segmentamos
    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    # Lo que no forma parte del segmento
    remaining = vector1[~seg_mask]
    segment = vector2[seg_mask]

    # Separamos en conjunto dentro y fuera del segmento del genotipo 2
    overlap = np.isin(remaining, segment)
    conflicting = remaining[overlap]
    no_conflict = np.sort(remaining[~overlap])

    # Añadimos los elementos sin conflicto (que no están dentro del segmento del genotipo 2)
    idx_no_conflict = np.where(np.isin(vector2, no_conflict))[0]
    child[idx_no_conflict] = no_conflict

    # Tratamos conflicto
    for elem in conflicting:
        pos = elem.copy()
        while pos != -1:
            genotype_in_pos = pos
            pos = child[np.where(vector2 == genotype_in_pos)][0]
        child[np.where(vector2 == genotype_in_pos)] = elem
    return child


def order_cross(vector1, vector2):
    cross_point1 = random.randrange(0, vector1.size - 2)
    cross_point2 = random.randrange(cross_point1, vector1.size)

    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    remianing_unused = np.setdiff1d(vector2, child)
    remianing_unused = np.roll(remianing_unused, cross_point1)

    child[~seg_mask] = remianing_unused

    return child


def cross_inter_avg(vector, population, n_ind):
    """
    Performs an intermediate average crossover between the vector and 'n-1' individuals the population
    """

    if n_ind >= len(population):
        n_ind = len(population)

    other_parents = random.sample(population, n_ind - 1)
    parents = [parent.genotype for parent in other_parents] + [vector]
    return np.mean(parents, axis=0)


def weighted_average(vector1, vector2, alpha):
    """
    Performs a weighted average between the two given vectors
    """

    return alpha * vector1 + (1 - alpha) * vector2


def blxalpha(vector1, vector2, alpha):
    """
    Performs the BLX alpha crossing operator between two vectors.
    """

    alpha *= RAND_GEN.random()
    return alpha * vector1 + (1 - alpha) * vector2


def sbx(vector1, vector2, strength):
    """
    Performs the SBX crossing operator between two vectors.
    """

    beta = np.zeros(vector1.shape)
    u = RAND_GEN.random(vector1.shape)
    for idx, val in enumerate(u):
        if val <= 0.5:
            beta[idx] = (2 * val) ** (1 / (strength + 1))
        else:
            beta[idx] = (0.5 * (1 - val)) ** (1 / (strength + 1))

    sign = random.choice([-1, 1])
    return 0.5 * (vector1 + vector2) + sign * 0.5 * beta * (vector1 - vector2)


def DE_rand1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/rand/1
    """

    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = r1.genotype + F * (r2.genotype - r3.genotype)
        mask_pos = RAND_GEN.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DE_best1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/best/1
    """

    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = best.genotype + F * (r1.genotype - r2.genotype)
        mask_pos = RAND_GEN.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DE_rand2(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/rand/2
    """

    if len(population) > 5:
        r1, r2, r3, r4, r5 = random.sample(population, 5)

        v = r1.genotype + F * (r2.genotype - r3.genotype) + F * (r4.genotype - r5.genotype)
        mask_pos = RAND_GEN.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DE_best2(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/best/2
    """

    if len(population) > 5:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2, r3, r4 = random.sample(population, 4)

        v = best.genotype + F * (r1.genotype - r2.genotype) + F * (r3.genotype - r4.genotype)
        mask_pos = RAND_GEN.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DE_current_to_rand1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-rand/1
    """

    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = vector + RAND_GEN.random() * (r1.genotype - vector) + F * (r2.genotype - r3.genotype)
        mask_pos = RAND_GEN.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DE_current_to_best1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-best/1
    """

    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = vector + F * (best.genotype - vector) + F * (r1.genotype - r2.genotype)
        mask_pos = RAND_GEN.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DE_current_to_pbest1(vector, population, F, CR, P):
    """
    Performs the differential evolution operator DE/current-to-pbest/1
    """

    if len(population) > 3:
        fitness = [i.fitness for i in population]
        upper_idx = max(1, math.ceil(len(population) * P))
        pbest_idx = random.choice(np.argsort(fitness)[:upper_idx])
        pbest = population[pbest_idx]
        r1, r2 = random.sample(population, 2)

        v = vector + F * (pbest.genotype - vector) + F * (r1.genotype - r2.genotype)
        mask_pos = RAND_GEN.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def pso_operator(indiv, population, global_best, w, c1, c2):
    """
    Performs a step of the Particle Swarm algorithm
    """

    c1 = c1 * RAND_GEN.random(indiv.genotype.shape)
    c2 = c2 * RAND_GEN.random(indiv.genotype.shape)

    indiv.speed = w * indiv.speed + c1 * (indiv.best - indiv.genotype) + c2 * (global_best.genotype - indiv.genotype)
    return indiv.apply_speed()


def firefly(solution, population, objfunc, alpha_0, beta_0, delta, gamma):
    """
    Performs a step of the Firefly algorithm
    """

    sol_range = objfunc.up_lim - objfunc.low_lim
    n_dim = solution.genotype.size
    new_vector = solution.genotype.copy()
    for idx, ind in enumerate(population):
        if solution.fitness < ind.fitness:
            r = np.linalg.norm(solution.genotype - ind.genotype)
            alpha = alpha_0 * delta**idx
            beta = beta_0 * np.exp(-gamma * (r / (sol_range * np.sqrt(n_dim))) ** 2)
            new_vector = new_vector + beta * (ind.genotype - new_vector) + alpha * sol_range * random.random() - 0.5
            new_vector = objfunc.repair_solution(new_vector)

    return new_vector


def dummy_op(vector, scale=1000):
    """
    Replaces the vector with one consisting of all the same value

    Only for testing, not useful for real applications
    """

    return np.ones(vector.shape) * scale
