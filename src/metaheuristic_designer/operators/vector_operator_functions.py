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
    BINOMIAL = enum.auto()
    CATEGORICAL = enum.auto()
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
    "gaussian": ProbDist.GAUSS,
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
    "binomial": ProbDist.BINOMIAL,
    "categorical": ProbDist.CATEGORICAL,
    "custom": ProbDist.CUSTOM,
}


class multicategorical:
    def __init__(categories, weight_matrix):
        self.categories = categories
        weight_matrix = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)
        self.cumsum_matrix = weight_matrix.cumsum(axis=1)
        self.sample_fn = np.vectorize(np.searchsorted, signature="(n),()->()", cache=True)

    def rvs(size=None, random_state=None):
        if size is None:
            size = len(self.cumsum_matrix.shape[0])

        if random_state is None:
            random_state = np.random.default_rng()

        index_rnd = random_state.uniform(0, 1, size=size)
        return self.sample_fn(self.cumsum_matrix, index_rnd)


def cross_1p(population):
    """
    Performs a 1-point crossover between one half of the population and the rest.
    """

    half_size = population.shape[0] / 2
    parents1 = population[: math.ceil(half_size)]
    parents2 = population[math.floor(half_size) :]

    cross_points = RAND_GEN.integers(1, population.shape[1], (parents1.shape[0], 1))
    idx_matrix = np.tile(np.arange(population.shape[1]), parents1.shape[0]).reshape(parents1.shape)
    cross_mask = idx_matrix < cross_points

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population.shape[0]]


def cross_2p(population):
    """
    Performs a 2-point crossover between one half of the population and the rest.
    """

    half_size = population.shape[0] / 2
    parents1 = population[: math.ceil(half_size)]
    parents2 = population[math.floor(half_size) :]

    cross_points1 = RAND_GEN.integers(1, population.shape[1] - 1, (parents1.shape[0], 1))
    cross_points2 = RAND_GEN.integers(cross_points1 + 1, parents1.shape[1], (parents1.shape[0], 1))

    idx_matrix = np.tile(np.arange(population.shape[1]), parents1.shape[0]).reshape(parents1.shape)
    slice_mask1 = idx_matrix < cross_points1
    slice_mask2 = idx_matrix >= cross_points2
    cross_mask = slice_mask1 | slice_mask2

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population.shape[0]]


def cross_mp(population):
    """
    Performs a multipoint crossover between one half of the population and the rest.
    """

    half_size = population.shape[0] / 2
    parents1 = population[: math.ceil(half_size)]
    parents2 = population[math.floor(half_size) :]

    cross_mask = RAND_GEN.uniform(0, 1, parents1.shape) < 0.5

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population.shape[0]]


def weighted_average_cross(population, alpha):
    """
    Performs a weighted average between each individual and a random member of the population.
    """

    population_shuffled = population[RAND_GEN.permutation(population.shape[0]), :]

    return (1 - alpha) * population + alpha * population_shuffled


def blxalpha(population, alpha, lower_bounds=None, upper_bounds=None):
    """
    Performs the BLX alpha crossing operator between two vectors.
    """

    parent_min = np.min(population, axis=0)
    parent_max = np.max(population, axis=0)
    diff = alpha * (parent_max - parent_min)
    low = population - diff
    high = population + diff

    return RAND_GEN.uniform(0, 1, population.shape) * (high - low) + low


def sbx(population, strength):
    """
    Performs the SBX crossing operator between two vectors.
    """

    population_shuffled = population[RAND_GEN.permutation(population.shape[0])]

    beta = np.empty_like(population)
    u = RAND_GEN.uniform(0, 1, population.shape)

    _val1 = (2 * u) ** (1 / (strength + 1))
    _val2 = (0.5 * (1 - u)) ** (1 / (strength + 1))

    beta[u <= 0.5] = _val1[u <= 0.5]
    beta[u > 0.5] = _val2[u > 0.5]
    sign = RAND_GEN.choice([-1, 1], size=population.shape[0])

    return 0.5 * (population + population_shuffled) + sign * 0.5 * beta * (population - population_shuffled)


def xor_cross(population):
    """
    Applies the XOR operation between each component of individuals in the population. The crossover is performed
    between the first and second half of the population
    """

    population_shuffled = population[RAND_GEN.permutation(population.shape[0])]

    return population ^ population_shuffled


def multi_cross(population, n_indiv):
    """
    Performs a multipoint crossover between 'n_indiv' randomly chosen individuals for each member of the population.
    """

    n_indiv = np.minimum(n_indiv, population.shape[0])

    indiv_chosen = np.tile(np.arange(population.shape[0]), n_indiv).reshape((n_indiv, population.shape[0]))
    indiv_chosen = RAND_GEN.permuted(indiv_chosen, axis=1).T

    selection_mask = RAND_GEN.integers(0, n_indiv, population.shape)

    components_chosen = indiv_chosen[np.arange(indiv_chosen.shape[0])[:, None], selection_mask]

    return population[components_chosen, np.arange(population.shape[1])]


def cross_inter_avg(population, n_indiv):
    """
    Performs an intermediate average crossover between the vector and 'n-1' individuals the population.
    """

    n_indiv = np.minimum(n_indiv, population.shape[0])

    # TODO: individuals should be chosen with replacement
    for i in range(n_indiv):
        population_shuffled = population[RAND_GEN.permutation(population.shape[0]), :]
        population += population_shuffled

    return population / n_indiv


def xor_mask(population, n, mode="byte"):
    """
    Applies an XOR operation between a random number and the input vector.
    """

    mask_pos = np.tile(np.arange(population.shape[1]) < n, population.shape[0]).reshape(population.shape)
    mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

    if mode == "bin":
        mask = mask_pos
    elif mode == "byte":
        mask = RAND_GEN.integers(1, 0xFF, size=population.shape) * mask_pos
    elif mode == "int":
        mask = RAND_GEN.integers(1, 0xFFFF, size=population.shape) * mask_pos

    return population ^ mask


def gaussian_mutation(population, strength):
    """
    Adds random noise following a Gaussian distribution to the vector.
    """

    return rand_noise(population, distrib=ProbDist.GAUSS, F=strength)


def cauchy_mutation(population, strength):
    """
    Adds random noise following a Cauchy distribution to the vector.
    """

    return rand_noise(population, distrib=ProbDist.CAUCHY, F=strength)


def laplace_mutation(population, strength):
    """
    Adds random noise following a Laplace distribution to the vector.
    """

    return rand_noise(population, distrib=ProbDist.LAPLACE, F=strength)


def uniform_mutation(population, strength):
    """
    Adds random noise following an Uniform distribution to the vector.
    """

    return rand_noise(population, distrib=ProbDist.UNIFORM, F=strength, min=-1, max=1)


def poisson_mutation(population, strength, mu):
    """
    Adds random noise following a Poisson distribution to the vector.
    """

    return rand_noise(population, distrib=ProbDist.POISSON, F=strength, mu=mu)


def mutate_sample(population, **params):
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

    mask_pos = np.tile(np.arange(population.shape[1]) < n, population.shape[0]).reshape((population.shape[0], n))
    mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

    if loc is None or (type(loc) is str and loc == "calculated"):
        loc = population[mask_pos].mean(axis=0)
    if scale is None or (type(scale) is str and scale == "calculated"):
        scale = population[mask_pos].std(axis=0)

    rand_vec = sample_distribution((population.shape[0], n), loc, scale, params)

    population[mask_pos] = population
    return population


def mutate_noise(population, **params):
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

    mask_pos = np.tile(np.arange(population.shape[1]) < n, population.shape[0]).reshape((population.shape[0], n))
    mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

    rand_vec = sample_distribution((population.shape[0], n), loc, scale, params)

    population[mask_pos] = population[mask_pos] + strength * rand_vec
    return population


def rand_sample(population, **params):
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

    if loc is None or (type(loc) is str and loc == "calculated"):
        loc = population.mean(axis=0)
    if scale is None or (type(scale) is str and scale == "calculated"):
        scale = population.std(axis=0)

    rand_population = sample_distribution(popuplation.shape, loc, scale, params)

    return rand_population


def rand_noise(population, **params):
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

    noise = sample_distribution(population.shape, loc, scale, **params)

    return population + strength * noise


def sample_distribution(n, loc=None, scale=None, **params):
    """
    Takes 'n' samples from a given probablility distribution and returns them as a vector.
    """

    distrib = params["distrib"]
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
        p = params.get("p", 0.5)
        prob_distrib = sp.stats.bernoulli(p, loc=loc)
    elif distrib == ProbDist.BINOMIAL:
        n = params["n"]
        p = params.get("p", 0.5)
        prob_distrib = sp.stats.binomial(n, p, loc=loc)
    elif distrib == ProbDist.CATEGORICAL:
        p = params["p"]
        prob_distrib = sp.stats.rv_discrete(name="categorical", values=(np.arange(p.size), p / np.sum(p)))
    elif distrib == ProbDist.MULTICATEGORICAL:
        p = params["p"]
        prob_distrib = mulitcategorial(np.arange(p.shape[0]), weight_matrix=p)
    elif distrib == ProbDist.CUSTOM:
        if "distrib_class" not in params:
            raise Exception("To use a custom probability distribution you must specify it with the 'distrib_class' parameter.")
        prob_distrib = params["distrib_class"]
    else:
        raise ValueError("Invalid probability distribution")

    return prob_distrib.rvs(size=n, random_state=RAND_GEN)


def generate_statistic(population, **params):
    stat_name = params.get("statistic", "mean")

    new_population = None
    if stat_name == "mean":
        new_population = np.mean(population, axis=0)
    elif stat_name == "average":
        weights = params.get("weights", np.ones(population.shape[1]))
        new_population = np.average(population, weights=weights, axis=0)
    elif stat_name == "median":
        new_population = np.median(population, axis=0)
    elif stat_name == "std":
        new_population = np.std(population, axis=0)

    return new_population


def sample_1_sigma(population, n, epsilon, tau):
    """
    Replaces 'n' components of the input vector with a value sampled from the mutate 1 sigma function.

    In future, it should be integrated in mutate_sample and sample_distribution functions, considering
    np.exp(tau * N(0,1)) as a distribution function with a minimum value of epsilon.
    """

    mask_pos = np.tile(np.arange(population.shape[1]) < n, population.shape[0]).reshape((population.shape[0], n))
    mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

    sampled = np.maximum(epsilon, population * np.exp(tau * RAND_GEN.normal(0, 1, sigma.shape[0])))
    population[mask_pos] = sampled[mask_pos]
    return vector


def mutate_1_sigma(population, epsilon, tau):
    """
    Mutate a sigma value in base of tau param, where epsilon is de minimum value that a sigma can have.
    """

    return np.maximum(epsilon, population * np.exp(tau * RAND_GEN.normal(0, 1, sigma.shape[0])))


def mutate_n_sigmas(population, epsilon, tau, tau_multiple):
    """
    Mutate a list of sigmas values in base of tau and tau_multiple params, where epsilon is de minimum value that a sigma can have.
    """

    return np.maximum(
        epsilon,
        population * np.exp(tau * RAND_GEN.normal(0, 1, population.shape[0]) + tau_multiple * RAND_GEN.normal(0, 1, population.shape[0])),
    )


def permutation(vector, n):
    """
    Randomly permutes 'n' of the components of the input vector.
    """

    mask_pos = np.tile(np.arange(population.shape[1]) < n, population.shape[0]).reshape((population.shape[0], n))
    mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

    if np.count_nonzero(mask_pos == 1) < 2:
        mask_pos[random.sample(range(mask_pos.size), 2)] = 1

    shuffled_vec = vector[mask_pos]
    RAND_GEN.shuffle(shuffled_vec)
    vector[mask_pos] = shuffled_vec

    return vector


def roll(population, n, random_state=None):
    """
    Rolls a selection of components of the vector.
    """
    if random_state is None:
        random_state = np.random.default_rng()
    # roll_start = RAND_GEN.integers(0, population.shape[1] - 2, population.shape[0])
    # roll_end = RAND_GEN.integers(roll_start, population.shape[1], population.shape[0])
    roll_start = random_state.integers(0, population.shape[1] - 2, population.shape[0])
    roll_end = random_state.integers(roll_start, population.shape[1], population.shape[0])

    def roll_individual(indiv, start, end, n):
        individual[roll_start:roll_endidual] = np.roll(indiv[roll_start:roll_end], n)
        return individual

    roll_vec = np.vectorize(roll_individual, signature="(m,n),(m),(m),()->(m,n)")
    population = roll_vec(population, roll_start, roll_end, n)
    # for idx, indiv in enumerate(population):
    #     population[idx, roll_start[idx]:roll_end[idx]] = np.roll(indiv[roll_start[idx]:roll_end[idx]], n)
    return population


def invert_mutation(vector):
    """
    Inverts the order a selection of components of the vector.
    """

    seg_start = RAND_GEN.integers(0, vector.size - 2)
    seg_end = RAND_GEN.integers(seg_start, vector.size)

    for idx, indiv in enumerate(population):
        population[idx, seg_start[idx] : seg_end[idx]] = np.flip(indiv[seg_start[idx] : seg_end[idx]], n)

    return vector


def pmx(vector1, vector2):
    """
    Partially mapped crossover.

    Taken from https://github.com/cosminmarina/A1_ComputacionEvolutiva
    """

    cross_point1 = RAND_GEN.integers(0, vector1.size - 2)
    cross_point2 = RAND_GEN.integers(cross_point1, vector1.size)

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
    cross_point1 = RAND_GEN.integers(0, vector1.size - 2)
    cross_point2 = RAND_GEN.integers(cross_point1, vector1.size)

    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    remianing_unused = np.setdiff1d(vector2, child)
    remianing_unused = np.roll(remianing_unused, cross_point1)

    child[~seg_mask] = remianing_unused

    return child


def DE_rand1(population, F, CR):
    """
    Performs the differential evolution operator DE/rand/1
    """

    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]

    v = r1 + F * (r2 - r3)
    mask_pos = RAND_GEN.uniform(0, 1, population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_best1(population, fitness, F, CR):
    """
    Performs the differential evolution operator DE/best/1
    """

    r_best = population[np.argmax(fitness)][None, :]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]

    v = r_best + F * (r1 - r2)
    mask_pos = RAND_GEN.uniform(0, 1, population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_rand2(population, F, CR):
    """
    Performs the differential evolution operator DE/rand/2
    """

    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]
    r4 = population[RAND_GEN.permutation(population.shape[0])]
    r5 = population[RAND_GEN.permutation(population.shape[0])]

    v = r1 + F * (r2 - r3) + F * (r4 - r5)
    mask_pos = RAND_GEN.uniform(0, 1, population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_best2(population, fitness, F, CR):
    """
    Performs the differential evolution operator DE/best/2
    """

    r_best = population[np.argmax(fitness)][None, :]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]
    r4 = population[RAND_GEN.permutation(population.shape[0])]

    v = r_best + F * (r1 - r2) + F * (r3 - r4)
    mask_pos = RAND_GEN.uniform(0, 1, population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_current_to_rand1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-rand/1
    """

    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]
    r4 = population[RAND_GEN.permutation(population.shape[0])]

    v = population + RAND_GEN.uniform(0, 1) * (r1 - population) + F * (r2 - r3)
    mask_pos = RAND_GEN.uniform(0, 1, population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_current_to_best1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-best/1
    """

    r_best = population[np.argmax(fitness)][None, :]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]

    v = population + RAND_GEN.uniform(0, 1) * (r_best - population) + F * (r1 - r2)
    mask_pos = RAND_GEN.uniform(0, 1, population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_current_to_pbest1(vector, population, F, CR, P):
    """
    Performs the differential evolution operator DE/current-to-pbest/1
    """

    n_best_max_idx = np.ceil(population.shape[0] * P)
    n_best_idx = np.argsort(fitness)[::-1][:n_best_max_idx]
    chosen_idx = np.choice(n_best_idx, replace=True, size=population.shape[0])

    r_best = population[chosen_idx][None, :]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]

    v = population + RAND_GEN.uniform(0, 1) * (r_best - population) + F * (r1 - r2)
    mask_pos = RAND_GEN.uniform(0, 1, population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def pso_operator(population, population_speed, historical_best, global_best, w, c1, c2):
    """
    Performs a step of the Particle Swarm algorithm
    """

    global_best = global_best[None, :]

    c1 = c1 * RAND_GEN.uniform(0, 1, population.shape)
    c2 = c2 * RAND_GEN.uniform(0, 1, population.shape)

    speed = w * population_speed + c1 * (historical_best - population) + c2 * (global_best - population)

    return population + speed, speed


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
            new_vector = new_vector + beta * (ind.genotype - new_vector) + alpha * sol_range * RAND_GEN.uniform(0, 1) - 0.5
            new_vector = objfunc.repair_solution(new_vector)

    return new_vector


def dummy_op(population, scale=1000):
    """
    Replaces the vector with one consisting of all the same value

    Only for testing, not useful for real applications
    """

    return np.full(population.shape, scale)
