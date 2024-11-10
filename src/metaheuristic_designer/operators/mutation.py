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

    mask_pos = np.tile(np.arange(population.shape[1]) < n, population.shape[0]).reshape(population.shape)
    mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

    rand_vec = sample_distribution(population.shape, loc, scale, **params)

    population[mask_pos] = population[mask_pos] + strength * rand_vec[mask_pos]
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


def sample_distribution(shape, loc=None, scale=None, **params):
    """
    Takes samples as a matrix with shape 'shape' from a given probablility distribution and returns them as a vector.
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

    return prob_distrib.rvs(size=shape, random_state=RAND_GEN)


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


def dummy_op(population, scale=1000):
    """
    Replaces the vector with one consisting of all the same value

    Only for testing, not useful for real applications
    """

    return np.full(population.shape, scale)
