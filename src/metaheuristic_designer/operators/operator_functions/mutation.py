"""
Mutation operator implementations based on probability distributions.
"""

import logging
from copy import copy
import enum
from enum import Enum
import numpy as np
import scipy as sp
from ...utils import check_random_state

logger = logging.getLogger(__name__)


class ProbDist(Enum):
    UNIFORM = enum.auto()
    GAUSS = enum.auto()
    MULTIGAUSS = enum.auto()
    CAUCHY = enum.auto()
    LAPLACE = enum.auto()
    GAMMA = enum.auto()
    EXPON = enum.auto()
    LEVYSTABLE = enum.auto()
    POISSON = enum.auto()
    BERNOULLI = enum.auto()
    BINOMIAL = enum.auto()
    VONMISES = enum.auto()
    CATEGORICAL = enum.auto()
    MULTICATEGORICAL = enum.auto()
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
    "multivariate_normal": ProbDist.MULTIGAUSS,
    "multinormal": ProbDist.MULTIGAUSS,
    "multivariate_gauss": ProbDist.MULTIGAUSS,
    "multigauss": ProbDist.MULTIGAUSS,
    "multivariate_gaussian": ProbDist.MULTIGAUSS,
    "multigaussian": ProbDist.MULTIGAUSS,
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
    "binom": ProbDist.BINOMIAL,
    "binomial": ProbDist.BINOMIAL,
    "vonmises": ProbDist.VONMISES,
    "vonmises-fisher": ProbDist.VONMISES,
    "tikhonov": ProbDist.VONMISES,
    "categorical": ProbDist.CATEGORICAL,
    "multicategorical": ProbDist.MULTICATEGORICAL,
    "multivariate_categorical": ProbDist.MULTICATEGORICAL,
    "multivariatecategorical": ProbDist.MULTICATEGORICAL,
    "custom": ProbDist.CUSTOM,
}


class multivariate_categorical:
    def __init__(self, categories, weight_matrix):
        self.categories = categories
        weight_matrix = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)
        self.cumsum_matrix = weight_matrix.cumsum(axis=1)
        self.sample_fn = np.vectorize(np.searchsorted, signature="(n),()->()", cache=True)

    def rvs(self, size=None, random_state=None):
        if size is None:
            size = self.cumsum_matrix.shape[0]
        elif np.asarray(size).ndim == 0:
            size = (size, len(self.categories))
        else:
            size = tuple(size) + (len(self.categories),)

        random_state = check_random_state(random_state)

        index_rnd = random_state.random(size=size)
        return self.sample_fn(self.cumsum_matrix, index_rnd)


def mutate_sample(population, _fitness, random_state=None, **kwargs):
    """
    Replaces 'n' components of the input vector with a random value sampled from a given probability distribution.
    """

    random_state = check_random_state(random_state)

    n = round(kwargs["N"])
    distrib = kwargs["distrib"]

    loc = kwargs.get("loc")
    scale = kwargs.get("scale")
    if "loc" in kwargs:
        del kwargs["loc"]
    if "scale" in kwargs:
        del kwargs["scale"]

    if distrib == ProbDist.UNIFORM and "max" in kwargs and "min" in kwargs:
        minim = kwargs["min"]
        maxim = kwargs["max"]
        loc = minim
        scale = maxim - minim

    mask_pos = np.tile(np.arange(population.shape[1]) < n, (population.shape[0], 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    if loc is None or (isinstance(loc, str) and loc == "calculated"):
        loc = population[mask_pos].mean(axis=0)
    if scale is None or (isinstance(scale, str) and scale == "calculated"):
        scale = population[mask_pos].std(axis=0)

    rand_vec = sample_distribution(population.shape, loc, scale, random_state, **kwargs)

    population[mask_pos] = rand_vec[mask_pos]

    logger.debug("Resampled components of the vector %s, with mask %s", population[mask_pos], mask_pos.astype(int))

    return population


def mutate_noise(population, _fitness, random_state=None, **kwargs):
    """
    Adds random noise with a given probability distribution to 'n' components of the input vector.
    """

    random_state = check_random_state(random_state)

    n = round(kwargs["N"])
    distrib = kwargs["distrib"]

    loc = kwargs.get("loc")
    scale = kwargs.get("scale")

    if "loc" in kwargs:
        del kwargs["loc"]
    if "scale" in kwargs:
        del kwargs["scale"]

    if distrib == ProbDist.UNIFORM and "max" in kwargs and "min" in kwargs:
        minim = kwargs["min"]
        maxim = kwargs["max"]
        loc = minim
        scale = maxim - minim
    strength = np.asarray(kwargs.get("F", 1))
    if strength.ndim == 1:
        strength = strength[:, None]

    mask_pos = np.tile(np.arange(population.shape[1]) < n, (population.shape[0], 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    rand_vec = sample_distribution(population.shape, loc, scale, random_state, **kwargs)

    population[mask_pos] = population[mask_pos] + (strength * rand_vec)[mask_pos]

    logger.debug(
        "Mutated components of the vector:\nvector = %s\nnoise_added = %s\nmask = %s",
        population[mask_pos],
        (strength * rand_vec)[mask_pos],
        mask_pos.astype(int),
    )

    return population


def rand_sample(population, _fitness, random_state=None, **kwargs):
    """
    Picks a vector with components sampled from a probability distribution.
    """

    random_state = check_random_state(random_state)

    distrib = kwargs["distrib"]

    loc = kwargs.get("loc")
    scale = kwargs.get("scale")

    if "loc" in kwargs:
        del kwargs["loc"]
    if "scale" in kwargs:
        del kwargs["scale"]

    if distrib == ProbDist.UNIFORM and "max" in kwargs and "min" in kwargs:
        minim = kwargs["min"]
        maxim = kwargs["max"]
        loc = minim
        scale = maxim - minim

    if loc is None or (isinstance(loc, str) and loc == "calculated"):
        loc = population.mean(axis=0)
    if scale is None or (isinstance(scale, str) and scale == "calculated"):
        scale = population.std(axis=0)

    rand_population = sample_distribution(population.shape, loc, scale, random_state, **kwargs)

    logger.debug("Resampled vector %s", rand_population)

    return rand_population


def rand_noise(population, _fitness, random_state=None, **kwargs):
    """
    Adds random noise with a given probability distribution to all components of the input vector.
    """

    random_state = check_random_state(random_state)

    distrib = kwargs["distrib"]

    loc = kwargs.get("loc")
    scale = kwargs.get("scale")
    if "loc" in kwargs:
        del kwargs["loc"]
    if "scale" in kwargs:
        del kwargs["scale"]

    if distrib == ProbDist.UNIFORM and "max" in kwargs and "min" in kwargs:
        minim = kwargs["min"]
        maxim = kwargs["max"]
        loc = minim
        scale = maxim - minim
    strength = np.asarray(kwargs.get("F", 1))
    if strength.ndim == 1:
        strength = strength[:, None]

    noise = sample_distribution(population.shape, loc, scale, random_state, **kwargs)
    result = population + strength * noise

    logger.debug("Added noise to vector %s", result)

    return result


def sample_distribution(shape, loc=None, scale=None, random_state=None, **kwargs):
    """
    Takes samples as a matrix with shape 'shape' from a given probablility distribution and returns them as a vector.
    """

    random_state = check_random_state(random_state)

    result = None

    distrib = kwargs["distrib"]
    if isinstance(distrib, str):
        distrib = ProbDist.from_str(distrib.lower())
    loc = 0 if loc is None else loc
    scale = 1 if scale is None else scale

    match distrib:
        case ProbDist.GAUSS:
            prob_distrib = sp.stats.norm(loc=loc, scale=scale)
        case ProbDist.MULTIGAUSS:
            mean = kwargs.get("mean", np.full(shape[1], loc) if np.asarray(loc).ndim <= 1 else loc)
            cov = kwargs.get("cov", (np.eye(shape[1]) * scale if np.asarray(scale).ndim <= 1 else np.diagflat(scale)))
            if mean.ndim <= 1 and cov.ndim <= 2:
                prob_distrib = sp.stats.multivariate_normal(mean=mean, cov=cov)
                shape = shape[0]
            else:
                result = np.empty(shape)
                for i in range(shape[0]):
                    mean_i = loc if np.asarray(mean).ndim <= 1 else loc[i]
                    scale_i = scale if np.asarray(cov).ndim <= 2 else scale[i]
                    prob_distrib = sp.stats.multivariate_normal(mean=mean_i, kappa=scale_i)
                    result[i, :] = prob_distrib.rvs(random_state=random_state)
        case ProbDist.UNIFORM:
            prob_distrib = sp.stats.uniform(loc=loc, scale=scale)
        case ProbDist.CAUCHY:
            prob_distrib = sp.stats.cauchy(loc=loc, scale=scale)
        case ProbDist.LAPLACE:
            prob_distrib = sp.stats.laplace(loc=loc, scale=scale)
        case ProbDist.GAMMA:
            a = kwargs.get("a", 1)
            prob_distrib = sp.stats.gamma(a, loc=loc, scale=scale)
        case ProbDist.EXPON:
            prob_distrib = sp.stats.expon(loc=loc, scale=scale)
        case ProbDist.LEVYSTABLE:
            a = kwargs.get("a", 2)
            b = kwargs.get("b", 0)
            prob_distrib = sp.stats.levy_stable(a, b, loc=loc, scale=scale)
        case ProbDist.POISSON:
            mu = kwargs.get("mu", 0)
            prob_distrib = sp.stats.poisson(mu, loc=loc)
        case ProbDist.BERNOULLI:
            p = kwargs.get("p", 0.5)
            prob_distrib = sp.stats.bernoulli(p, loc=loc)
        case ProbDist.VONMISES:
            mu = kwargs.get("mu", random_state.uniform(-1, 1, shape))
            if mu.ndim <= 1:
                mu = mu / np.linalg.norm(mu)
            else:
                mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)
            if mu.ndim <= 1 and np.asarray(scale).ndim <= 1:
                prob_distrib = sp.stats.vonmises_fisher(mu=mu, kappa=1 / scale)
                shape = shape[0]
            else:
                result = np.empty(shape)
                for i in range(shape[0]):
                    mu_i = mu if mu.ndim <= 1 else mu[i]
                    scale_i = scale if np.asarray(scale).ndim <= 1 else scale[i]
                    prob_distrib = sp.stats.vonmises_fisher(mu=mu_i, kappa=1 / scale_i)
                    result[i, :] = prob_distrib.rvs(random_state=random_state)
        case ProbDist.BINOMIAL:
            n = kwargs["n"]
            p = kwargs.get("p", 0.5)
            prob_distrib = sp.stats.binom(n, p, loc=loc)
        case ProbDist.CATEGORICAL:
            p = kwargs["p"]
            prob_distrib = sp.stats.rv_discrete(name="categorical", values=(np.arange(p.size), p / np.sum(p)))
        case ProbDist.MULTICATEGORICAL:
            p = kwargs["p"]
            prob_distrib = multivariate_categorical(np.arange(p.shape[1]), weight_matrix=p)
            shape = shape[0]
        case ProbDist.CUSTOM:
            if "distrib_class" not in kwargs:
                raise ValueError("To use a custom probability distribution you must specify it with the 'distrib_class' parameter.")
            prob_distrib = kwargs["distrib_class"]
        case _:
            raise ValueError(f"Invalid probability distribution {distrib}")

    if result is None:
        result = prob_distrib.rvs(size=shape, random_state=random_state)

    logger.debug("Generated random noise vector %s", result)

    return result


def sample_1_sigma(population, _fitness, random_state=None, **kwargs):
    """
    Replaces 'n' components of the input vector with a value sampled from the mutate 1 sigma function.

    In future, it should be integrated in mutate_sample and sample_distribution functions, considering
    np.exp(tau * N(0,1)) as a distribution function with a minimum value of epsilon.
    """

    random_state = check_random_state(random_state)

    epsilon = kwargs["epsilon"]
    sigma = kwargs["sigma"]
    tau = kwargs["tau"]
    n = kwargs["n"]

    mask_pos = np.tile(np.arange(population.shape[1]) < n, (population.shape[0], 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    sampled = np.maximum(epsilon, population * np.exp(tau * random_state.normal(0, 1, sigma.shape[0])))
    population[mask_pos] = sampled[mask_pos]
    return population


def mutate_1_sigma(population, _fitness, random_state=None, **kwargs):
    """
    Mutate a sigma value in base of tau param, where epsilon is de minimum value that a sigma can have.
    """

    random_state = check_random_state(random_state)

    epsilon = kwargs["epsilon"]
    tau = kwargs["tau"]

    return np.maximum(epsilon, population * np.exp(tau * random_state.normal(0, 1, population.shape[0])[:, None]))


def mutate_n_sigmas(population, _fitness, random_state=None, **kwargs):
    """
    Mutate a list of sigmas values in base of tau and tau_multiple params, where epsilon is de minimum value that a sigma can have.
    """

    random_state = check_random_state(random_state)

    epsilon = kwargs["epsilon"]
    tau = kwargs["tau"]
    tau_multiple = kwargs["tau_multiple"]

    return np.maximum(
        epsilon,
        population
        * np.exp(
            tau * random_state.normal(0, 1, population.shape[0])[:, None] + tau_multiple * random_state.normal(0, 1, population.shape[0])[:, None]
        ),
    )


def xor_mask(population, _fitness, random_state=None, **kwargs):
    """
    Applies an XOR operation between a random number and the input vector.
    """

    random_state = check_random_state(random_state)

    n = kwargs["N"]
    mode = kwargs.get("BinRep", "byte")

    mask_pos = np.tile(np.arange(population.shape[1]) < n, (population.shape[0], 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    match mode:
        case "bin":
            mask = mask_pos
        case "byte":
            mask = random_state.integers(1, 0xFF, size=population.shape) * mask_pos
        case "int":
            mask = random_state.integers(1, 0xFFFF, size=population.shape) * mask_pos
        case _:
            mask = 0

    return population ^ mask
