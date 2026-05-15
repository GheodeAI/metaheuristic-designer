"""
Factory and registry for probability distributions
"""

import logging
from typing import Callable, Optional
import scipy as sp

from metaheuristic_designer.utils import MatrixLike
from .probability_distributions import (
    Distribution,
    ScipyMultivarDistribution,
    ScipyUnivarDistribution,
    bernoulli_heuristic,
    binomial_heuristic,
    cauchy_heuristic,
    expon_heuristic,
    gamma_heuristic,
    laplace_heuristic,
    multivariate_categorical,
    multivariate_normal_heuristic,
    normal_heuristic,
    poisson_heuristic,
    tikhinov_fisher_heuristic,
    tikhinov_heuristic,
    uniform_heuristic,
    uniform_param_fix,
)

logger = logging.getLogger(__name__)

# fmt: off
scipy_univar_map = {
    # Normal
    "norm": (sp.stats.norm, None, normal_heuristic),
    "normal": (sp.stats.norm, None, normal_heuristic),
    "gauss": (sp.stats.norm, None, normal_heuristic),
    "gaussian": (sp.stats.norm, None, normal_heuristic),

    # Uniform
    "uniform": (sp.stats.uniform, uniform_param_fix, uniform_heuristic),

    # Cauchy
    "cauchy": (sp.stats.cauchy, None, cauchy_heuristic),

    # Laplace
    "laplace": (sp.stats.laplace, None, laplace_heuristic),

    # Gamma
    "gamma": (sp.stats.gamma, None, gamma_heuristic),

    # Exponential
    "exponential": (sp.stats.expon, None, expon_heuristic),
    "expon": (sp.stats.expon, None, expon_heuristic),
    "exp": (sp.stats.expon, None, expon_heuristic),

    # Levy-stable
    "levy_stable": (sp.stats.levy_stable, None, cauchy_heuristic),
    "levy": (sp.stats.levy_stable, None, cauchy_heuristic),

    # Poisson
    "poisson": (sp.stats.poisson, None, poisson_heuristic),
    
    # Bernoulli
    "bernoulli": (sp.stats.bernoulli, None, bernoulli_heuristic),

    # Binomial
    "binom": (sp.stats.binom, None, binomial_heuristic),
    "binomial": (sp.stats.binom, None, binomial_heuristic),

    # Categorical
    "categorical": (sp.stats.rv_discrete, None, None),

    # Tikhinov
    "tikhinov": (sp.stats.vonmises_line, None, tikhinov_heuristic),
    "vonmises": (sp.stats.vonmises_line, None, tikhinov_heuristic),
}

scipy_multivar_map = {
    # mutlivariate gaussian
    "multigauss": (sp.stats.multivariate_normal, None, multivariate_normal_heuristic),
    "multivariate_gauss": (sp.stats.multivariate_normal, None, multivariate_normal_heuristic),
    "multinormal": (sp.stats.multivariate_normal, None, multivariate_normal_heuristic),
    "multivariate_normal": (sp.stats.multivariate_normal, None, multivariate_normal_heuristic),
    "mvn": (sp.stats.multivariate_normal, None, multivariate_normal_heuristic),

    # dirichlet
    "dirichlet": (sp.stats.dirichlet, None, None),

    # Von-Mises Fisher
    "vonmises_fisher": (sp.stats.vonmises_fisher, None, tikhinov_fisher_heuristic),
}

custom_distrib_map = {
    "multicategorical": (multivariate_categorical, None, None)
}
# fmt: on

# fmt: off
distribution_registry = {
    "scipy-univar": scipy_univar_map,
    "scipy-multivar": scipy_multivar_map,
    "custom": custom_distrib_map
}
# fmt: on


def create_prob_distribution(
    distribution_name: str,
    population_matrix: Optional[MatrixLike] = None,
    parameter_heuristic_fn: Optional[Callable] = None,
    **kwargs
) -> Distribution:
    """Instantiate a probability distribution by name.

    Parameters
    ----------
    distribution_name : str
        Distribution key.  Can use dot-notation (e.g.,
        ``"scipy-univar.norm"``) or a short name (``"norm"``) if
        unambiguous.
    population_matrix : MatrixLike
        Population data used when automatic parameter estimation
        (``"calculated"``) is requested.
    parameter_heuristic_fn : Callable, optional
        An optional callable that overrides the registered heuristic
        for this call.  If ``None``, the registered heuristic (or
        a no-op) is used.
    **kwargs
        Parameters forwarded to the distribution constructor (e.g.,
        ``loc``, ``scale``, ``min``, ``max``).

    Returns
    -------
    Distribution
        A callable distribution object ready for sampling.
    """

    distrib_name_lower = distribution_name.lower()

    if "." in distrib_name_lower:
        distrib_reg_name, distrib_name_lower, *_ = distrib_name_lower.split(".")
    else:
        distrib_reg_name = None
        for k, v in distribution_registry.items():
            if distrib_name_lower in v:
                if distrib_reg_name is None:
                    distrib_reg_name = k
                else:
                    logger.warning("Found duplicate distribuiton '%s' in both registries '%s' and '%s'", distrib_name_lower, distrib_reg_name, k)

        if distrib_reg_name is None:
            raise ValueError("Distribution not found in any registry")

    distrib_fn, param_processor, param_heuristic = distribution_registry[distrib_reg_name][distrib_name_lower]
    if param_processor is not None:
        kwargs = param_processor(**kwargs)
    if parameter_heuristic_fn is None:
        param_heuristic = parameter_heuristic_fn
    if param_heuristic is None:
        param_heuristic = lambda _, **kwargs: kwargs
    kwargs = param_heuristic(population_matrix, **kwargs)

    if distrib_reg_name == "scipy-univar":
        distrib = ScipyUnivarDistribution(distrib_fn, **kwargs)
    elif distrib_reg_name == "scipy-multivar":
        distrib = ScipyMultivarDistribution(distrib_fn, **kwargs)
    elif distrib_reg_name == "custom":
        distrib = distrib_fn(**kwargs)

    return distrib

def add_distribution_entry(
    distribution_class: Distribution,
    distribution_name: str,
    registry: str = "custom",
    param_processor: Optional[Callable] = None,
    heuristic_fn: Optional[Callable] = None,
):
    """
    Register a new probability distribution so that it can be created
    via :func:`create_prob_distribution`.

    Parameters
    ----------
    distribution_class : type[Distribution]
        A concrete :class:`Distribution` subclass (or a callable that
        returns one).  For SciPy distributions, pass the SciPy class
        directly (e.g., ``scipy.stats.norm``); it will be wrapped
        automatically.
    distribution_name : str
        Key used to retrieve this distribution (e.g., ``"my_noise"``).
    registry : str, optional
        Sub-registry to add the distribution to.  Currently supported
        values are ``"scipy-univar"``, ``"scipy-multivar"``, and
        ``"custom"`` (default).  If the registry does not exist, it is
        created.
    param_processor : callable, optional
        A function that receives ``**kwargs`` and returns a modified
        dictionary before passing it to the distribution constructor.
        Useful for converting parameter names (see
        :func:`uniform_param_fix`).
    heuristic_fn : callable, optional
        A function ``(population_matrix, **kwargs) -> kwargs`` that
        computes default parameter values when ``"calculated"`` is
        requested.  If ``None`` (the default), the distribution does
        not support automatic parameter estimation.
    """
    if registry not in distribution_registry:
        distribution_registry[registry] = {}
        logger.info('Added a new distribution registry named "%s"', registry)

    reg_map = distribution_registry[registry]
    if distribution_name in reg_map:
        logger.warning(
            'Overwritten distribution "%s" in registry "%s".',
            distribution_name,
            registry,
        )

    reg_map[distribution_name] = (distribution_class, param_processor, heuristic_fn)

    logger.info(
        'Added a new distribution "%s" in registry "%s".',
        distribution_name,
        registry,
    )


def list_distributions() -> list[str]:
    """
    Return a list of all available distribution keys.

    The keys are formatted as ``"registry.distribution_name"`` and can be
    passed directly to :func:`create_prob_distribution`.

    Returns
    -------
    list of str
        Fully qualified distribution names.
    """
    all_names = []
    for reg_name, reg_map in distribution_registry.items():
        for dist_name in reg_map:
            all_names.append(f"{reg_name}.{dist_name}")
    return all_names