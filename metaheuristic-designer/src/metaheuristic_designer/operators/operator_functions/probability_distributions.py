from abc import ABC, abstractmethod
import logging
import numpy as np
import scipy as sp
from ...utils import RNGLike, TensorLike, check_random_state

logger = logging.getLogger(__name__)


# ---------------------------------------------
# Distribution sampling classes
# ---------------------------------------------
class Distribution(ABC):
    """Abstract base class for all probability distributions.

    Defines the interface that every distribution must implement:
    a ``sample`` method to generate random variates and an optional
    ``estimate_parameters`` method to compute heuristic parameters from data.
    """
    @abstractmethod
    def sample(self, shape: tuple, random_state: RNGLike) -> TensorLike:
        """Draw random samples from the distribution.

        Parameters
        ----------
        shape : tuple
            Shape of the requested output array.
        random_state : RNGLike
            Random number generator.

        Returns
        -------
        TensorLike
            Array of samples with the requested shape.
        """

    def estimate_parameters(self, data_matrix, **kwargs):
        """Compute heuristic distribution parameters from a data matrix.

        This method is optional and may be overridden by subclasses.
        The default implementation does nothing and returns ``None``.

        Parameters
        ----------
        data_matrix : np.ndarray
            2-D array of shape ``(N, M)`` containing the data used for
            estimation.
        **kwargs
            Additional keyword arguments that may influence the estimation.

        Returns
        -------
        None
        """
        return None


class ScipyUnivarDistribution(Distribution):
    """Wraps a SciPy univariate distribution.

    Parameters
    ----------
    distribution_cls : type
        A SciPy distribution class (e.g., ``scipy.stats.norm``).
    **kwargs
        Parameters forwarded to the distribution constructor.
    """
    def __init__(self, distribution_cls, **kwargs):
        self.dist = distribution_cls(**kwargs)

    def sample(self, shape: tuple, random_state: RNGLike) -> TensorLike:
        """Draw random samples.

        Parameters
        ----------
        shape : tuple
            Desired output shape, e.g., ``(N, M)``.
        random_state : RNGLike
            Random number generator.

        Returns
        -------
        np.ndarray
            Array of independent samples of the requested shape.
        """
        return self.dist.rvs(size=shape, random_state=random_state)


class ScipyMultivarDistribution(Distribution):
    """Wraps a SciPy multivariate distribution.

    Parameters
    ----------
    distribution_cls : type
        A SciPy multivariate distribution class (e.g., ``scipy.stats.multivariate_normal``).
    **kwargs
        Parameters forwarded to the distribution constructor.
    """
    def __init__(self, distribution_cls, **kwargs):
        self.dist = distribution_cls(**kwargs)

    def sample(self, shape: tuple, random_state: RNGLike) -> TensorLike:
        """Draw random samples.

        For a multivariate distribution, ``shape[1]`` is ignored;
        the output shape is determined by the number of individuals
        (``shape[0]``) and the dimension of the distribution.

        Parameters
        ----------
        shape : tuple
            Requested shape; only ``shape[0]`` is used.
        random_state : RNGLike
            Random number generator.

        Returns
        -------
        np.ndarray
            Array of shape ``(shape[0], dim)``.
        """
        return self.dist.rvs(size=shape[0], random_state=random_state)


class multivariate_categorical(Distribution):
    """Multivariate categorical distribution with per-row probability weights.

    Parameters
    ----------
    categories : array-like
        List of category values.
    weight_matrix : np.ndarray
        2-D array of shape ``(N, K)`` containing non-negative weights for
        each of the ``K`` categories in each row.  Weights are normalised
        row-wise before sampling.
    """
    def __init__(self, categories, weight_matrix):
        self.categories = categories
        weight_matrix = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)
        self.cumsum_matrix = weight_matrix.cumsum(axis=1)
        self.sample_fn = np.vectorize(np.searchsorted, signature="(n),()->()", cache=True)

    def sample(self, shape: tuple, random_state: RNGLike) -> TensorLike:
        """Draw random samples.

        Parameters
        ----------
        shape : tuple
            Requested shape; if ``None`` the number of rows is taken from
            ``self.cumsum_matrix.shape[0]``.  Otherwise the first element
            gives the number of rows, and additional dimensions are appended.
        random_state : RNGLike
            Random number generator.

        Returns
        -------
        np.ndarray
            Array of categorical samples with shape determined by *shape*.
        """
        if shape is None:
            shape = self.cumsum_matrix.shape[0]
        elif np.asarray(shape).ndim == 0:
            shape = (shape, len(self.categories))
        else:
            shape = tuple(shape) + (len(self.categories),)

        random_state = check_random_state(random_state)

        index_rnd = random_state.random(size=shape)
        return self.sample_fn(self.cumsum_matrix, index_rnd)


# ---------------------------------------------
# Parameter reinterpretting functions
# ---------------------------------------------
def uniform_param_fix(min=None, max=None, **kwargs):
    """Convert ``min``/``max`` arguments to ``loc``/``scale`` for a uniform distribution.

    Parameters
    ----------
    min : float or array-like, optional
        Lower bound of the uniform interval.
    max : float or array-like, optional
        Upper bound of the uniform interval.
    **kwargs : dict
        Remaining keyword arguments (e.g., ``loc``, ``scale``).

    Returns
    -------
    dict
        A copy of *kwargs* with ``loc`` and ``scale`` set appropriately
        when both *min* and *max* are provided.  *min* and *max* are
        removed from the dictionary.
    """
    if min is not None and max is not None:
        kwargs["loc"] = min
        kwargs["scale"] = max - min
    return kwargs


# ---------------------------------------------
# Heuristic precomputing of parameters
# ---------------------------------------------
def normal_heuristic(population_matrix, loc=None, scale=None, **kwargs):
    """Heuristic parameter estimation for the normal distribution.

    When *loc* or *scale* is the string ``"calculated"``, the sample mean
    or standard deviation (computed over axis 0) is used.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    loc : None, float, array-like, or ``"calculated"``
        Location parameter. If ``"calculated"``, it is replaced by the
        per-column mean.
    scale : None, float, array-like, or ``"calculated"``
        Scale parameter. If ``"calculated"``, it is replaced by the
        per-column standard deviation.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``loc`` and ``scale``.
    """
    if loc == "calculated":
        kwargs["loc"] = population_matrix.mean(axis=0)
    elif loc is not None:
        kwargs["loc"] = loc

    if scale == "calculated":
        kwargs["scale"] = population_matrix.std(axis=0)
    elif scale is not None:
        kwargs["scale"] = scale

    return kwargs


def uniform_heuristic(population_matrix, loc=None, scale=None, **kwargs):
    """Heuristic parameter estimation for the uniform distribution.

    When *loc* or *scale* is ``"calculated"``, the per-column minimum is
    used as *loc*, and ``max - min`` is used as *scale*.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    loc : None, float, array-like, or ``"calculated"``
        Lower bound. If ``"calculated"``, it is replaced by the per-column
        minimum.
    scale : None, float, array-like, or ``"calculated"``
        Interval length. If ``"calculated"``, it is set to
        ``max - min`` per column.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``loc`` and ``scale``.
    """
    if loc == "calculated":
        kwargs["loc"] = population_matrix.min(axis=0)
    elif loc is not None:
        kwargs["loc"] = loc

    if scale == "calculated":
        kwargs["scale"] = population_matrix.max(axis=0) - population_matrix.min(axis=0)
    elif scale is not None:
        kwargs["scale"] = scale

    return kwargs


def cauchy_heuristic(population_matrix, loc=None, scale=None, **kwargs):
    """Heuristic parameter estimation for the Cauchy distribution.

    *loc* is estimated by the per-column median; *scale* is estimated by
    half the per-column interquartile range (IQR / 2).

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    loc : None, float, array-like, or ``"calculated"``
        Location parameter. If ``"calculated"``, it is replaced by the
        per-column median.
    scale : None, float, array-like, or ``"calculated"``
        Scale parameter. If ``"calculated"``, it is estimated as
        ``scipy.stats.iqr(data, axis=0) / 2``.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``loc`` and ``scale``.
    """
    if loc == "calculated":
        kwargs["loc"] = np.median(population_matrix, axis=0)
    elif loc is not None:
        kwargs["loc"] = loc

    if scale == "calculated":
        kwargs["scale"] = sp.stats.iqr(population_matrix, axis=0) / 2
    elif scale is not None:
        kwargs["scale"] = scale

    return kwargs


def laplace_heuristic(population_matrix, loc=None, scale=None, **kwargs):
    """Heuristic parameter estimation for the Laplace distribution.

    *loc* is estimated by the per-column median; *scale* is estimated by
    the median absolute deviation (MAD).

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    loc : None, float, array-like, or ``"calculated"``
        Location parameter. If ``"calculated"``, it is replaced by the
        per-column median.
    scale : None, float, array-like, or ``"calculated"``
        Scale parameter. If ``"calculated"``, it is estimated using
        ``scipy.stats.median_abs_deviation`` along axis 0.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``loc`` and ``scale``.
    """
    if loc == "calculated":
        kwargs["loc"] = np.median(population_matrix, axis=0)
    elif loc is not None:
        kwargs["loc"] = loc

    if scale == "calculated":
        kwargs["scale"] = sp.stats.median_abs_deviation(population_matrix, axis=0)
    elif scale is not None:
        kwargs["scale"] = scale

    return kwargs


def gamma_heuristic(population_matrix, a=None, scale=None, **kwargs):
    """Heuristic parameter estimation for the gamma distribution.

    Uses method of moments: shape ``a = mean² / variance`` and
    ``scale = variance / mean``.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    a : None, float, array-like, or ``"calculated"``
        Shape parameter. If ``"calculated"``, it is computed as
        ``mean² / var``.
    scale : None, float, array-like, or ``"calculated"``
        Scale parameter. If ``"calculated"``, it is computed as
        ``var / mean``.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``a`` and ``scale``.
    """
    mean = None
    var = None
    if a == "calculated":
        mean = population_matrix.mean(axis=0)
        var = population_matrix.var(axis=0)
        kwargs["a"] = mean * mean / var
    elif a is not None:
        kwargs["a"] = a

    if scale == "calculated":
        if mean is None and var is None:
            mean = population_matrix.mean(axis=0)
            var = population_matrix.var(axis=0)
        kwargs["scale"] = var / mean
    elif scale is not None:
        kwargs["scale"] = scale

    return kwargs


def expon_heuristic(population_matrix, scale=None, **kwargs):
    """Heuristic parameter estimation for the exponential distribution.

    If *scale* is ``"calculated"``, it is set to the per-column mean
    minus *loc* (which defaults to 0).

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    scale : None, float, array-like, or ``"calculated"``
        Scale parameter. If ``"calculated"``, ``scale = mean - loc``.
    **kwargs
        Additional keyword arguments passed through unchanged; expected
        to contain *loc* if a non-zero shift is used.

    Returns
    -------
    dict
        The updated *kwargs* with resolved *scale*.
    """
    if scale == "calculated":
        loc = kwargs.get("loc", 0)
        kwargs["scale"] = population_matrix.mean(axis=0) - loc
    elif scale is not None:
        kwargs["scale"] = scale

    return kwargs


def levy_stable_heuristic(population_matrix, loc=None, scale=None, **kwargs):
    """Heuristic parameter estimation for the Lévy-stable distribution.

    *loc* is estimated by the per-column median; *scale* by the median
    absolute deviation (MAD).  Note: this is a very rough approximation
    and should be used only when no better estimator is available.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    loc : None, float, array-like, or ``"calculated"``
        Location parameter. If ``"calculated"``, it is replaced by the
        per-column median.
    scale : None, float, array-like, or ``"calculated"``
        Scale parameter. If ``"calculated"``, it is estimated using
        ``scipy.stats.median_abs_deviation`` along axis 0.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``loc`` and ``scale``.
    """
    if loc == "calculated":
        kwargs["loc"] = np.median(population_matrix, axis=0)
    elif loc is not None:
        kwargs["loc"] = loc

    if scale == "calculated":
        kwargs["scale"] = sp.stats.median_abs_deviation(population_matrix, axis=0)
    elif scale is not None:
        kwargs["scale"] = scale

    return kwargs


def poisson_heuristic(population_matrix, mu=None, **kwargs):
    """Heuristic parameter estimation for the Poisson distribution.

    If *mu* is ``"calculated"``, it is set to the per-column mean minus
    *loc* (default 0).

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    mu : None, float, array-like, or ``"calculated"``
        Rate parameter. If ``"calculated"``, ``mu = mean - loc``.
    **kwargs
        Additional keyword arguments passed through unchanged; expected
        to contain *loc* if a non-zero shift is used.

    Returns
    -------
    dict
        The updated *kwargs* with resolved *mu*.
    """
    if mu == "calculated":
        loc = kwargs.get("loc", 0)
        kwargs["mu"] = population_matrix.mean(axis=0) - loc
    elif mu is not None:
        kwargs["mu"] = mu

    return kwargs


def bernoulli_heuristic(population_matrix, p=None, **kwargs):
    """Heuristic parameter estimation for the Bernoulli distribution.

    If *p* is ``"calculated"``, it is estimated as the per-column mean
    (proportion of ones), adjusting for *loc* if given (default 0).

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    p : None, float, array-like, or ``"calculated"``
        Success probability. If ``"calculated"``, ``p = mean - loc``.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved *p*.
    """
    if p == "calculated":
        loc = kwargs.get("loc", 0)
        kwargs["p"] = population_matrix.mean(axis=0) - loc
    elif p is not None:
        kwargs["p"] = p

    return kwargs


def binomial_heuristic(population_matrix, p=None, **kwargs):
    """Heuristic parameter estimation for the binomial distribution.

    *n* must be provided externally.  If *p* is ``"calculated"``, it is
    estimated as ``(mean - loc) / n``.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    p : None, float, array-like, or ``"calculated"``
        Success probability. If ``"calculated"``, ``p = (mean - loc) / n``.
    **kwargs
        Additional keyword arguments passed through unchanged; must
        contain the integer *n* (number of trials) and optionally *loc*.

    Returns
    -------
    dict
        The updated *kwargs* with resolved *p*.
    """
    if p == "calculated":
        loc = kwargs.get("loc", 0)
        n = kwargs["n"]
        kwargs["p"] = (population_matrix.mean(axis=0) - loc) / n
    elif p is not None:
        kwargs["p"] = p

    return kwargs


def tikhinov_heuristic(population_matrix, loc=None, kappa=None, **kwargs):
    """Heuristic parameter estimation for the von Mises (circular) distribution.

    *loc* is estimated as the circular mean (``arctan2(mean sin, mean cos)``).
    *kappa* is approximated from the mean resultant length *R* as
    ``kappa = R / (1 - R)``.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)`` of angles in radians.
    loc : None, float, array-like, or ``"calculated"``
        Mean direction. If ``"calculated"``, the per-column circular mean
        is used.
    kappa : None, float, array-like, or ``"calculated"``
        Concentration parameter. If ``"calculated"``, it is approximated
        from the mean resultant length.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``loc`` and ``kappa``.
    """
    mean_cos = None
    mean_sin = None
    if loc == "calculated":
        mean_cos = np.cos(population_matrix).mean(axis=0)
        mean_sin = np.sin(population_matrix).mean(axis=0)
        kwargs["loc"] = np.arctan2(mean_sin, mean_cos)
    elif loc is not None:
        kwargs["loc"] = loc

    if kappa == "calculated":
        if mean_cos is None and mean_sin is None:
            mean_cos = np.cos(population_matrix).mean(axis=0)
            mean_sin = np.sin(population_matrix).mean(axis=0)
        radius = np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
        kwargs["kappa"] = radius / (1 - radius)
    elif kappa is not None:
        kwargs["kappa"] = kappa

    return kwargs


def multivariate_normal_heuristic(population_matrix, mean=None, cov=None, **kwargs):
    """Heuristic parameter estimation for the multivariate normal distribution.

    *mean* is estimated as the per-column average.  Automatic estimation of
    the full covariance matrix is **not supported**, use with caution.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    mean : None, array-like, or ``"calculated"``
        Mean vector. If ``"calculated"``, it is set to the per-column mean.
    cov : None, array-like, or ``"calculated"``
        Covariance matrix. If ``"calculated"``, an error is raised because
        automatic estimation is not implemented.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved *mean* (and possibly *cov*).

    Raises
    ------
    ValueError
        If *cov* is ``"calculated"``, automatic covariance estimation is
        not supported.
    """
    if mean == "calculated":
        kwargs["mean"] = population_matrix.mean(axis=0)
    elif mean is not None:
        kwargs["mean"] = mean

    if cov == "calculated":
        raise ValueError(
            "Automatic covariance estimation is not supported. "
            "Provide an explicit covariance matrix."
        )
    elif cov is not None:
        kwargs["cov"] = cov

    return kwargs


def dirichlet_heuristic(population_matrix, alpha=None, **kwargs):
    """Heuristic parameter estimation for the Dirichlet distribution.

    Automatic estimation of the concentration parameter vector *alpha*
    is **not supported**.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    alpha : None, array-like, or ``"calculated"``
        Concentration parameters. If ``"calculated"``, an error is raised.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved *alpha*.

    Raises
    ------
    ValueError
        If *alpha* is ``"calculated"``, automatic estimation is not
        supported.
    """
    if alpha == "calculated":
        raise ValueError(
            "Automatic Dirichlet parameter estimation is not supported. "
            "Provide an explicit `alpha` vector."
        )
    elif alpha is not None:
        kwargs["alpha"] = alpha

    return kwargs


def tikhinov_fisher_heuristic(population_matrix, loc=None, kappa=None, **kwargs):
    """Heuristic parameter estimation for the von Mises-Fisher distribution.

    *loc* is estimated by normalising the mean vector of the data.
    *kappa* is approximated using the mean resultant length *R* and the
    dimension *d*:
    ``kappa = R * (d - R²) / (1 - R²)``.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, d)`` where each row is a point on the
        *d*-dimensional sphere.
    loc : None, array-like, or ``"calculated"``
        Mean direction (unit vector). If ``"calculated"``, it is set to the
        normalised sample mean.
    kappa : None, float, or ``"calculated"``
        Concentration parameter. If ``"calculated"``, it is approximated
        from the mean resultant length.
    **kwargs
        Additional keyword arguments passed through unchanged.

    Returns
    -------
    dict
        The updated *kwargs* with resolved ``loc`` and ``kappa``.
    """
    sample_mean = None
    if loc == "calculated":
        sample_mean = population_matrix.mean(axis=0)
        radius = np.linalg.norm(sample_mean)
        kwargs["loc"] = sample_mean / radius
    elif loc is not None:
        kwargs["loc"] = loc

    if kappa == "calculated":
        if sample_mean is None:
            sample_mean = population_matrix.mean(axis=0)
            radius = np.linalg.norm(sample_mean)
        else:
            radius = np.linalg.norm(sample_mean)
        dimension = population_matrix.shape[1]
        squared_radius = radius * radius
        kwargs["kappa"] = radius * (dimension - squared_radius) / (1 - squared_radius)
    elif kappa is not None:
        kwargs["kappa"] = kappa

    return kwargs