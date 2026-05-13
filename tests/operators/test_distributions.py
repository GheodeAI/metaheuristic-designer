import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import scipy as sp

from conftest import rng

# factory + classes
from metaheuristic_designer.operators.operator_functions.probability_distributions_factory import (
    create_prob_distribution,
    ScipyUnivarDistribution,
    ScipyMultivarDistribution,
    multivariate_categorical,
)
# heuristics
from metaheuristic_designer.operators.operator_functions.probability_distributions import (
    normal_heuristic, uniform_heuristic, cauchy_heuristic, laplace_heuristic,
    gamma_heuristic, expon_heuristic, poisson_heuristic, bernoulli_heuristic,
    binomial_heuristic, tikhinov_heuristic, multivariate_normal_heuristic,
    dirichlet_heuristic, tikhinov_fisher_heuristic,
)

# -------------------------------------------------------------------
#  Factory & sampling (univariate)
# -------------------------------------------------------------------
def test_create_normal_distribution(rng):
    dist = create_prob_distribution("norm", population_matrix=np.zeros((1,1)), loc=0, scale=1)
    assert isinstance(dist, ScipyUnivarDistribution)
    sample = dist.sample((10, 3), rng)
    assert sample.shape == (10, 3)
    assert np.abs(sample.mean()) < 0.5
    assert 0.5 < sample.std() < 1.5


def test_create_uniform_distribution_with_min_max(rng):
    dist = create_prob_distribution("uniform", population_matrix=np.zeros((1,1)), min=0, max=10)
    sample = dist.sample((100, 5), rng)
    assert sample.shape == (100, 5)
    assert np.all(sample >= 0)
    assert np.all(sample <= 10)


def test_create_uniform_distribution_with_loc_scale(rng):
    dist = create_prob_distribution("uniform", population_matrix=np.zeros((1,1)), loc=5, scale=5)
    sample = dist.sample((100, 3), rng)
    assert np.all(sample >= 5)
    assert np.all(sample <= 10)


def test_create_poisson_distribution(rng):
    dist = create_prob_distribution("poisson", population_matrix=np.zeros((1,1)), mu=3)
    sample = dist.sample((50, 4), rng)
    assert np.all(sample >= 0)
    assert 2.0 < sample.mean() < 4.0


# -------------------------------------------------------------------
#  Multivariate
# -------------------------------------------------------------------
def test_create_multivariate_normal(rng):
    mean = [0, 10]
    cov = [[1, 0.5], [0.5, 2]]
    dist = create_prob_distribution("multivariate_normal", population_matrix=np.zeros((1,2)), mean=mean, cov=cov)
    sample = dist.sample((100, 2), rng)
    assert sample.shape == (100, 2)
    assert_allclose(sample.mean(axis=0), mean, atol=0.5)


# -------------------------------------------------------------------
#  Heuristics tests – unchanged except gamma (needs fix in source)
# -------------------------------------------------------------------
# ... keep all heuristic tests exactly as you wrote them, but note
# that gamma_heuristic will pass only after you fix the overwrite bug.


# -------------------------------------------------------------------
#  Heuristics: normal
# -------------------------------------------------------------------
def test_normal_heuristic_calculated(rng):
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    kwargs = normal_heuristic(data, loc="calculated", scale="calculated")
    assert_allclose(kwargs["loc"], data.mean(axis=0))
    assert_allclose(kwargs["scale"], data.std(axis=0))

def test_normal_heuristic_explicit_values(rng):
    data = np.random.randn(10, 3)
    kwargs = normal_heuristic(data, loc=5.0, scale=0.5)
    assert kwargs["loc"] == 5.0
    assert kwargs["scale"] == 0.5


# -------------------------------------------------------------------
#  Heuristics: uniform
# -------------------------------------------------------------------
def test_uniform_heuristic_calculated(rng):
    data = np.array([[0, 10], [5, 7], [2, 12], [8, 9]], dtype=float)
    kwargs = uniform_heuristic(data, loc="calculated", scale="calculated")
    assert_allclose(kwargs["loc"], data.min(axis=0))
    assert_allclose(kwargs["scale"], data.max(axis=0) - data.min(axis=0))


# -------------------------------------------------------------------
#  Heuristics: cauchy
# -------------------------------------------------------------------
def test_cauchy_heuristic_calculated(rng):
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    kwargs = cauchy_heuristic(data, loc="calculated", scale="calculated")
    assert_allclose(kwargs["loc"], np.median(data, axis=0))
    # scale = IQR/2 (scipy.stats.iqr returns IQR)
    expected_scale = sp.stats.iqr(data, axis=0) / 2.0
    assert_allclose(kwargs["scale"], expected_scale)


# -------------------------------------------------------------------
#  Heuristics: laplace (similar to cauchy but uses MAD)
# -------------------------------------------------------------------
def test_laplace_heuristic_calculated(rng):
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    kwargs = laplace_heuristic(data, loc="calculated", scale="calculated")
    assert_allclose(kwargs["loc"], np.median(data, axis=0))
    expected_scale = sp.stats.median_abs_deviation(data, axis=0)
    assert_allclose(kwargs["scale"], expected_scale)


# -------------------------------------------------------------------
#  Heuristics: gamma
# -------------------------------------------------------------------
def test_gamma_heuristic_calculated(rng):
    data = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=float)
    kwargs = gamma_heuristic(data, a="calculated", scale="calculated")
    mean = data.mean(axis=0)
    var = data.var(axis=0)
    expected_a = mean**2 / var
    expected_scale = var / mean
    assert_allclose(kwargs["a"], expected_a)
    assert_allclose(kwargs["scale"], expected_scale)


# -------------------------------------------------------------------
#  Heuristics: exponential
# -------------------------------------------------------------------
def test_expon_heuristic_calculated(rng):
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    kwargs = expon_heuristic(data, scale="calculated", loc=0)
    expected = data.mean(axis=0)
    assert_allclose(kwargs["scale"], expected)


# -------------------------------------------------------------------
#  Heuristics: poisson
# -------------------------------------------------------------------
def test_poisson_heuristic_calculated(rng):
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    kwargs = poisson_heuristic(data, mu="calculated", loc=0)
    expected = data.mean(axis=0)
    assert_allclose(kwargs["mu"], expected)


# -------------------------------------------------------------------
#  Heuristics: bernoulli
# -------------------------------------------------------------------
def test_bernoulli_heuristic_calculated(rng):
    data = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    kwargs = bernoulli_heuristic(data, p="calculated", loc=0)
    expected = data.mean(axis=0)
    assert_allclose(kwargs["p"], expected)


# -------------------------------------------------------------------
#  Heuristics: binomial
# -------------------------------------------------------------------
def test_binomial_heuristic_calculated(rng):
    data = np.array([[2, 4], [3, 6], [4, 8]], dtype=float)   # n = 10 assumed
    kwargs = binomial_heuristic(data, p="calculated", n=10, loc=0)
    expected_p = data.mean(axis=0) / 10.0
    assert_allclose(kwargs["p"], expected_p)


# -------------------------------------------------------------------
#  Heuristics: von Mises (tikhinov)
# -------------------------------------------------------------------
def test_tikhinov_heuristic_calculated(rng):
    # Create angles around 0 and pi/2
    data = np.array([[0.0, np.pi/2], [0.2, np.pi/2+0.1], [-0.1, np.pi/2-0.1]])
    kwargs = tikhinov_heuristic(data, loc="calculated", kappa="calculated")
    # loc should be circular mean (arctan2 of mean sin/cos)
    mean_cos = np.cos(data).mean(axis=0)
    mean_sin = np.sin(data).mean(axis=0)
    expected_loc = np.arctan2(mean_sin, mean_cos)
    assert_allclose(kwargs["loc"], expected_loc, atol=1e-6)
    # kappa approximation
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    expected_kappa = R / (1 - R)
    assert_allclose(kwargs["kappa"], expected_kappa, atol=1e-6)


# -------------------------------------------------------------------
#  Heuristics: multivariate normal
# -------------------------------------------------------------------
def test_multivariate_normal_heuristic_mean_calculated(rng):
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    kwargs = multivariate_normal_heuristic(data, mean="calculated")
    assert_allclose(kwargs["mean"], data.mean(axis=0))

def test_multivariate_normal_heuristic_cov_raises():
    data = np.random.randn(5, 3)
    with pytest.raises(ValueError, match="covariance estimation"):
        multivariate_normal_heuristic(data, cov="calculated")


# -------------------------------------------------------------------
#  Heuristics: Dirichlet (raises on "calculated")
# -------------------------------------------------------------------
def test_dirichlet_heuristic_calculated_raises():
    data = np.random.rand(5, 3)
    with pytest.raises(ValueError, match="Dirichlet parameter"):
        dirichlet_heuristic(data, alpha="calculated")


# -------------------------------------------------------------------
#  Heuristics: von Mises-Fisher
# -------------------------------------------------------------------
def test_tikhinov_fisher_heuristic_calculated(rng):
    # Data on unit sphere (2D points)
    data = np.array([[1, 0], [0.8, 0.6], [0.6, 0.8]])  # approximately unit vectors
    kwargs = tikhinov_fisher_heuristic(data, loc="calculated", kappa="calculated")
    sample_mean = data.mean(axis=0)
    radius = np.linalg.norm(sample_mean)
    expected_loc = sample_mean / radius
    assert_allclose(kwargs["loc"], expected_loc, atol=1e-6)
    d = data.shape[1]
    expected_kappa = radius * (d - radius**2) / (1 - radius**2)
    assert_allclose(kwargs["kappa"], expected_kappa, atol=1e-6)


# -------------------------------------------------------------------
#  Parameter re‑interpretation: uniform fix
# -------------------------------------------------------------------
def test_uniform_param_fix():
    from metaheuristic_designer.operators.operator_functions.probability_distributions_factory import uniform_param_fix
    kwargs = uniform_param_fix(min=0, max=10, extra=42)
    assert kwargs["loc"] == 0
    assert kwargs["scale"] == 10
    assert kwargs["extra"] == 42
    # min/max should not be in the returned dict
    assert "min" not in kwargs
    assert "max" not in kwargs


# -------------------------------------------------------------------
#  Edge case: unknown distribution
# -------------------------------------------------------------------
def test_create_prob_distribution_unknown():
    with pytest.raises(ValueError):
        create_prob_distribution("nonexistent", None)