import numpy as np
from numpy.testing import assert_almost_equal

from conftest import MockGaussianModel

from metaheuristic_designer.operators.BO_operator import _acquisition_function


def test_acquisition_function_expected_improvement():
    # Mock model: mean=0.5, std=0.1
    model = MockGaussianModel(mean=0.5, std=0.1)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # dummy training data, unused by mock
    x_in = np.array([2.0, 3.0])
    max_y = 0.6

    ei = _acquisition_function(model, X, x_in, max_y)

    # Manually compute expected EI
    mean_y = 0.5
    std_y = 0.1
    z = (mean_y - max_y) / std_y  # (0.5 - 0.6)/0.1 = -1.0
    from scipy.stats import norm
    expected = (mean_y - max_y) * norm.cdf(z) + std_y * norm.pdf(z)
    assert_almost_equal(ei, expected)


def test_acquisition_function_zero_std_clamped():
    # std is clamped to 1e-10
    model = MockGaussianModel(mean=0.0, std=0.0)
    X = np.array([[0.0]])
    x_in = np.array([0.0])
    max_y = 0.0
    ei = _acquisition_function(model, X, x_in, max_y)
    # z = (0 - 0)/1e-10 = 0, EI = (0)*0.5 + 1e-10*0.3989 ≈ 3.989e-11
    from scipy.stats import norm
    expected = 1e-10 * norm.pdf(0.0)
    assert_almost_equal(ei, expected, decimal=10)