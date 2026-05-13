import numpy as np
from hypothesis import given, strategies as st
from .simple_hillclimber import simple_hillclimber

@given(
    dimension = st.integers(1, 20),
    sigma = st.floats(0.1, 10.0),
    max_evals = st.integers(2, 100),
    seed = st.integers(0, 2**32 - 1),
)
def test_hillclimber_monotonic_trace(dimension, sigma, max_evals, seed):
    # Fixed bounds for test
    lower = np.full(dimension, -10.0)
    upper = np.full(dimension, 10.0)

    # Use a simple deterministic objective that we can trust
    def objective(x):
        # For maximisation, return negative squared distance from a target
        target = np.zeros(dimension)
        return -np.sum((x - target) ** 2)

    result, trace = simple_hillclimber(
        objective, dimension, lower, upper, max_evals, sigma, seed
    )

    # The property: trace should be non-decreasing (since we maximise)
    for i in range(len(trace) - 1):
        assert trace[i + 1] >= trace[i], f"Violation at step {i}: {trace[i]} > {trace[i+1]}"