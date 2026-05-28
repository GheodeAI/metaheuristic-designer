"""
Swarm intelligence operator implementations.
"""

from typing import Optional
import numpy as np
from ...population import Population
from ...initializer import Initializer
from ...encodings import ParameterExtendingEncoding
from ...utils import MatrixLike, check_random_state, RNGLike


def pso_operator(
    population_matrix: MatrixLike,
    population_speed: MatrixLike,
    historical_best: MatrixLike,
    global_best: MatrixLike,
    random_state: Optional[RNGLike] = None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> tuple[MatrixLike, MatrixLike]:
    """
    Perform a single step of the standard Particle Swarm optimization (PSO).

    Velocity is updated as:

    .. math::

        v_{i} = w v_{i} + c_1 r_1 (p_{i} - x_{i}) + c_2 r_2 (g - x_{i})

    where :math:`p_{i}` is the historical best of particle *i*,
    :math:`g` is the global best, and :math:`r_1`, :math:`r_2` are
    uniform random numbers in [0, 1].

    The new position is :math:`x_{i} + v_{i}`.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current positions, shape ``(N, D)``.
    population_speed : MatrixLike
        Current velocities, shape ``(N, D)``.
    historical_best : MatrixLike
        Personal best positions, shape ``(N, D)``.
    global_best : MatrixLike
        Global best position, shape ``(D,)`` (broadcast to ``(N, D)``).
    random_state : RNGLike, optional
        Random number generator.
    w : float, optional
        Inertia weight (default 0.7).
    c1 : float, optional
        Cognitive acceleration coefficient (default 1.5).
    c2 : float, optional
        Social acceleration coefficient (default 1.5).

    Returns
    -------
    tuple[MatrixLike, MatrixLike]
        The new positions and the new velocities, both shape ``(N, D)``.
    """

    random_state = check_random_state(random_state)

    c1 = c1 * random_state.random(population_matrix.shape)
    c2 = c2 * random_state.random(population_matrix.shape)

    speed = w * population_speed + c1 * (historical_best - population_matrix) + c2 * (global_best - population_matrix)

    return population_matrix + speed, speed


def pso_operator_wrapper(
    population: Population,
    initializer: Initializer,
    random_state: Optional[RNGLike] = None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> Population:
    """
    Wrapper that integrates the PSO operator with the library's Population API.

    Extracts the solution and velocity parts from the population (which must
    use a :class:`ParameterExtendingEncoding` with a ``"speed"`` parameter),
    applies the standard PSO update, and encodes the result back into the
    population's genotype matrix.

    Parameters
    ----------
    population : Population
        Current population. Its encoding must be a
        :class:`~metaheuristic_designer.encodings.ParameterExtendingEncoding`
        that includes a ``"speed"`` parameter.
    _initializer : Initializer
        Initializer (unused; kept for interface compatibility).
    random_state : RNGLike, optional
        Random number generator.
    w : float, optional
        Inertia weight (default 0.7).
    c1 : float, optional
        Cognitive coefficient (default 1.5).
    c2 : float, optional
        Social coefficient (default 1.5).

    Returns
    -------
    Population
        The updated population with new positions and velocities.
    """

    population_encoding = population.encoding
    if (not isinstance(population_encoding, ParameterExtendingEncoding)) or ("speed" not in population_encoding.extended_parameters):
        raise ValueError('Encoding of the population must be a ParameterExtendingEncoding with a "speed" parameter')

    population_genotype = population_encoding.extract_solution(population.genotype_matrix)
    population_params = population_encoding.decode_params(population.genotype_matrix)
    historical_best_solution = population_encoding.extract_solution(population.historical_best_matrix)
    global_best_solution = population_encoding.extract_solution(population.best[None, :])[0]

    population_solutions, population_params["speed"] = pso_operator(
        population_genotype, population_params["speed"], historical_best_solution, global_best_solution, random_state=random_state, w=w, c1=c1, c2=c2
    )

    population_matrix = population.encoding.encode(population_solutions, population_params)
    return population.update_genotype(population_matrix)
