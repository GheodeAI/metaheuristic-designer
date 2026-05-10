"""
Base class for the Objective Function module.

This module implements the objective function that will measure the quality of the solutions.
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np

from .encodings import ParameterExtendingEncoding
from .constraint_handlers import ConstraintHandler, ClipBoundConstraint, CompositeConstraint, ExtendedConstraintHandler
from .parametrizable_mixin import ParametrizableMixin
from .utils import MatrixLike, VectorLike, ScalarLike

if TYPE_CHECKING:
    from metaheuristic_designer.population import Population

logger = logging.getLogger(__name__)


class ObjectiveFunc(ParametrizableMixin, ABC):
    """
    Abstract Fitness function class.

    For each problem a new class will inherit from this one
    and implement the fitness function, random solution generation,
    mutation function and crossing of solutions.

    Parameters
    ----------
    mode: str, optional
        Whether to maximize or minimize the function (using the string 'max' or 'min').
    name: str, optional
        The name that will be displayed to represent this function.
    vectorized: bool, optional
        Indicates that the function will calculate the fitness of the entire population in one function call.
    recalculate: bool, optional
        Whether to calculate the fitness of the individuals even if they were already calculated before.
    """

    def __init__(
        self,
        dimension: int,
        lower_bound: Optional[ScalarLike | VectorLike] = None,
        upper_bound: Optional[ScalarLike | VectorLike] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        mode: str = "max",
        name: str = "Some function",
        vectorized: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Constructor for the ObjectiveFunc class
        """
        super().__init__()

        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if lower_bound is not None and upper_bound is not None:
            bound_constraint_handler = ClipBoundConstraint(dimension, lower_bound, upper_bound)
            if constraint_handler is None:
                constraint_handler = bound_constraint_handler
            else:
                constraint_handler = CompositeConstraint([constraint_handler, bound_constraint_handler])

        self.constraint_handler = constraint_handler
        self.name = name
        self.counter = 0
        self.factor = 1
        self.vectorized = vectorized
        self.recalculate = recalculate

        if mode not in ["max", "min"]:
            raise ValueError('Optimization objective (mode) must be "min" or "max".')
        self.mode = mode

        if mode == "min":
            self.factor = -1

        self.store_kwargs(**kwargs)

    def __call__(self, population: Population, adjusted: bool = True, parallel: bool = False, threads: int = 8) -> VectorLike:
        """
        Shorthand for executing the objective function on a vector.
        """

        return self.fitness(population, adjusted)

    def fitness(self, population: Population, parallel: bool = False, threads: int = 8) -> VectorLike:
        """
        Returns the value of the objective function given an individual.
        If the fitness is adjusted, the sign will be switched for minimization problems
        and a penalty will be applied if the solution violates any restriction.

        Parameters
        ----------
        indiv: Individual
            The individual for which the fitness will be calculated.
        adjusted: bool, optional
            Whether to adjust the fitness value or not.
        parallel: bool, optional
            Whether to evaluate the individuals in the population in parallel.
        threads: int, optional
            Number of processes to use at once if calculating the fitness in parallel.

        Returns
        -------
        fitness: ndarray
            Fitness value of the individual.
        """

        if parallel:
            logger.warning("Parallel fitness computing not available at the moment. Ignoring parallel option.")

        logger.debug("Calculating fitness of the population...")
        fitness = population.fitness
        objective = population.objective
        solutions = population.decode()
        genotypes = population.genotype_matrix

        if not self.recalculate and np.all(population.fitness_calculated == 1):
            logger.debug("Fitness was not calculated. Every individual is duplicated.")
            return population.fitness

        if self.recalculate:
            fitness_mask = np.ones(population.population_size, dtype=bool)
        else:
            fitness_mask = population.fitness_calculated == 0

        # Penalty is always vectorized. We use the genotype instead of the decoded solutions
        penalty_vector = self.constraint_handler.penalty(genotypes[fitness_mask])

        if self.vectorized:
            if isinstance(solutions, np.ndarray):
                solutions = solutions[fitness_mask]
            else:
                solutions = [solutions[i] for i, include_value in enumerate(fitness_mask) if include_value]

            objective_values = self.objective(solutions)
            fitness_values = self.factor * (objective_values - penalty_vector)
            fitness[fitness_mask] = fitness_values
            objective[fitness_mask] = objective_values
        else:
            # Expand the penalty to have the size `pop_size`
            penalty_vector_aux = np.zeros(population.population_size)
            penalty_vector_aux[fitness_mask] = penalty_vector
            penalty_vector = penalty_vector_aux

            for idx, (solution, do_calculation) in enumerate(zip(solutions, fitness_mask)):
                if self.recalculate or do_calculation:
                    objective_value = self.objective(solution)
                    fitness_value = self.factor * (objective_value - penalty_vector[idx])

                    fitness[idx] = fitness_value
                    objective[idx] = objective_value

        self.counter += np.count_nonzero(fitness_mask)
        population.fitness_calculated = np.ones_like(fitness_mask)

        # Write the fitness and objective values in-place
        population.fitness = fitness
        population.objective = objective

        logger.debug("Done calculating the fitness.")
        return fitness

    @abstractmethod
    def objective(self, solution: Any) -> VectorLike | ScalarLike:
        """
        Implementation of the objective function.

        Parameters
        ----------
        solution: Any
            The solution for which the fitness will be calculated.

        Returns
        -------
        objective_value: float | ndarray
            Value of the objective function given a solution.
        """

    def repair_solution(self, solution: MatrixLike) -> MatrixLike:
        """
        Transforms an invalid vector into one that satisfies the restrictions of the problem.

        Parameters
        ----------
        solution: Any
            A solution that could be violating the restrictions of the problem.

        Returns
        -------
        repaired_solution: Any
            A modified version of the solution passed that satisfies the restrictions of the problem.
        """

        return self.constraint_handler.repair_solution(solution)

    def add_parameter_constraints(self, parameter_extending_encoding: ParameterExtendingEncoding, param_handlers: dict[str, ConstraintHandler]):
        if isinstance(self.constraint_handler, ExtendedConstraintHandler):
            assert self.constraint_handler.param_handler_dict.keys() == param_handlers.keys()

        base_constraint_handler = self.constraint_handler

        self.constraint_handler = ExtendedConstraintHandler(
            solution_handler=base_constraint_handler, param_handler_dict=param_handlers, encoding=parameter_extending_encoding
        )

    def restart(self):
        self.counter = 0

    def get_state(self):
        data = {"class_name": self.__class__.__name__, "name": self.name, "constraint": self.constraint_handler.get_state(), **self.get_params()}

        return data


class NullObjectiveFunc(ObjectiveFunc):
    def __init__(self, **kwargs):
        super().__init__(name="Null objective", **kwargs)

    def objective(self, _) -> VectorLike | ScalarLike:
        return 0


class ObjectiveFromLambda(ObjectiveFunc):
    """
    Objective function indicated by a function call.

    Parameters
    ----------
    obj_func: Callable
        Objective function as a callable object.
    dimension: int
        The dimension of the vectors accepted by the objective function.
    mode: str, optional
        Whether to maximize or minimize the function (using the string 'max' or 'min').
    lower_bound: float, optional
        Lower limit restriction for the vectors.
    upper_bound: float, optional
        Upper limit restriction for the vectors.
    name: str, optional
        The name that will be displayed to represent this function.
    """

    def __init__(
        self,
        obj_func: Callable,
        dimension: int,
        lower_bound: Optional[ScalarLike | VectorLike] = None,
        upper_bound: Optional[ScalarLike | VectorLike] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        mode: str = "max",
        name: Optional[str] = None,
        vectorized: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Constructor for the ObjectiveFromLambda class
        """

        if name is None:
            name = obj_func.__name__

        self.obj_func = obj_func

        super().__init__(
            dimension=dimension,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            constraint_handler=constraint_handler,
            mode=mode,
            name=name,
            vectorized=vectorized,
            recalculate=recalculate,
            **kwargs,
        )

    def objective(self, solution: Any) -> VectorLike | ScalarLike:
        return self.obj_func(solution, **self.current_kwargs)
