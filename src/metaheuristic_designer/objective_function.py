"""
Base class for the Objective Function module.

This module implements the objective function that will measure the quality of the solutions.
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
from .constraint_handler import ConstraintHandler, NullConstraint
from .constraint_handlers import ClipBoundConstraint, CompositeConstraint
from .parametrizable_mixin import ParametrizableMixin
from .utils import check_random_state, RNGLike, VectorLike, ScalarLike

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
        constraint_handler: Optional[ConstraintHandler] = None,
        mode: str = "max",
        name: str = "some function",
        vectorized: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Constructor for the ObjectiveFunc class
        """
        super().__init__()

        if constraint_handler is None:
            constraint_handler = NullConstraint()
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

        logger.info("Calculating fitness of the population...")
        fitness = population.fitness
        objective = population.objective
        solutions = population.decode()
        if self.vectorized:
            if not self.recalculate:
                solutions = solutions[population.fitness_calculated == 0, :]

            objective_values = self.objective(solutions)
            fitness_values = self.factor * (objective_values - self.constraint_handler.penalty(solutions))

            if self.recalculate:
                # Using a slice we overwrite fitness values
                fitness[:] = fitness_values 
                objective[:] = objective_values
            else:
                fitness[population.fitness_calculated == 0] = fitness_values
                objective[population.fitness_calculated == 0] = objective_values

        else:
            for idx, (solution, already_calculated) in enumerate(zip(solutions, population.fitness_calculated)):
                if self.recalculate or not already_calculated:
                    objective_value = self.objective(solution)
                    fitness_value = self.factor * (objective_value - self.constraint_handler.penalty(solution))

                    fitness[idx] = fitness_value
                    objective[idx] = objective_value

        if self.recalculate:
            self.counter += int(population.pop_size)
        else:
            self.counter += int(population.pop_size - population.fitness_calculated.sum())

        population.fitness_calculated = np.ones_like(population.fitness_calculated)

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

    def repair_solution(self, solution: Any) -> Any:
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

    def restart(self):
        self.counter = 0
    
    def get_state(self):
        data = {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "constraint": self.constraint_handler.get_state(),
            **self.get_params()
        }

        return data


class VectorObjectiveFunc(ObjectiveFunc):
    """
    Objective function that accepts vectors as an input.

    Parameters
    ----------
    vecsize: int
        The dimension of the vectors accepted by the objective function.
    mode: str, optional
        Whether to maximize or minimize the function (using the string 'max' or 'min').
    low_lim: float, optional
        Lower limit restriction for the vectors.
    up_lim: float, optional
        Upper limit restriction for the vectors.
    name: str, optional
        The name that will be displayed to represent this function.
    """

    def __init__(
        self,
        vecsize: int,
        low_lim: float,
        up_lim: float,
        constraint_handler: Optional[ConstraintHandler] = None,
        mode: str = "max",
        name: str = "Some function",
        vectorized: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Constructor for the ObjectiveVectorFunc class
        """

        self.vecsize = vecsize
        self.low_lim = low_lim
        self.up_lim = up_lim

        bound_constraint_handler = ClipBoundConstraint(vecsize, low_lim, up_lim)
        if constraint_handler is None:
            constraint_handler = bound_constraint_handler
        else:
            constraint_handler = CompositeConstraint([constraint_handler, bound_constraint_handler])

        super().__init__(constraint_handler=constraint_handler, mode=mode, name=name, vectorized=vectorized, recalculate=recalculate, **kwargs)


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
    vecsize: int
        The dimension of the vectors accepted by the objective function.
    mode: str, optional
        Whether to maximize or minimize the function (using the string 'max' or 'min').
    low_lim: float, optional
        Lower limit restriction for the vectors.
    up_lim: float, optional
        Upper limit restriction for the vectors.
    name: str, optional
        The name that will be displayed to represent this function.
    """

    def __init__(
        self,
        obj_func: Callable,
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

        super().__init__(constraint_handler=constraint_handler, mode=mode, name=name, vectorized=vectorized, recalculate=recalculate, **kwargs)

    def objective(self, solution: Any) -> VectorLike | ScalarLike:
        return self.obj_func(solution, **self.current_kwargs)
