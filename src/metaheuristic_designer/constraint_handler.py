"""
Base class for the Constraint Handler module.

This module implements ways to enforce constraints on the objective function.
"""

from __future__ import annotations
from copy import copy
from typing import Any, Callable, Iterable, Optional
from abc import ABC, abstractmethod
import numpy as np
from .population import Population
from .parametrizable_mixin import ParametrizableMixin
from .utils import ScalarLike, VectorLike, MatrixLike


class ConstraintHandler(ParametrizableMixin, ABC):
    """Abstract base for all constraint handlers.

    A constraint handler can **repair** solutions (make them
    feasible) and/or compute a **penalty** that is subtracted from
    the objective value.  Subclasses must implement at least one of
    these operations.

    Parameters
    ----------
    \\*\\*kwargs
        Additional keyword arguments stored as schedulable
        parameters.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.store_kwargs(**kwargs)

    def repair_population(self, population: Population) -> Population:
        population = copy(population)
        population_matrix = population.genotype_matrix
        repaired_matrix = self.repair_solutions(population_matrix)
        repaired_population = population.update_genotype(repaired_matrix)

        return repaired_population

    @abstractmethod
    def repair_solutions(self, population_matrix: MatrixLike) -> MatrixLike:
        """
        Modifies the incoming solution so that it follows the problem's constraints.

        Parameters
        ----------
        solution: Any
            The input solution.

        Returns
        -------
        fixed_solution: Any
            Modified version of the input solution that fits the problem's constraints
        """

    @abstractmethod
    def penalty(self, population_matrix: Iterable) -> VectorLike:
        """
        Offset to the objective value for the solution corresponding to violations of the problem's constraints.

        Parameters
        ----------
        solution: Any
            The input solution.

        Returns
        -------
        penalty: float
            The amount of penalty to apply to the current solution.
        """

    def get_state(self):
        data = {
            "class_name": self.__class__.__name__,
        }

        return data


class ConstraintHandlerFromLambda(ConstraintHandler):
    """Constraint handler built from plain callables.

    At least one of *repair_solution_fn* or *penalty_fn* must be
    given.

    Parameters
    ----------
    repair_solution_fn : callable, optional
        A function ``(solution) -> repaired_solution``.
    penalty_fn : callable, optional
        A function ``(solution) -> penalty_value``.
    \\*\\*kwargs
        Keyword arguments forwarded to
        :class:`ConstraintHandler`.
    """

    def __init__(self, repair_solution_fn: Optional[Callable] = None, penalty_fn: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)

        if repair_solution_fn is None and penalty_fn is None:
            raise ValueError("You must give the implementation of the repairing procedure or the penalty calculation.")

        self.repair_solution_fn = repair_solution_fn
        self.penalty_fn = penalty_fn

    def repair_solutions(self, solution: Iterable) -> Iterable:
        if self.repair_solution_fn is None:
            return copy(solution)

        return self.repair_solution_fn(solution)

    def penalty(self, solutions: Any) -> ScalarLike:
        if self.penalty_fn is None:
            return np.zeros(len(solutions))

        return self.penalty_fn(solutions)


class NullConstraint(ConstraintHandler):
    """Constraint handler that enforces no restrictions.

    The penalty is always zero, and repairing returns the solution
    unchanged.

    Parameters
    ----------
    encoding : Encoding, optional
        See :class:`ConstraintHandler`.
    \\*\\*kwargs
        See :class:`ConstraintHandler`.
    """

    def repair_solutions(self, solution: MatrixLike) -> MatrixLike:
        return copy(solution)

    def penalty(self, solutions: Iterable) -> VectorLike:
        return np.zeros(len(solutions))


class PenalizeConstraint(ConstraintHandler, ABC):
    """Abstract handler that only computes penalties.

    Repairing does nothing (returns a copy).
    Subclasses must override :meth:`penalty`.

    Parameters
    ----------
    encoding : Encoding, optional
        See :class:`ConstraintHandler`.
    \\*\\*kwargs
        See :class:`ConstraintHandler`.
    """

    def repair_solutions(self, solution: Iterable) -> Iterable:
        return copy(solution)


class RepairConstraint(ConstraintHandler, ABC):
    """Abstract handler that only repairs solutions.

    The penalty is always zero.
    Subclasses must override :meth:`repair_solution`.

    Parameters
    ----------
    encoding : Encoding, optional
        See :class:`ConstraintHandler`.
    \\*\\*kwargs
        See :class:`ConstraintHandler`.
    """

    def penalty(self, solutions: Iterable) -> VectorLike:
        return np.zeros(len(solutions))
