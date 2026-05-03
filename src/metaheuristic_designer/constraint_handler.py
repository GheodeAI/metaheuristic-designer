"""
Base class for the Constraint Handler module.

This module implements ways to enforce constraints on the objective function.
"""

from __future__ import annotations
from copy import copy
from typing import Any, Callable, Optional
from abc import ABC, abstractmethod
from .parametrizable_mixin import ParametrizableMixin
from .utils import ScalarLike


class ConstraintHandler(ParametrizableMixin, ABC):
    """
    Class responsible for enforcing restrictions of the optimization problem.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.store_kwargs(**kwargs)

    @abstractmethod
    def repair_solution(self, solution: Any) -> Any:
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
    def penalty(self, solution: Any) -> ScalarLike:
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
    """
    Constraint handler class constructed with function calls.

    At least one of `repair_solution_fn` and `penalty_fn` must be specified. Both is acceptable too but
    not recommended, if a solution is repaired the penalty should always be 0.

    Parameters
    ----------
    repair_solution: callable, optional
        Function to repair an input solution.

    penalty_fn: callable, optional
        Function to calculate the penalty of the current solution.
    """

    def __init__(self, repair_solution_fn: Optional[Callable] = None, penalty_fn: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)

        if repair_solution_fn is None and penalty_fn is None:
            raise ValueError("You must give the implementation of the repairing procedure or the penalty calculation.")

        self.repair_solution_fn = repair_solution_fn
        self.penalty_fn = penalty_fn

    def repair_solution(self, solution: Any) -> Any:
        if self.repair_solution_fn is None:
            return copy(solution)

        return self.repair_solution_fn(solution)

    def penalty(self, solution: Any) -> ScalarLike:
        if self.penalty_fn is None:
            return 0

        return self.penalty_fn(solution)


class NullConstraint(ConstraintHandler):
    """
    Constraint handler that enforces no constraints. The penalty is 0 and repairing the solution does nothing.
    """

    def repair_solution(self, solution: Any) -> Any:
        return copy(solution)

    def penalty(self, _solution: Any) -> ScalarLike:
        return 0


class PenalizeConstraint(ConstraintHandler, ABC):
    """
    Abstract constraint handler for applying a penalty to solutions that violate the constraints.

    The `penalty` function must be implemented.
    """

    def repair_solution(self, solution: Any) -> Any:
        return copy(solution)


class RepairConstraint(ConstraintHandler, ABC):
    """
    Abstract constraint handler for repairing solutions that violate the constraints.

    The `repair_solution` function must be implemented.
    """

    def penalty(self, _solution: Any) -> ScalarLike:
        return 0
