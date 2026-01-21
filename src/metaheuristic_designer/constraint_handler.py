from __future__ import annotations
from copy import copy
from typing import Any
from abc import ABC, abstractmethod


class ConstraintHandler(ABC):
    """
    Class responsible for enforcing restrictions of the optimization problem.
    """

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
    def penalty(self, solution: Any) -> float:
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

    def __init__(self, repair_solution_fn: callable = None, penalty_fn: callable = None):
        if repair_solution_fn is None and penalty_fn is None:
            raise ValueError("You must give the implementation of the repairing procedure or the penalty calculation.")

        self.repair_solution_fn = repair_solution_fn
        self.penalty_fn = penalty_fn


    def repair_solution(self, solution: Any) -> Any:
        if self.repair_solution_fn is None:
            return copy(solution)
        
        return self.repair_solution_fn(solution)

    def penalty(self, solution: Any) -> float:
        if self.penalty_fn is None:
            return 0
        
        return self.penalty_fn(solution)


class NullConstraint(ConstraintHandler):
    """
    Constraint handler that enforces no constraints. The penalty is 0 and reparing the solution does nothing.
    """

    def repair_solution(self, solution):
        return copy(solution)

    def penalty(self, solution):
        return 0


class PenalizeConstraint(ConstraintHandler, ABC):
    """
    Abstract constraint handler for applying a penalty to solutions that violate the constraints.

    The `penalty` function must be implemented.
    """

    def repair_solution(self, solution):
        return copy(solution)


class RepareConstraint(ConstraintHandler, ABC):
    """
    Abstract constraint handler for reparing solutions that violate the constraints.

    The `repair_solution` function must be implemented.
    """

    def penalty(self, solution):
        return 0


