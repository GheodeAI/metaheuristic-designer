from __future__ import annotations
from typing import Iterable
from copy import copy
from ..constraint_handler import ConstraintHandler
from ..utils import MatrixLike, ScalarLike


class CompositeConstraint(ConstraintHandler):
    """
    Aplies every constraint handler in succession.

    Parameters
    ----------
    constraints: Iterable
        List of constraint handlers.
    """

    def __init__(self, constraints: Iterable, **kwargs):
        self.constraints = constraints
        super().__init__(**kwargs)
    
    def gather_params(self):
        all_params = self.get_params()
        for const in self.constraints:
            all_params.update(const.gather_params())
        
        return all_params

    def repair_solution(self, solution: MatrixLike) -> MatrixLike:
        repaired_solution = copy(solution)
        for c in self.constraints:
            repaired_solution = c.repair_solution(repaired_solution)

        return repaired_solution

    def penalty(self, solution: MatrixLike) -> ScalarLike:
        return sum(c.penalty(solution) for c in self.constraints)
