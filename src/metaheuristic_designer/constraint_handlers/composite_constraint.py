from __future__ import annotations
from typing import Iterable
from copy import copy
from ..constraint_handler import ConstraintHandler


class CompositeConstraint(ConstraintHandler):
    """
    Aplies every constraint handler in succession.

    Parameters
    ----------
    constraints: Iterable
        List of constraint handlers.
    """

    def __init__(self, constraints: Iterable):
        self.constraints = constraints

    def repair_solution(self, solution):
        repaired_solution = copy(solution)
        for c in self.constraints:
            repaired_solution = c.repair_solution(repaired_solution)

        return repaired_solution

    def penalty(self, solution):
        return sum(c.penalty(solution) for c in self.constraints)
