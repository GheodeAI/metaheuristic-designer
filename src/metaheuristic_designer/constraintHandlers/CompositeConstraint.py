from __future__ import annotations
from copy import copy
from ..ConstraintHandler import ConstraintHandler

class CompositeConstraint(ConstraintHandler):
    def __init__(self, constraints: Iterable):
        self.constraints = constraints

    def repair_solution(self, solution):
        repaired_solution = copy(solution)
        for c in self.constraints:
            repaired_solution = c.repair_solution(repaired_solution)

        return repaired_solution
    
    def penalty(self, solution):
        return sum(c.penalty(solution) for c in self.constraints)
