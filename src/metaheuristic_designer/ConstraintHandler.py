from __future__ import annotations
from copy import copy
from typing import Any
from abc import ABC, abstractmethod


class ConstraintHandler(ABC):
    """ """

    def repair_solution(self, solution: Any) -> Any:
        """ """

    def penalty(self, solution: Any) -> float:
        """ """


class PenalizeConstraint(ConstraintHandler, ABC):
    def repair_solution(self, solution):
        return copy(solution)


class RepareConstraint(ConstraintHandler, ABC):
    def penalty(self, solution):
        return 0


class NullConstraint(ConstraintHandler):
    def repair_solution(self, solution):
        return copy(solution)

    def penalty(self, solution):
        return 0
