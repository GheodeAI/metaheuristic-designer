from __future__ import annotations
from typing import Iterable
import enum
from enum import Enum
from copy import copy
import numpy as np
from ..operator import Operator
from ..param_scheduler import ParamScheduler
from ..utils import RAND_GEN


class CompositeOperator(Operator):
    """
    Operator class that utilizes a list of operators to modify individuals.

    Parameters
    ----------
    method: str
        Type of operator that will be applied.
    op_list: List[Operator]
        List of operators that will be used.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(
        self,
        op_list: Iterable[Operator],
        name: str = None,
    ):
        """
        Constructor for the OperatorMeta class
        """

        self.op_list = op_list

        if name is None:
            op_names = []
            for op in op_list:
                if not isinstance(op, Operator):
                    op_names.append("lambda_func")
                else:
                    op_names.append(op.name)

            joined_names = ", ".join(op_names)
            name = f"Sequence ({joined_names})"

        super().__init__(name=name)

    def evolve(self, population, initializer=None):
        new_population = copy(population)

        for op in self.op_list:
            new_population = op.evolve(new_population, initializer)

        return new_population

    def step(self, progress: float):
        super().step(progress)

        for op in self.op_list:
            if isinstance(op, Operator):
                op.step(progress)

    def get_state(self) -> dict:
        data = super().get_state()

        data["op_list"] = []

        for op in self.op_list:
            if isinstance(op, Operator):
                data["op_list"].append(op.get_state())
            else:
                data["op_list"].append("lambda_func")

        return data
