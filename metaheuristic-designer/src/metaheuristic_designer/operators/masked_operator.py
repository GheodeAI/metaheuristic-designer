from __future__ import annotations
from typing import Iterable
from copy import copy
import numpy as np
from ..operator import Operator
from ..utils import MaskLike


class MaskedOperator(Operator):
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

    def __init__(self, op_list: Iterable[Operator], mask: MaskLike, name: str = None, **kwargs):
        """
        Constructor for the OperatorMeta class
        """
        if name is None:
            op_names = []
            for op in op_list:
                if not isinstance(op, Operator):
                    op_names.append("lambda_func")
                else:
                    op_names.append(op.name)

            joined_names = ", ".join(op_names)
            name = f"Split ({joined_names})"

        self.op_list = op_list
        super().__init__(name, mask=mask, **kwargs)

    def gather_params(self):
        all_params = self.get_params()
        for op in self.op_list:
            all_params.update(op.gather_params())

        return all_params

    def evolve(self, population, initializer=None):
        new_population = copy(population)

        # In masked_operator.py, inside the loop over op_list:
        for idx_op, op in enumerate(self.op_list):
            split_mask = self.params.mask == idx_op
            if np.any(split_mask):
                split_population = new_population.take_slice(split_mask)
                split_population = op.evolve(split_population, initializer)
                new_population = new_population.apply_slice(split_population, split_mask)

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
