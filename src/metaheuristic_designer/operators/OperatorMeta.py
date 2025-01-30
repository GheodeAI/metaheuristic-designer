from __future__ import annotations
from typing import Iterable
import enum
from enum import Enum
from copy import copy
import numpy as np
from ..Operator import Operator
from ..ParamScheduler import ParamScheduler
from ..utils import RAND_GEN


class MetaOpMethods(Enum):
    BRANCH = enum.auto()
    SEQUENCE = enum.auto()
    SPLIT = enum.auto()
    PICK = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in meta_ops_map:
            raise ValueError(f'Operator on operators "{str_input}" not defined')

        return meta_ops_map[str_input]


meta_ops_map = {
    "branch": MetaOpMethods.BRANCH,
    "sequence": MetaOpMethods.SEQUENCE,
    "split": MetaOpMethods.SPLIT,
    "pick": MetaOpMethods.PICK,
}


class OperatorMeta(Operator):
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
        method: str,
        op_list: Iterable[Operator],
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor for the OperatorMeta class
        """

        self.op_list = op_list

        if params is None:
            # Default parameters
            params = {
                "p": 0.5,
                "weights": [1] * len(op_list),
                "mask": 0,
                "init_idx": -1,
            }

        if name is None:
            op_names = []
            for op in op_list:
                if not isinstance(op, Operator):
                    op_names.append("lambda_func")
                else:
                    op_names.append(op.name)

            joined_names = ", ".join(op_names)
            name = f"{method}({joined_names})"

        super().__init__(params, name)

        self.method = MetaOpMethods.from_str(method)

        # Record of the index of the last operator used
        self.chosen_idx = params.get("init_idx", -1)
        self.mask = params.get("mask", 0)

        # If we have a branch with 2 operators and "p" is given as an input
        if self.method == MetaOpMethods.BRANCH and "weights" not in params and "p" in params and len(op_list) == 2:
            params["weights"] = [params["p"], 1 - params["p"]]

    def evolve(self, population, initializer=None):
        new_population = copy(population)

        match self.method:
            case MetaOpMethods.BRANCH:
                self.chosen_idx = RAND_GEN.choice(np.arange(len(self.op_list)), size=population.pop_size, p=self.params["weights"])
                for idx, op in enumerate(self.op_list):
                    split_mask = self.chosen_idx == idx

                    if np.any(split_mask):
                        split_population = population.take_selection(split_mask)
                        split_population = op.evolve(split_population, initializer)
                        new_population = new_population.apply_selection(split_population, split_mask)

            case MetaOpMethods.PICK:
                if isinstance(self.chosen_idx, np.ndarray) and self.chosen_idx.ndim > 0:
                    chosen_idx = self.chosen_idx
                else:
                    chosen_idx = np.asarray([self.chosen_idx] * len(population))

                # the chosen index is assumed to be changed by the user
                for idx, op in enumerate(self.op_list):
                    split_mask = chosen_idx == idx

                    if np.any(split_mask):
                        split_population = new_population.take_selection(split_mask)
                        split_population = op.evolve(split_population, initializer)
                        new_population = new_population.apply_selection(split_population, split_mask)

            case MetaOpMethods.SEQUENCE:
                for op in self.op_list:
                    new_population = op.evolve(new_population, initializer)

            case MetaOpMethods.SPLIT:
                for idx_op, op in enumerate(self.op_list):
                    split_mask = self.mask == idx_op

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
