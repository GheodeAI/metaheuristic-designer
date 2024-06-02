from __future__ import annotations
import random
from ..Operator import Operator
from copy import copy, deepcopy
import numpy as np
import enum
from enum import Enum


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
        op_list: List[Operator],
        params: Union[ParamScheduler, dict] = None,
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

    def evolve(self, population, objfunc, global_best, initializer):
        if self.method == MetaOpMethods.BRANCH:
            self.chosen_idx = random.choices(range(len(self.op_list)), k=1, weights=self.params["weights"])[0]
            chosen_op = self.op_list[self.chosen_idx]
            result = chosen_op(population, objfunc, global_best, initializer)

        elif self.method == MetaOpMethods.PICK:
            # the chosen index is assumed to be changed by the user
            chosen_op = self.op_list[self.chosen_idx]
            result = chosen_op(population, objfunc, global_best, initializer)

        elif self.method == MetaOpMethods.SEQUENCE:
            result = population
            for op in self.op_list:
                result = op(result, population, objfunc, global_best, initializer)

        elif self.method == MetaOpMethods.SPLIT:
            population_copy = deepcopy(population)

            for idx_op, op in enumerate(self.op_list):
                idx_mask = self.mask == idx_op
                if np.any(idx_mask):
                    filtered_population = [self._filter_indiv(indiv, idx_mask) for indiv in population]
                    filtered_global_best = self._filter_indiv(global_best, idx_mask)

                    new_population = op(filtered_population, objfunc, filtered_global_best, initializer)

                    for indiv, new_indiv in zip(population_copy, new_population):
                        indiv.genotype[idx_mask] = new_indiv.genotype

        return result
    
    @static
    def _filter_indiv(indiv, mask):
        indiv_copy = copy(indiv)
        indiv_copy.genotype = indiv.genotype[self.mask == idx_op]
        return indiv_copy

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
