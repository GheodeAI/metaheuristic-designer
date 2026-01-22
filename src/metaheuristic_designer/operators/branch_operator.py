from __future__ import annotations
from typing import Iterable
import enum
from enum import Enum
from copy import copy
import numpy as np
from ..operator import Operator
from ..param_scheduler import ParamScheduler
from ..utils import RAND_GEN

class BranchOpMethods(Enum):
    RANDOM = enum.auto()
    PICK = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in branch_ops_map:
            raise ValueError(f'Operator on operators "{str_input}" not defined')

        return branch_ops_map[str_input]

branch_ops_map = {
    "random": BranchOpMethods.RANDOM,
    "rand": BranchOpMethods.RANDOM,
    "pick": BranchOpMethods.PICK,
    "choose": BranchOpMethods.PICK,
}

class BranchOperator(Operator):
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
        method: str = None, 
        params: ParamScheduler | dict = None,
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
            name = f"{method}({joined_names})"

        if params is None or params == "default":
            # Default parameters
            params = {
                "p": 0.5,
                "weights": [1] * len(op_list),
                "init_idx": -1,
            }
        
        if method is None:
            self.method = BranchOpMethods.RANDOM
        else:
            self.method = BranchOpMethods.from_str(method)


        super().__init__(params, name)

        self.chosen_idx = params.get("in_description_it_idx", -1)

        # If we have a branch with 2 operators and "p" is given as an input
        if "weights" not in params and "p" in params and len(op_list) == 2:
            params["weights"] = [params["p"], 1 - params["p"]]

    def evolve(self, population, initializer=None):
        new_population = copy(population)

        if self.method == BranchOpMethods.RANDOM:
            self.chosen_idx = RAND_GEN.choice(
                np.arange(len(self.op_list)),
                size=population.pop_size,
                p=self.params["weights"],
            )

        if isinstance(self.chosen_idx, np.ndarray) and self.chosen_idx.ndim > 0:
            chosen_idx = self.chosen_idx
        else:
            chosen_idx = np.asarray([self.chosen_idx] * len(population))

        for idx, op in enumerate(self.op_list):
            split_mask = chosen_idx == idx

            if np.any(split_mask):
                split_population = population.take_selection(split_mask)
                split_population = op.evolve(split_population, initializer)
                new_population = new_population.apply_selection(split_population, split_mask)

        return new_population
    
    def choose_index(idx: int):
        """
        Manually chooses the operator to use next

        Parameters
        ----------
        idx : int
            Index of the operator in the list.
        """
        
        self.chosen_idx = idx

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

