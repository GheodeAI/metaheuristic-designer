from __future__ import annotations
from typing import Iterable
import enum
from enum import Enum
from copy import copy
import numpy as np
from ..operator import Operator


class BranchOpMethods(Enum):
    RANDOM = enum.auto()
    PICK = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in branch_ops_map:
            raise ValueError(f'Operator on operators "{str_input}" not defined')

        return branch_ops_map[str_input]


branch_ops_map = {"random": BranchOpMethods.RANDOM, "rand": BranchOpMethods.RANDOM, "pick": BranchOpMethods.PICK, "choose": BranchOpMethods.PICK}


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

    def __init__(self, op_list: Iterable[Operator], method: str = None, name: str = None, encoding=None, random_state=None, idx=-1, p=0.5, **kwargs):
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

        super().__init__(name=name, encoding=encoding, random_state=random_state, p=p, **kwargs)

        if method is None:
            self.method = BranchOpMethods.RANDOM
        else:
            self.method = BranchOpMethods.from_str(method)

        self.chosen_idx = idx
        self.weights = np.array([self.params.p, 1 - self.params.p])
    
    def gather_params(self):
        all_params = self.get_params()
        for op in self.op_list:
            all_params.update(op.gather_params())

        return all_params

    def evolve(self, population, initializer=None):
        new_population = copy(population)

        if self.method == BranchOpMethods.RANDOM:
            self.chosen_idx = self.random_state.choice(range(len(self.op_list)), size=(population.pop_size,), replace=True, p=self.weights)

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

    def choose_index(self, idx: int):
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

        self.weights = np.array([self.params.p, 1 - self.params.p])

    def get_state(self) -> dict:
        data = super().get_state()

        data["op_list"] = []

        for op in self.op_list:
            if isinstance(op, Operator):
                data["op_list"].append(op.get_state())
            else:
                data["op_list"].append("lambda_func")

        return data
