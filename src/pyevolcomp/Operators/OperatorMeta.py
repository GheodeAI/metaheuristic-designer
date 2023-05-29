from __future__ import annotations
import random
from ..Operator import Operator
from copy import copy
import numpy as np
from enum import Enum


class MetaOpMethods(Enum):
    BRANCH = 1
    SEQUENCE = 2
    SPLIT = 3
    PICK = 4

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in meta_ops_map:
            raise ValueError(f"Operator on operators \"{str_input}\" not defined")

        return meta_ops_map[str_input]


meta_ops_map = {
    "branch": MetaOpMethods.BRANCH,
    "sequence": MetaOpMethods.SEQUENCE,
    "split": MetaOpMethods.SPLIT,
    "pick": MetaOpMethods.PICK
}


class OperatorMeta(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, method: str, op_list: List[Operator], params: Union[ParamScheduler, dict] = None, name: str = None):
        """
        Constructor for the Operator class
        """

        self.op_list = op_list

        if params is None:

            # Default parameters
            params = {
                "p": 0.5,
                "weights": [1] * len(op_list),
                "mask": 0,
                "init_idx": -1
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
        self.chosen_idx = params["init_idx"] if "init_idx" in params else -1
        self.mask = params["mask"] if "mask" in params else 0

        # If we have a branch with 2 operators and "p" is given as an input
        if self.method == MetaOpMethods.BRANCH and "weights" not in params and "p" in params and len(op_list) == 2:
            params["weights"] = [params["p"], 1 - params["p"]]

    def evolve(self, indiv, population, objfunc, global_best, initializer=None):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        if self.method == MetaOpMethods.BRANCH:
            self.chosen_idx = random.choices(range(len(self.op_list)), k=1, weights=self.params["weights"])[0]
            chosen_op = self.op_list[self.chosen_idx]
            result = chosen_op(indiv, population, objfunc, global_best, initializer)
        
        elif self.method == MetaOpMethods.PICK:
            # the chosen index is assumed to be changed by the user
            chosen_op = self.op_list[self.chosen_idx]
            result = chosen_op(indiv, population, objfunc, global_best, initializer)

        elif self.method == MetaOpMethods.SEQUENCE:
            result = indiv
            for op in self.op_list:
                result = op(result, population, objfunc, global_best, initializer)
        
        elif self.method == MetaOpMethods.SPLIT:
            result = copy(indiv)
            indiv_copy = copy(indiv)
            global_best_copy = copy(global_best)
            population_copy = [copy(i) for i in population]

            for idx_op, op in enumerate(self.op_list):
                if np.any(self.mask == idx_op):
                    indiv_copy.genotype = indiv.genotype[self.mask == idx_op]
                    global_best_copy.genotype = global_best.genotype[self.mask == idx_op]

                    for idx_pop, val in enumerate(population_copy):
                        val.genotype = population[idx_pop].genotype[self.mask == idx_op]

                    aux_indiv = op(indiv_copy, population_copy, objfunc, global_best, initializer)
                    result.genotype[self.mask == idx_op] = aux_indiv.genotype

        return result
    
    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        super().step(progress)
        
        for op in self.op_list:
            if isinstance(op, Operator):
                op.step(progress)
            
    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.
        """
        
        data = super().get_state()

        data["op_list"] = []

        for op in self.op_list:
            if isinstance(op, Operator):
                data["op_list"].append(op.get_state())
            else:
                data["op_list"].append("lambda_func")
        
        return data

