from __future__ import annotations
import random
from ..Operator import Operator
from enum import Enum


class MetaOpMethods(Enum):
    BRANCH = 1
    SEQUENCE = 2
    SPLIT = 3
    PICK_ONE = 4

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
    "pickone": MetaOpMethods.PICK_ONE
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
            joined_names = "+".join([op.name for op in op_list if op.method.lower() != "nothing"])
            name = f"{method}: {joined_names}" 

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
            result = self.op_list[self.chosen_idx].evolve(indiv, population, objfunc, global_best, initializer)
        
        elif self.method == MetaOpMethods.PICK_ONE:
            # the chosen index is assumed to be changed by the user
            result = self.op_list[self.chosen_idx].evolve(indiv, population, objfunc, global_best, initializer)

        elif self.method == MetaOpMethods.SEQUENCE:
            result = indiv
            for op in self.op_list:
                result = op.evolve(result, population, objfunc, global_best, initializer)
        
        elif self.method == MetaOpMethods.SPLIT:
            indiv_copy = indiv.copy()
            global_best_copy = global_best.copy()
            population_copy = [i.copy() for i in population]

            for idx, op in enumerate(self.op_list):
                indiv_copy.genotype = indiv.genotype[self.mask == idx]
                global_best_copy.genotype = global_best.genotype[self.mask == idx]

                for idx, val in enumeate(population_copy):
                    val.genotype = population[idx].genotype[self.mask == idx]

                aux_indiv = op.evolve(indiv_copy, population_copy, objfunc, global_best, initializer)
                indiv.genotype[self.mask == idx] = aux_indiv.genotype

        return result
    
    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        super().step(progress)
        
        for op in self.op_list:
            op.step(progress)

