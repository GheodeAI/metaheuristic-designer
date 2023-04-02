import random
from copy import copy
from typing import List, Union
from ..ParamScheduler import ParamScheduler
from ..Operator import Operator
from enum import Enum


class MetaOpMethods(Enum):
    BRANCH2 = 1
    BRANCH = 2
    SEQUENCE = 3

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in meta_ops_map:
            raise ValueError(f"Operator on operators \"{str_input}\" not defined")
        
        return meta_ops_map[str_input]

meta_ops_map = {
    "branch2": MetaOpMethods.BRANCH2,
    "branch": MetaOpMethods.BRANCH,
    "sequence": MetaOpMethods.SEQUENCE,
}


class OperatorMeta(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, method: str, op_list: List[Operator], params: Union[ParamScheduler, dict]=None, name=None):
        """
        Constructor for the Operator class
        """

        self.op_list = op_list

        if params is None:

            # Default parameters
            params = {
                "p": 0.5,
                "weights": [1]*len(op_list),
                "mask": 0
            }

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = MetaOpMethods.from_str(method)
    
    
    def evolve(self, indiv, population, objfunc, global_best):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        if self.method == MetaOpMethods.BRANCH2:
            if random.random() > self.params["P"]:
                op = self.op_list[0]
            else:
                op = self.op_list[1]
            result = op(indiv, population, objfunc, global_best)
        
        elif self.method == MetaOpMethods.BRANCH:
            op = random.choices(op_list, weights=self.params["weights"])
            result = op(indiv, population, objfunc, global_best)
        
        elif self.method == MetaOpMethods.SEQUENCE:
            result = indiv
            for op in self.op_list:
                result = op(result, population, objfunc, global_best)
        
        return result