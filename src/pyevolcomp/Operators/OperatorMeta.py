import random
from copy import copy
from typing import List, Union
from ..ParamScheduler import ParamScheduler
from ..Operator import Operator


_meta_ops = [
    "branch2",
    "branch",
    "sequence"
]


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

        super().__init__(method, params, name)

        if self.method not in _meta_ops:
            raise ValueError(f"Operator \"{self.method}\" not defined")
    
    
    def evolve(self, indiv, population, objfunc, global_best):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        if self.method == "branch2":
            if random.random() > self.params["P"]:
                op = self.op_list[0]
            else:
                op = self.op_list[1]
            result = op(indiv, population, objfunc, global_best)
        
        elif self.method == "branch":
            op = random.choices(op_list, weights=self.params["weights"])
            result = op(indiv, population, objfunc, global_best)
        
        elif self.method == "sequence":
            result = indiv
            for op in self.op_list:
                result = op(result, population, objfunc, global_best)
        
        return result