import random
from copy import copy
from typing import List, Union
from ..ParamScheduler import ParamScheduler
from ..Operator import Operator
from .Individual import Indiv


class OperatorSplit(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, op_list: List[Operator], mask: np.ndarray):
        """
        Constructor for the Operator class
        """

        self.op_list = op_list
        self.mask = mask

        name = "+".join([op.name for i in op_list])

        super().__init__(name, {})
    
    
    def evolve(self, indiv, population, objfunc, global_best=None):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """
        
        for idx, op in enumerate(op_list):
            aux_indiv = op.evolve(indiv, population, objfunc, global_best)
            indiv.genotype[self.mask == idx] = aux_indiv.genotype[self.mask == idx]

        return result