from __future__ import annotations
from ..Operator import Operator


class OperatorSplit(Operator):
    """
    Operator class that implements methods to apply different operators to different components of a vector
    """

    def __init__(self, op_list: List[Operator], mask: np.ndarray, name=None):
        """
        Constructor for the Operator class
        """

        self.op_list = op_list
        self.mask = mask

        name = "+".join([op.name for op in op_list if op.method.lower() != "nothing"])

        super().__init__({}, name)

    def evolve(self, indiv, population, objfunc, global_best, initializer):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        for idx, op in enumerate(op_list):
            aux_indiv = op.evolve(indiv, population, objfunc, global_best, initializer)
            indiv.genotype[self.mask == idx] = aux_indiv.genotype[self.mask == idx]

        return result
    
    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        super().step(progress)
        
        for op in self.op_list:
            op.step(progress)

