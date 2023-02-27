from ..Metaheuristic import *
from .ESPopulation import *


class ES(Metaheuristic):
    """
    Evolution strategy optimization algorithm
    """
    
    def __init__(self, objfunc, mutation_op, cross_op, parent_sel_op, replace_op, params):
        """
        Constructor of the Evolution strategy class
        """

        super().__init__("ES", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.mutation_op = mutation_op
        self.cross_op = cross_op
        self.parent_sel_op = parent_sel_op
        self.replace_op = replace_op
        self.population = ESPopulation(objfunc, mutation_op, cross_op, parent_sel_op, replace_op, params)


    def restart(self):
        """
        Resets the data of the algorithm
        """

        super().restart()

        self.population = ESPopulation(self.objfunc, self.mutation_op, self.cross_op,self.parent_sel_op, self.replace_op, self.params)

    
    def step(self, progress):
        """
        One step of the algorithm
        """

        self.population.cross()
        
        self.population.mutate()

        self.population.selection()

        self.population.step(progress)
        
        super().step(progress)