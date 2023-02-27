from ..Metaheuristic import *
from .HSPopulation import *


class HS(Metaheuristic):
    """
    Harmony Search optimization algorithm
    """

    
    def __init__(self, objfunc, mutation_op, replace_op, params):
        """
        Constructor of the Harmony Search algorithm
        """

        super().__init__("HS", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.mutation_op = mutation_op
        self.replace_op = replace_op
        self.population = HSPopulation(objfunc, mutation_op, replace_op, params)

    def restart(self):
        """
        Resets the data of the algorithm
        """

        super().restart()

        self.population = HSPopulation(self.objfunc, self.mutation_op, self.replace_op, self.params)

    def step(self, progress, depredate=True, classic=False):
        """
        One step of the algorithm
        """
        
        self.population.evolve()

        self.population.selection()

        self.population.step(progress)
        
        super().step(progress)