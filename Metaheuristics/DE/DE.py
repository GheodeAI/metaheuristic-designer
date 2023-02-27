from ..Metaheuristic import *
from .DEPopulation import *


class DE(Metaheuristic):
    """
    The differential evolution optimization algorithm
    """
    
    def __init__(self, objfunc, diffev_op, replace_op, params):
        """
        Constructor of the Differential Evolution class
        """

        super().__init__("DE", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.diffev_op = diffev_op
        self.replace_op = replace_op
        self.population = DEPopulation(objfunc, diffev_op, replace_op, params)


    def restart(self):
        """
        Resets the data of the algorithm
        """

        super().restart()

        self.population = DEPopulation(self.objfunc, self.diffev_op, self.replace_op, self.params)

    
    def step(self, progress):
        """
        One step of the algorithm
        """

        self.population.evolve()

        self.population.selection()

        self.population.step(progress)
        
        super().step(progress)