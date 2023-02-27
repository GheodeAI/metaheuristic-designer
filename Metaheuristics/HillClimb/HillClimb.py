from ..Metaheuristic import *
from .HillClimbEvolve import *


class HillClimb(Metaheuristic):
    """
    Simulated Annealing optimization algorithm 
    """

    def __init__(self, objfunc, perturb_op, params):
        """
        Constructor of the Simulated Annealing algorithm
        """

        super().__init__("HillClimb", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.perturb_op = perturb_op
        self.population = HillClimbEvolve(objfunc, perturb_op, params)


    def restart(self):
        """
        Resets the data of the algorithm
        """

        super().restart()

        self.population = HillClimbEvolve(self.objfunc, self.perturb_op, self.params)

    
    def step(self, progress, depredate=True, classic=False):
        """
        One step of the algorithm
        """

        self.population.perturb_and_test()

        #self.perturb_op.step(progress)
        self.population.step(progress)
        super().step(progress)