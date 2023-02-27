from ..Metaheuristic import *
from .SimAnnEvolve import *


class SimAnn(Metaheuristic):
    """
    Simulated Annealing optimization algorithm 
    """

    def __init__(self, objfunc, perturb_op, params):
        """
        Constructor of the Simulated Annealing algorithm
        """

        super().__init__("SA", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.perturb_op = perturb_op
        self.population = SimAnnEvolve(objfunc, perturb_op, params)


    def restart(self):
        """
        Resets the data of the algorithm
        """

        super().restart()

        self.population = SimAnnEvolve(self.objfunc, self.perturb_op, self.params)

    
    def step(self, progress, depredate=True, classic=False):
        """
        One step of the algorithm
        """

        self.population.perturb_and_test()

        #self.perturb_op.step(progress)
        self.population.step(progress)
        super().step(progress)
    
    def step_info(self, gen, start_time):
        """
        Displays information about the current state of the algotithm
        """
        
        super().step_info(gen, start_time)

        print(f"\ttemperature: {self.population.temp:0.3}")
        print(f"\taccept prob: {np.exp(-1/self.population.temp):0.3}")
        print()