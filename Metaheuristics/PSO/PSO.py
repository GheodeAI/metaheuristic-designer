from ..Metaheuristic import *
from .PSOPopulation import *


class PSO(Metaheuristic):
    """
    Particle swarm optimization algorithm 
    """

    def __init__(self, objfunc, params):
        """
        Constructor of the PSO algorithm
        """

        super().__init__("PSO", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.population = PSOPopulation(objfunc, params)


    def restart(self):
        """
        Resets the data of the algorithm
        """

        super().restart()

        self.population = PSOPopulation(self.objfunc, self.params)

    
    def step(self, progress):
        """
        One step of the algorithm
        """

        self.population.particle_step()
        
        self.population.step(progress)

        super().step(progress)