from ..Metaheuristic import *
from .GeneticPopulation import *


class Genetic(Metaheuristic):
    """
    Genetic algorithm optimization algorithm
    """

    def __init__(self, objfunc, mutation_op, cross_op, parent_sel_op, surv_op, params):
        """
        Constructor of the Genetic algorithm class
        """

        super().__init__("GA", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.mutation_op = mutation_op
        self.cross_op = cross_op
        self.parent_sel_op = parent_sel_op
        self.surv_op = surv_op
        self.population = GeneticPopulation(objfunc, mutation_op, cross_op, parent_sel_op, surv_op, params)


    def restart(self):
        """
        Resets the data of the algorithm
        """
        super().restart()

        self.population = GeneticPopulation(self.objfunc, self.mutation_op, self.cross_op, self.params)


    def step(self, progress):
        """
        One step of the algorithm
        """

        self.population.cross()
        
        self.population.mutate()

        self.population.selection()
        
        self.population.step(progress)

        super().step(progress)