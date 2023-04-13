from __future__ import annotations
from ..ParamScheduler import ParamScheduler
from ..SurvivorSelection import SurvivorSelection
from ..Algorithm import Algorithm


class StaticPopulation(Algorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, operator: Operator, params: Union[ParamScheduler, dict] = {}, selection_op: SurvivorSelection = None,
                 name: str = "stpop"):
        """
        Constructor of the GeneticPopulation class
        """

        super().__init__(name)

        # Hyperparameters of the algorithm
        self.params = params
        self.popsize = params["popSize"] if "popSize" in params else 100
        self.operator = operator

        if selection_op is None:
            selection_op = SurvivorSelection("Generational")
        self.selection_op = selection_op

        self.best = None

    def perturb(self, parent_list, objfunc, progress=0, history=None):
        offspring = []
        for indiv in parent_list:

            # Apply operator
            new_indiv = self.operator(indiv, parent_list, objfunc, self.best)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Store best vector for individual
            new_indiv.store_best(indiv)

            # Add to offspring list
            offspring.append(new_indiv)

        # Update best solution
        current_best = max(offspring, key=lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best

        return offspring

    def select_individuals(self, population, offspring, progress=0, history=None):
        return self.selection_op(population, offspring)

    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """

        self.operator.step(progress)
        self.selection_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
