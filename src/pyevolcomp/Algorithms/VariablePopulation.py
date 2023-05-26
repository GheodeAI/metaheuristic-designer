from __future__ import annotations
from ..ParamScheduler import ParamScheduler
from ..SurvivorSelection import SurvivorSelection
from ..ParentSelection import ParentSelection
from ..Algorithm import Algorithm
import random


class VariablePopulation(Algorithm):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, pop_init: Initializer, operator: Operator, params: Union[ParamScheduler, dict] = {}, parent_sel_op: ParentSelection = None, 
                 selection_op: SurvivorSelection = None, name: str = "stpop"):
        """
        Constructor of the GeneticPopulation class
        """

        # Hyperparameters of the algorithm
        self.params = params
        self.operator = operator

        self.n_offspring = params["offspringSize"] if "offspringSize" in params else pop_init.pop_size

        if selection_op is None:
            selection_op = SurvivorSelection("Generational")
        self.selection_op = selection_op

        if parent_sel_op is None:
            parent_sel_op = ParentSelection("Nothing")
        self.parent_sel_op = parent_sel_op

        self.best = None

        super().__init__(pop_init, params=params, name=name)
    
    
    def perturb(self, parent_list, objfunc, progress=0, history=None):
        offspring = []

        while len(offspring) < self.n_offspring:

            # Apply operator
            indiv = random.choice(parent_list)
            new_indiv = self.operator(indiv, parent_list, objfunc, self.best, self.pop_init)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

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

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
