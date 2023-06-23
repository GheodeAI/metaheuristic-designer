from __future__ import annotations
from typing import Union
from copy import copy
import warnings
from ...ParamScheduler import ParamScheduler
from ...Algorithm import Algorithm
from ...Operator import Operator
from ...Operators import OperatorMeta
from ...SurvivorSelection import SurvivorSelection
import time

class RVNS(Algorithm):
    """
    Reduced Variable Neighborhood Search

    As seen in:
        Hansen, P., & Mladenovic, N. (2003). A tutorial on variable neighborhood search. Les Cahiers du GERAD ISSN, 711, 2440.
    """

    def __init__(self, pop_init: Initializer, op_list: List[Operator], selection_op: SurvivorSelection = None,
                 params: Union[ParamScheduler, dict] = {}, name: str = "RVNS"):

        self.op_list = op_list
        self.perturb_op = OperatorMeta("Pick", op_list, {"init_idx": 0})

        self.current_op = 0

        if selection_op is None:
            selection_op = SurvivorSelection("One-to-One")
        self.selection_op = selection_op

        if pop_init.pop_size > 1:
            pop_init.pop_size = 1
            warnings.warn("The RVNS algorithm work on a single individual. The population size has been set to 1.", stacklevel=2)

        super().__init__(pop_init, params=params, name=name)

    def perturb(self, indiv_list, objfunc, progress=0, history=None):
        """
        Performs a step of the algorithm
        """
    
        offspring = []
        for indiv in indiv_list:

            # Perturb individual
            new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best, self.pop_init)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)

            offspring.append(new_indiv)

        # Keep best individual regardless of selection method
        current_best = max(offspring, key=lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best
        
        return offspring

    def select_individuals(self, population, offspring, progress=0, history=None):
        new_population = self.selection_op(population, offspring)

        if new_population[0].id == population[0].id:
            self.perturb_op.chosen_idx += 1
        else:
            self.perturb_op.chosen_idx = 0
        
        return new_population

    def update_params(self, progress=0):
        """
        Updates the parameters of each component of the algorithm
        """

        if isinstance(self.perturb_op, Operator):
            self.perturb_op.step(progress)
        
        if self.perturb_op.chosen_idx >= len(self.op_list) or self.perturb_op.chosen_idx < 0:
            self.perturb_op.chosen_idx = 0
    
    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """

        idx = self.perturb_op.chosen_idx

        print(f"\tCurrent Operator: {idx}/{len(self.op_list)}, {self.op_list[idx].name}")
        # time.sleep(0.25)
        
        