from __future__ import annotations
from typing import Union
import numpy as np
from ...selectionMethods import ParentSelectionNull, SurvivorSelectionMulti
from ...operators import OperatorVector, OperatorMeta, OperatorNull
from ..VariablePopulation import VariablePopulation
from multiprocessing import Pool


class NSGAII(VariablePopulation):
    """
    Genetic algorithm
    """

    def __init__(
        self,
        initializer: Initializer,
        mutation_op: Operator,
        cross_op: Operator,
        params: ParamScheduler | dict = {},
        name: str = "NSGA-II",
    ):
        self.pmut = params.get("pmut", 0.1)
        self.pcross = params.get("pcross", 0.9)

        null_operator = OperatorNull()

        prob_mut_op = OperatorMeta("Branch", [mutation_op, null_operator], {"p": self.pmut})
        prob_cross_op = OperatorMeta("Branch", [cross_op, null_operator], {"p": self.pcross})

        evolve_op = OperatorMeta("Sequence", [prob_mut_op, prob_cross_op])

        parent_sel = ParentSelectionNull()
        survivor_sel = SurvivorSelectionMulti("non-dominated-sorting", {"amount": initializer.pop_size})

        super().__init__(
            initializer,
            evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )

    def evaluate_population(self, population, objfunc, parallel=False, threads=8):
        if parallel:
            with Pool(threads) as p:
                result_pairs = p.map(evaluate_indiv, population)
            population, calculated = map(list, zip(*result_pairs))
            objfunc.counter += sum(calculated)
        else:
            [indiv.calculate_fitness() for indiv in population]

        return population
 
    def extra_step_info(self):
        popul_matrix = np.array([x.genotype for x in self.population])
        divesity = popul_matrix.std(axis=1).mean()
        print(f"\tdiversity: {divesity:0.3}")
