from __future__ import annotations
from typing import Iterable
import warnings
import numpy as np
from ...Initializer import Initializer
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator
from ...operators import OperatorMeta
from ...selectionMethods import SurvivorSelection


class RVNS(SearchStrategy):
    """
    Reduced Variable Neighborhood Search

    As seen in:
        Hansen, P., & Mladenovic, N. (2003). A tutorial on variable neighborhood search. Les Cahiers du GERAD ISSN, 711, 2440.
    """

    def __init__(
        self,
        initializer: Initializer,
        op_list: Iterable[Operator],
        survivor_sel: SurvivorSelection = None,
        params: ParamScheduler | dict = None,
        name: str = "RVNS",
    ):
        if params is None:
            params = {}

        self.op_list = op_list
        operator = OperatorMeta("Pick", op_list, {"init_idx": 0})

        self.current_op = 0

        if survivor_sel is None:
            survivor_sel = SurvivorSelection("One-to-One")

        if initializer.pop_size > 1:
            initializer.pop_size = 1
            warnings.warn(
                "The RVNS algorithm work on a single individual. The population size has been set to 1.",
                stacklevel=2,
            )

        super().__init__(
            initializer,
            operator=operator,
            survivor_sel=survivor_sel,
            params=params,
            name=name
        )

    def select_individuals(self, population, offspring, **kwargs):
        new_population = super().select_individuals(population, offspring, **kwargs)

        if np.all(new_population.genotype_set == population.genotype_set):
            self.operator.chosen_idx += 1
        else:
            self.operator.chosen_idx = 0

        return new_population

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

        if self.operator.chosen_idx >= len(self.op_list) or self.operator.chosen_idx < 0:
            self.operator.chosen_idx = 0

    def extra_step_info(self):
        idx = self.operator.chosen_idx

        print(f"\tCurrent Operator: {idx}/{len(self.op_list)}, {self.op_list[idx].name}")
