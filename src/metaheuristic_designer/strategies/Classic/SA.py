from __future__ import annotations
from typing import Union
import random
import numpy as np
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...Operator import Operator


class SA(SearchStrategy):
    """
    Simulated annealing
    """

    def __init__(
        self,
        pop_init: Initializer,
        perturb_op: Operator,
        params: Union[ParamScheduler, dict] = {},
        name: str = "SA",
    ):
        self.iter = params.get("iter", 100)
        self.temp_init = params.get("temp_init", 100)
        self.temp = self.temp_init
        self.alpha = params.get("alpha", 0.99)

        self.perturb_op = perturb_op

        super().__init__(pop_init, params=params, name=name)

    def perturb(self, indiv_list, objfunc, **kwargs):
        indiv = indiv_list[0]
        for _ in range(self.iter):
            new_indiv = self.perturb_op(
                indiv, indiv_list, objfunc, self.best, self.pop_init
            )
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)

            # Accept the new solution even if it is worse with a probability
            p = np.exp(-1 / self.temp)
            if new_indiv.fitness > indiv.fitness or random.random() < p:
                indiv = new_indiv

            if new_indiv.fitness > self.best.fitness:
                self.best = new_indiv

        return [indiv]

    def update_params(self, **kwargs):
        progress = kwargs["progress"]

        if isinstance(self.perturb_op, Operator):
            self.perturb_op.step(progress)

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
            self.iter = round(self.params["iter"])
            self.alpha = self.params["alpha"]

        self.temp = self.temp * self.alpha

    def extra_step_info(self):
        print()
        print(f"\tTemperature: {float(self.temp):0.3}")
        print(f"\tAccept prob: {np.exp(-1 / self.temp):0.3}")
