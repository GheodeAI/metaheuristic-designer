from __future__ import annotations
from typing import Union
import random
import numpy as np
from ...ParamScheduler import ParamScheduler
from ...SearchStrategy import SearchStrategy
from ...selectionMethods import SurvivorSelection
from ...Operator import Operator
from ..HillClimb import HillClimb


class SA(HillClimb):
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
        self.iter_count = 0
        self.temp_init = params.get("temp_init", 100)
        self.temp = self.temp_init
        self.alpha = params.get("alpha", 0.99)

        selection_op = SurvivorSelection("ProbHillClimb", {"p": np.exp(-1 / self.temp_init)})

        super().__init__(pop_init, perturb_op, selection_op, params=params, name=name)

    def update_params(self, **kwargs):
        progress = kwargs["progress"]

        if isinstance(self.perturb_op, Operator):
            self.perturb_op.step(progress)

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
            self.iter = round(self.params["iter"])
            self.alpha = self.params["alpha"]

        self.iter_count += 1
        if self.iter_count > self.iter:
            self.temp = self.temp * self.alpha
            self.selection_op.params["p"] = np.exp(-1 / self.temp)
            self.iter_count = 0

    def extra_step_info(self):
        print()
        print(f"\tTemp iters: {self.iter_count}/{self.iter}")
        print(f"\tTemperature: {float(self.temp):0.3}")
        print(f"\tAccept prob: {np.exp(-1 / self.temp):0.3}")
