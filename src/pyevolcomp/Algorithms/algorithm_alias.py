from __future__ import annotations
import numpy as np
from .ES import ES
from typing import Union, List
from ..Individual import Individual
from ..Operators import OperatorReal, OperatorMeta
from ..SurvivorSelection import SurvivorSelection
from ..ParentSelection import ParentSelection
from .StaticPopulation import StaticPopulation
from .HillClimb import HillClimb
from ..ParamScheduler import ParamScheduler


class HS(ES):
    def __init__(self, params: Union[ParamScheduler, dict] = {}, name: str = "HS"):

        params["popSize"] = params["HMS"]
        params["offspringSize"] = 1

        parent_select = ParentSelection("Nothing")
        selection = SurvivorSelection("(m+n)")

        cross = OperatorReal("Multicross", {"Nindiv": params["HMS"]})

        mutate1 = OperatorReal("MutNoise", {"method": "Gauss", "F": params["BW"], "Cr": params["HMCR"] * params["PAR"]})
        rand1 = OperatorReal("RandomMask", {"Cr": 1 - params["HMCR"]})

        mutate = OperatorMeta("Sequence", [mutate1, rand1])

        super().__init__(mutate, cross, parent_select, selection, params, name)


class DE(StaticPopulation):
    def __init__(self, de_op: Operator, params: Union[ParamScheduler, dict] = {}, selection_op: SurvivorSelection = None, name: str = "DE"):
        if selection_op is None:
            selection_op = SurvivorSelection("One-to-one")

        super().__init__(de_op, params, selection_op, name)


class PSO(StaticPopulation):
    def __init__(self, params: Union[ParamScheduler, dict] = {}, pso_op: Operator = None, name: str = "PSO"):
        if pso_op is None:
            w = params["w"] if "w" in params else 0.7
            c1 = params["c1"] if "c1" in params else 1.5
            c2 = params["c2"] if "c2" in params else 1.5
            pso_op = OperatorReal("PSO", ParamScheduler("Linear", {"w": w, "c1": c1, "c2": c2}))

        selection_op = SurvivorSelection("Generational")

        super().__init__(pso_op, params, selection_op, name)

    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """

        popul_matrix = np.array(list(map(lambda x: x.genotype, self.population)))
        speed_matrix = np.array(list(map(lambda x: x.speed, self.population)))
        divesity = popul_matrix.std(axis=1).mean()
        mean_speed = speed_matrix.mean()
        print(f"\tdiversity: {divesity:0.3}")
        print(f"\tmean speed: {mean_speed:0.3}")


class NoSearch(StaticPopulation):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, params: Union[ParamScheduler, dict] = {}, name: str = "No search"):
        noop = OperatorReal("Nothing")
        selection_op = SurvivorSelection("Generational")
        super().__init__(noop, params, selection_op, name)

    def perturb(self, parent_list, pop_init, objfunc, progress=0, history=None):
        return parent_list


class RandomSearch(HillClimb):
    def __init__(self, name="RandomSearch"):
        op = OperatorReal("Random", {})
        super().__init__(op, name=name)


