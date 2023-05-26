from __future__ import annotations
import numpy as np
# from .ES import ES
from typing import Union, List
from copy import copy
from ..Individual import Individual
from ..Operators import OperatorReal, OperatorMeta
from ..SurvivorSelection import SurvivorSelection
from ..ParentSelection import ParentSelection
from .StaticPopulation import StaticPopulation
from .VariablePopulation import VariablePopulation
from .HillClimb import HillClimb
from ..ParamScheduler import ParamScheduler


class ES(VariablePopulation):
    def __init__(self, pop_init: Initializer, mutation_op: Operator, cross_op: Operator, parent_sel_op: ParentSelection, 
                 selection_op: SurvivorSelection, params: Union[ParamScheduler, dict] = {}, name: str = "ES"):

        evolve_op = OperatorMeta("Sequence", [mutation_op, cross_op])

        super().__init__(
            pop_init, 
            evolve_op,
            params=params,
            parent_sel_op=parent_sel_op,
            selection_op=selection_op, 
            name=name
        )


class GA(VariablePopulation):
    def __init__(self, pop_init: Initializer, mutation_op: Operator, cross_op: Operator, parent_sel_op: ParentSelection, 
                 selection_op: SurvivorSelection, params: Union[ParamScheduler, dict] = {}, name: str = "GA"):

        self.pmut = params["pmut"] if "pmut" in params else 0.1
        self.pcross = params["pcross"] if "pcross" in params else 0.9

        null_operator = OperatorReal("Nothing", {})

        prob_mut_op = OperatorMeta("Branch", [mutation_op, null_operator], {"p": self.pmut})
        prob_cross_op = OperatorMeta("Branch", [cross_op, null_operator], {"p": self.pcross}) 

        evolve_op = OperatorMeta("Sequence", [prob_mut_op, prob_cross_op])

        super().__init__(
            pop_init, 
            evolve_op,
            params=params,
            parent_sel_op=parent_sel_op,
            selection_op=selection_op, 
            name=name
        )

    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """

        popul_matrix = np.array(list(map(lambda x: x.genotype, self.population)))
        divesity = popul_matrix.std(axis=1).mean()
        print(f"\tdiversity: {divesity:0.3}")


class HS(ES):
    def __init__(self, pop_init: Initializer, params: Union[ParamScheduler, dict] = {}, name: str = "HS"):

        params["offspringSize"] = 1

        parent_select = ParentSelection("Nothing")
        selection = SurvivorSelection("(m+n)")

        HSM = pop_init.pop_size
        cross = OperatorReal("Multicross", {"Nindiv": HSM})

        mutate1 = OperatorReal("MutNoise", {"method": "Gauss", "F": params["BW"], "Cr": params["HMCR"] * params["PAR"]})
        rand1 = OperatorReal("RandomMask", {"Cr": 1 - params["HMCR"]})

        mutate = OperatorMeta("Sequence", [mutate1, rand1])
        
        super().__init__(
            pop_init, 
            mutate, 
            cross,
            parent_sel_op,
            selection_op, 
            params=params,
            name=name
        )


class DE(StaticPopulation):
    def __init__(self, pop_init: Initializer, de_op: Operator, params: Union[ParamScheduler, dict] = {}, selection_op: SurvivorSelection = None, name: str = "DE"):
        if selection_op is None:
            selection_op = SurvivorSelection("One-to-one")

        super().__init__(
            pop_init, 
            de_op, 
            params=params, 
            selection_op=selection_op, 
            name=name
        )


class PSO(StaticPopulation):
    def __init__(self, pop_init: Initializer, params: Union[ParamScheduler, dict] = {}, pso_op: Operator = None, name: str = "PSO"):
        if pso_op is None:
            w = params["w"] if "w" in params else 0.7
            c1 = params["c1"] if "c1" in params else 1.5
            c2 = params["c2"] if "c2" in params else 1.5
            pso_op = OperatorReal("PSO", ParamScheduler("Linear", {"w": w, "c1": c1, "c2": c2}))

        selection_op = SurvivorSelection("Generational")

        super().__init__(
            pop_init,
            pso_op,
            params=params,
            selection_op=selection_op,
            name=name
        )

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


class CRO(StaticPopulation):
    def __init__(self, pop_init: Initializer, mutate: Operator, cross: Operator, params: Union[ParamScheduler, dict] = {}, name: str = "CRO"):

        evolve_op = OperatorMeta("Branch", [cross, mutate], {"p": params["Fb"]})

        selection_op = SurvivorSelection("CRO", {"Fd": params["Fd"], "Pd": params["Pd"], "attempts": params["attempts"], "maxPopSize": params["popSize"]})
        
        params = copy(params)
        params["popSize"] = round(params["popSize"] * params["rho"])
        
        super().__init__(
            pop_init,
            evolve_op,
            params=params,
            selection_op=selection_op,
            name=name
        )


class NoSearch(StaticPopulation):
    """
    Debug Algorithm that does nothing
    """

    def __init__(self, pop_init: Initializer, params: Union[ParamScheduler, dict] = {}, name: str = "No search"):
        noop = OperatorReal("Nothing", {})
        super().__init__(
            pop_init, 
            noop, 
            params=params,
            name=name
        )

    def perturb(self, parent_list, pop_init, objfunc, progress=0, history=None):
        return parent_list


class RandomSearch(HillClimb):
    def __init__(self, pop_init, name="RandomSearch"):
        op = OperatorReal("Random", {})
        super().__init__(pop_init, op, name=name)


