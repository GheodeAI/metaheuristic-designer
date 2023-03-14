from .ES import ES
from typing import Union, List
from ..Individual import Indiv
from ..Operators import Operator, OperatorReal, OperatorMeta
from ..SurvivorSelection import SurvivorSelection
from ..ParentSelection import ParentSelection
from .StaticPopulation import StaticPopulation
from ..ParamScheduler import ParamScheduler

class HS(ES):
    def __init__(self, params: Union[ParamScheduler, dict]={}, name: str="HS", population: List[Indiv]=None):
        
        params["popSize"] = params["HMS"]
        params["offspringSize"] = 1

        parent_select = ParentSelection("Nothing")
        selection = SurvivorSelection("(m+n)")

        cross = OperatorReal("Multicross", {"N":params["HMS"]})
        
        mutate1 = OperatorReal("MutNoise", {"method":"Gauss", "F":params["BW"], "Cr":params["HMCR"] * params["PAR"]})
        rand1 = OperatorReal("RandomMask", {"Cr":1-params["HMCR"]})

        mutate = OperatorMeta("Sequence", [mutate1, rand1])
        

        super().__init__(mutate, cross, parent_select, selection, params, name, population)


class DE(StaticPopulation):
    def __init__(self, de_op: Operator, params: Union[ParamScheduler, dict]={}, selection_op: SurvivorSelection = None, name: str="DE", population: List[Indiv]=None):
        if selection_op is None:
            selection_op = SurvivorSelection("One-to-one")

        super().__init__(de_op, params, selection_op, name, population)


class PSO(StaticPopulation):
    def __init__(self, params: Union[ParamScheduler, dict]={}, pso_op: Operator = None, name: str="PSO", population: List[Indiv]=None):
        if pso_op is None:
            w = params["w"] if "w" in params else 0.7
            c1 = params["c1"] if "c1" in params else 1.5
            c2 = params["c2"] if "c2" in params else 1.5
            pso_op = OperatorReal("PSO", ParamScheduler("Linear", {"w":w, "c1":c1, "c2":c2}))
        
        selection_op = SurvivorSelection("Generational")
        
        super().__init__(pso_op, params, selection_op, name, population)