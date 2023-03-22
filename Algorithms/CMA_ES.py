import numpy as np
from .ES import ES
from typing import Union, List
from ..BaseAlgorithm import BaseAlgorithm
from ..Individual import Indiv
from ..Operators import Operator, OperatorReal, OperatorMeta
from ..SurvivorSelection import SurvivorSelection
from ..ParentSelection import ParentSelection
from .StaticPopulation import StaticPopulation
from ..ParamScheduler import ParamScheduler
from ..Decoders import CMADecoder

class CMA_ES(ES):
    def __init__(self, mutation_op: Operator, cross_op: Operator, parent_sel_op: ParentSelection, selection_op: SurvivorSelection, 
                       params: Union[ParamScheduler, dict] = {}, name: str = "ES", population: List[Indiv] = None):

        

        parent_select = ParentSelection("Nothing")
        selection = SurvivorSelection("(m+n)")

        cross = OperatorReal("Multicross", {"N":params["HMS"]})
        
        mutate1 = OperatorReal("MutNoise", {"method":"Gauss", "F":params["BW"], "Cr":params["HMCR"] * params["PAR"]})
        rand1 = OperatorReal("RandomMask", {"Cr":1-params["HMCR"]})

        mutate = OperatorMeta("Sequence", [mutate1, rand1])
        

        super().__init__(mutate, cross, parent_select, selection, params, name, population)
    
    def initialize(self, objfunc):
        objfunc.decoder = CMADecoder(params["nparams"], pre_decoder=objfunc.decoder)
        super().initialize(objfunc)