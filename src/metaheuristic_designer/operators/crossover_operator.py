""" 
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from dataclasses import dataclass
from .operator_functions.utils import OperatorVectorDef
from .operator_functions.crossover import *
from ..operator import OperatorFromLambda

crossover_fn_map = {
    "1point": OperatorVectorDef(cross_1p),
    "2point": OperatorVectorDef(cross_2p),
    "multipoint": OperatorVectorDef(cross_mp),
    "multicross": OperatorVectorDef(multi_cross),
    "weightedavg": OperatorVectorDef(weighted_average_cross),
    "blxalpha": OperatorVectorDef(blxalpha),
    "sbx": OperatorVectorDef(sbx),
    "xorcross": OperatorVectorDef(xor_cross),
    "flipcross": OperatorVectorDef(xor_cross),
    "crossinteravg": OperatorVectorDef(cross_inter_avg),
}

def create_crossover_operator(method, encoding=None, vectorized=True, **kwargs):
    return OperatorFromLambda(
        operator_fn=crossover_fn_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )
