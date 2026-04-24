""" 
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from .operator_functions.utils import OperatorVectorDef
from .operator_functions.crossover import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    multiparent_discrete_crossover,
    averaged_crossover,
    blx_alpha_crossover,
    sbx_crossover,
    bitwise_xor_crossover,
    cross_inter_avg
)
from ..operator import OperatorFromLambda
crossover_ops_map = {
    "1point":                OperatorVectorDef(one_point_crossover),
    "onepoint":              OperatorVectorDef(one_point_crossover),
    "one_point":             OperatorVectorDef(one_point_crossover),
    "one_point_crossover":   OperatorVectorDef(one_point_crossover),

    "2point":                OperatorVectorDef(two_point_crossover),
    "twopoint":              OperatorVectorDef(two_point_crossover),
    "two_point":             OperatorVectorDef(two_point_crossover),
    "two_point_crossover":   OperatorVectorDef(two_point_crossover),

    "uniform":               OperatorVectorDef(uniform_crossover),
    "uniform_crossover":     OperatorVectorDef(uniform_crossover),

    "multicross":            OperatorVectorDef(multiparent_discrete_crossover),
    "multi_parent":          OperatorVectorDef(multiparent_discrete_crossover),
    "multi_parent_crossover":OperatorVectorDef(multiparent_discrete_crossover),
    "multi_parent_discrete_crossover": OperatorVectorDef(multiparent_discrete_crossover),

    "avgcross":              OperatorVectorDef(averaged_crossover),
    "averagecross":          OperatorVectorDef(averaged_crossover),
    "average_crossover":     OperatorVectorDef(averaged_crossover),
    "intermediate_crossover":OperatorVectorDef(averaged_crossover),

    "crossinteravg":         OperatorVectorDef(cross_inter_avg),
    "interavg":              OperatorVectorDef(cross_inter_avg),
    "intermediate_avg":      OperatorVectorDef(cross_inter_avg),
    "multi_parent_avg":      OperatorVectorDef(cross_inter_avg),

    "blxalpha":              OperatorVectorDef(blx_alpha_crossover),
    "blx_alpha":             OperatorVectorDef(blx_alpha_crossover),
    "blx_alpha_crossover":   OperatorVectorDef(blx_alpha_crossover),

    "sbx":                   OperatorVectorDef(sbx_crossover),
    "sbx_crossover":         OperatorVectorDef(sbx_crossover),
    "simulated_binary":      OperatorVectorDef(sbx_crossover),

    "xorcross":              OperatorVectorDef(bitwise_xor_crossover),
    "xor_crossover":         OperatorVectorDef(bitwise_xor_crossover),
    "bitwise_xor":           OperatorVectorDef(bitwise_xor_crossover),
    "flipcross":             OperatorVectorDef(bitwise_xor_crossover),
}

def create_crossover_operator(method, encoding=None, random_state=None, **kwargs):
    return OperatorFromLambda(
        operator_fn=crossover_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        random_state=random_state,
        **kwargs
    )
