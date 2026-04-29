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
    cross_inter_avg,
)
from ..operator import OperatorFromLambda

# fmt: off
crossover_ops_map = {
    # 1 point crossover
    "1point": OperatorVectorDef(one_point_crossover),
    "onepoint": OperatorVectorDef(one_point_crossover),
    "one_point": OperatorVectorDef(one_point_crossover),
    "one_point_crossover": OperatorVectorDef(one_point_crossover),

    # 2 point crossover
    "2point": OperatorVectorDef(two_point_crossover),
    "twopoint": OperatorVectorDef(two_point_crossover),
    "two_point": OperatorVectorDef(two_point_crossover),
    "two_point_crossover": OperatorVectorDef(two_point_crossover),

    # Uniform crossover
    "multipoint": OperatorVectorDef(uniform_crossover),
    "uniform": OperatorVectorDef(uniform_crossover),
    "uniform_crossover": OperatorVectorDef(uniform_crossover),

    # Multi-parent crossover
    "multicross": OperatorVectorDef(multiparent_discrete_crossover),
    "multi_parent": OperatorVectorDef(multiparent_discrete_crossover),
    "multi_parent_crossover": OperatorVectorDef(multiparent_discrete_crossover),
    "multi_parent_discrete_crossover": OperatorVectorDef(multiparent_discrete_crossover),

    # Arithmetic crossover
    "avgcross": OperatorVectorDef(averaged_crossover),
    "averagecross": OperatorVectorDef(averaged_crossover),
    "average_crossover": OperatorVectorDef(averaged_crossover),
    "arithmetic_crossover": OperatorVectorDef(averaged_crossover),
    "intermediate_crossover": OperatorVectorDef(averaged_crossover),

    # Cross intermediate
    "crossinteravg": OperatorVectorDef(cross_inter_avg),
    "interavg": OperatorVectorDef(cross_inter_avg),
    "intermediate_avg": OperatorVectorDef(cross_inter_avg),
    "multi_parent_avg": OperatorVectorDef(cross_inter_avg),
    
    # BLX-alpha
    "blxalpha": OperatorVectorDef(blx_alpha_crossover),
    "blx_alpha": OperatorVectorDef(blx_alpha_crossover),
    "blx_alpha_crossover": OperatorVectorDef(blx_alpha_crossover),
    
    # SBX
    "sbx": OperatorVectorDef(sbx_crossover),
    "sbx_crossover": OperatorVectorDef(sbx_crossover),
    "simulated_binary": OperatorVectorDef(sbx_crossover),
    "simulated_binary_crossover": OperatorVectorDef(sbx_crossover),

    # XOR crossover
    "xorcross": OperatorVectorDef(bitwise_xor_crossover),
    "xor_crossover": OperatorVectorDef(bitwise_xor_crossover),
    "bitwise_xor_crossover": OperatorVectorDef(bitwise_xor_crossover),
    "bitwise_xor": OperatorVectorDef(bitwise_xor_crossover),
    "flipcross": OperatorVectorDef(bitwise_xor_crossover),
    "bitflip_cross": OperatorVectorDef(bitwise_xor_crossover),
}
# fmt: on


def create_crossover_operator(method, encoding=None, random_state=None, name=None, **kwargs):
    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=crossover_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=False, random_state=random_state, **kwargs)
