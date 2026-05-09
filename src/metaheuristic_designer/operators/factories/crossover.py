"""
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from ..operator_functions.utils import OperatorVectorDef
from ..operator_functions.crossover import (
    k_point_crossover,
    uniform_crossover,
    multiparent_discrete_crossover,
    averaged_crossover,
    blend_crossover,
    sbx_crossover,
    bitwise_xor_crossover,
    multiparent_intermediate_crossover,
)
from ...operator import OperatorFromLambda

# fmt: off
crossover_ops_map = {
    # ------ dual parent crossover -------------
    # 1 point crossover
    "1point":                           OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),
    "onepoint":                         OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),
    "one_point":                        OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),
    "one_point_crossover":              OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),

    # 2 point crossover
    "2point":                           OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),
    "twopoint":                         OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),
    "two_point":                        OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),
    "two_point_crossover":              OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),

    # k-point crossover
    "multipoint":                       OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),
    "kpoint":                           OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),
    "k-point":                          OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),
    "k_point":                          OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),
    "multipoint_crossover":             OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),
    "kpoint_crossover":                 OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),
    "k-point_crossover":                OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),
    "k_point_crossover":                OperatorVectorDef(k_point_crossover, params={"crossover_prob": 1}),

    # Uniform crossover
    "uniform":                          OperatorVectorDef(uniform_crossover, params={"crossover_prob": 1}),
    "uniform_crossover":                OperatorVectorDef(uniform_crossover, params={"crossover_prob": 1}),

    # Arithmetic crossover
    "avgcross":                         OperatorVectorDef(averaged_crossover, params={"crossover_prob": 1}),
    "averagecross":                     OperatorVectorDef(averaged_crossover, params={"crossover_prob": 1}),
    "average_crossover":                OperatorVectorDef(averaged_crossover, params={"crossover_prob": 1}),
    "arithmetic_crossover":             OperatorVectorDef(averaged_crossover, params={"crossover_prob": 1}),
    "intermediate_crossover":           OperatorVectorDef(averaged_crossover, params={"crossover_prob": 1}),
    
    # BLX-alpha
    "blend":                            OperatorVectorDef(blend_crossover, params={"crossover_prob": 1}),
    "blend_crossover":                  OperatorVectorDef(blend_crossover, params={"crossover_prob": 1}),
    "blxalpha":                         OperatorVectorDef(blend_crossover, params={"crossover_prob": 1}),
    "blx_alpha":                        OperatorVectorDef(blend_crossover, params={"crossover_prob": 1}),
    "blxalpha_crossover":               OperatorVectorDef(blend_crossover, params={"crossover_prob": 1}),
    "blx_alpha_crossover":              OperatorVectorDef(blend_crossover, params={"crossover_prob": 1}),
    
    # SBX
    "sbx":                              OperatorVectorDef(sbx_crossover, params={"crossover_prob": 1}),
    "sbx_crossover":                    OperatorVectorDef(sbx_crossover, params={"crossover_prob": 1}),
    "simulated_binary":                 OperatorVectorDef(sbx_crossover, params={"crossover_prob": 1}),
    "simulated_binary_crossover":       OperatorVectorDef(sbx_crossover, params={"crossover_prob": 1}),

    # XOR crossover
    "xorcross":                         OperatorVectorDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "xor_crossover":                    OperatorVectorDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "bitwise_xor_crossover":            OperatorVectorDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "bitwise_xor":                      OperatorVectorDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "flipcross":                        OperatorVectorDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "bitflip_cross":                    OperatorVectorDef(bitwise_xor_crossover, params={"crossover_prob": 1}),

    # ------ multi-parent crossover -------------
    # Multi-parent crossover
    "multicross":                       OperatorVectorDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),
    "multiparent":                      OperatorVectorDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),
    "multiparent_crossover":            OperatorVectorDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),
    "multiparent_discrete_crossover":   OperatorVectorDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),

    # Cross intermediate
    "crossinteravg":                    OperatorVectorDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "interavg":                         OperatorVectorDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "intermediate_avg":                 OperatorVectorDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "multiparent_avg":                  OperatorVectorDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "multiparent_intermediate_crossover": OperatorVectorDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
}
# fmt: on


def create_crossover_operator(method, encoding=None, random_state=None, name=None, **kwargs):
    if name is None:
        name = method

    return OperatorFromLambda(
        operator_fn=crossover_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=False, random_state=random_state, **kwargs
    )
