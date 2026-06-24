"""
Crossover operator registry and factory.
"""

from typing import Optional

from ...encoding import Encoding
from ...utils import RNGLike

from ..operator_functions.utils import OperatorFnDef
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
    "one_point_crossover":              OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),
    "one_point":                        OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),
    "onepoint":                         OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),
    "1point":                           OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 1}),

    # 2 point crossover
    "two_point_crossover":              OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),
    "two_point":                        OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),
    "twopoint":                         OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),
    "2point":                           OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}, forced_params={"k": 2}),

    # k-point crossover
    "k_point_crossover":                OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),
    "k-point_crossover":                OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),
    "kpoint_crossover":                 OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),
    "k_point":                          OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),
    "k-point":                          OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),
    "kpoint":                           OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),
    "multipoint_crossover":             OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),
    "multipoint":                       OperatorFnDef(k_point_crossover, params={"crossover_prob": 1}),

    # Uniform crossover
    "uniform":                          OperatorFnDef(uniform_crossover, params={"crossover_prob": 1}),
    "uniform_crossover":                OperatorFnDef(uniform_crossover, params={"crossover_prob": 1}),

    # Arithmetic crossover
    "avgcross":                         OperatorFnDef(averaged_crossover, params={"crossover_prob": 1}),
    "averagecross":                     OperatorFnDef(averaged_crossover, params={"crossover_prob": 1}),
    "average_crossover":                OperatorFnDef(averaged_crossover, params={"crossover_prob": 1}),
    "arithmetic_crossover":             OperatorFnDef(averaged_crossover, params={"crossover_prob": 1}),
    "intermediate_crossover":           OperatorFnDef(averaged_crossover, params={"crossover_prob": 1}),
    
    # BLX-alpha
    "blend":                            OperatorFnDef(blend_crossover, params={"crossover_prob": 1}),
    "blend_crossover":                  OperatorFnDef(blend_crossover, params={"crossover_prob": 1}),
    "blxalpha":                         OperatorFnDef(blend_crossover, params={"crossover_prob": 1}),
    "blx_alpha":                        OperatorFnDef(blend_crossover, params={"crossover_prob": 1}),
    "blxalpha_crossover":               OperatorFnDef(blend_crossover, params={"crossover_prob": 1}),
    "blx_alpha_crossover":              OperatorFnDef(blend_crossover, params={"crossover_prob": 1}),
    
    # SBX
    "sbx":                              OperatorFnDef(sbx_crossover, params={"crossover_prob": 1}),
    "sbx_crossover":                    OperatorFnDef(sbx_crossover, params={"crossover_prob": 1}),
    "simulated_binary":                 OperatorFnDef(sbx_crossover, params={"crossover_prob": 1}),
    "simulated_binary_crossover":       OperatorFnDef(sbx_crossover, params={"crossover_prob": 1}),

    # XOR crossover
    "xorcross":                         OperatorFnDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "xor_crossover":                    OperatorFnDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "bitwise_xor_crossover":            OperatorFnDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "bitwise_xor":                      OperatorFnDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "flipcross":                        OperatorFnDef(bitwise_xor_crossover, params={"crossover_prob": 1}),
    "bitflip_cross":                    OperatorFnDef(bitwise_xor_crossover, params={"crossover_prob": 1}),

    # ------ multi-parent crossover -------------
    # Multi-parent crossover
    "multicross":                       OperatorFnDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),
    "multiparent":                      OperatorFnDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),
    "multiparent_crossover":            OperatorFnDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),
    "multiparent_discrete_crossover":   OperatorFnDef(multiparent_discrete_crossover, params={"crossover_prob": 1}),

    # Cross intermediate
    "crossinteravg":                    OperatorFnDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "interavg":                         OperatorFnDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "intermediate_avg":                 OperatorFnDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "multiparent_avg":                  OperatorFnDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
    "multiparent_intermediate_crossover": OperatorFnDef(multiparent_intermediate_crossover, params={"crossover_prob": 1}),
}
# fmt: on


def create_crossover_operator(
    method: str, encoding: Optional[Encoding] = None, rng: Optional[RNGLike] = None, name: Optional[str] = None, **kwargs
) -> OperatorFromLambda:
    """
    Create a crossover operator by name.

    Parameters
    ----------
    method : str
        Key into :data:`crossover_ops_map` (e.g., ``"one_point"``,
        ``"uniform"``).
    encoding : Encoding, optional
        Encoding applied to the genotype after crossover.
    rng : RNGLike, optional
        Random number generator.
    name : str, optional
        Display name; defaults to *method*.
    \\*\\*kwargs
        Additional parameters forwarded to the operator function
        (e.g., ``k``, ``crossover_prob``, ``pairing_method``).

    Returns
    -------
    OperatorFromLambda
        The wrapped crossover operator.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=crossover_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=False, rng=rng, **kwargs)
