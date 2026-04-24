from ..operator import OperatorFromLambda
from .operator_functions.permutation import *
from .operator_functions.utils import OperatorVectorDef

perm_ops_map = {
    "swap":                    OperatorVectorDef(permute_mutation, forced_params={"N": 2}),
    "swap_mutation":           OperatorVectorDef(permute_mutation, forced_params={"N": 2}),
    "two_swap":                OperatorVectorDef(permute_mutation, forced_params={"N": 2}),

    "shift":                   OperatorVectorDef(roll_mutation, forced_params={"N": 1}),
    "insert":                  OperatorVectorDef(roll_mutation, forced_params={"N": 1}),
    "roll1":                   OperatorVectorDef(roll_mutation, forced_params={"N": 1}),
    "block_shift":             OperatorVectorDef(roll_mutation, forced_params={"N": 1}),

    "scramble":                OperatorVectorDef(permute_mutation),
    "perm":                    OperatorVectorDef(permute_mutation),
    "scramble_mutation":       OperatorVectorDef(permute_mutation),
    "permutation_mutation":    OperatorVectorDef(permute_mutation),
    "permute_components":      OperatorVectorDef(permute_mutation),

    "invert":                  OperatorVectorDef(invert_mutation),
    "reverse":                 OperatorVectorDef(invert_mutation),
    "inversion_mutation":      OperatorVectorDef(invert_mutation),

    "roll":                    OperatorVectorDef(roll_mutation),
    "roll_mutation":           OperatorVectorDef(roll_mutation),
    "cyclic_shift":            OperatorVectorDef(roll_mutation),

    "pmx":                     OperatorVectorDef(pmx),
    "pmx_crossover":           OperatorVectorDef(pmx),
    "partially_mapped_crossover": OperatorVectorDef(pmx),

    "ox":                      OperatorVectorDef(order_cross),
    "ordercross":              OperatorVectorDef(order_cross),
    "order_crossover":         OperatorVectorDef(order_cross),
}

def create_permutation_operator(method, encoding=None, **kwargs):
    return OperatorFromLambda(
        operator_fn=perm_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )
