from ..operator import OperatorFromLambda
from .operator_functions.permutation import *
from .operator_functions.utils import OperatorVectorDef

# fmt: off
perm_ops_map = {
    # Swap components
    "swap": OperatorVectorDef(permute_mutation, forced_params={"N": 2}),
    "swap_mutation": OperatorVectorDef(permute_mutation, forced_params={"N": 2}),
    "two_swap": OperatorVectorDef(permute_mutation, forced_params={"N": 2}),

    # Shift
    "shift": OperatorVectorDef(roll_mutation, forced_params={"N": 1}),
    "insert": OperatorVectorDef(roll_mutation, forced_params={"N": 1}),
    "roll1": OperatorVectorDef(roll_mutation, forced_params={"N": 1}),
    "block_shift": OperatorVectorDef(roll_mutation, forced_params={"N": 1}),

    # Scramble
    "scramble": OperatorVectorDef(permute_mutation),
    "perm": OperatorVectorDef(permute_mutation),
    "permutate": OperatorVectorDef(permute_mutation),
    "scramble_mutation": OperatorVectorDef(permute_mutation),
    "permutation_mutation": OperatorVectorDef(permute_mutation),
    "permute_components": OperatorVectorDef(permute_mutation),

    # Reverse
    "invert": OperatorVectorDef(invert_mutation),
    "reverse": OperatorVectorDef(invert_mutation),
    "inversion_mutation": OperatorVectorDef(invert_mutation),

    # Roll
    "roll": OperatorVectorDef(roll_mutation),
    "roll_mutation": OperatorVectorDef(roll_mutation),
    "cyclic_shift": OperatorVectorDef(roll_mutation),

    # Partially mapped crossover
    "pmx": OperatorVectorDef(pmx),
    "pmx_crossover": OperatorVectorDef(pmx),
    "partially_mapped_crossover": OperatorVectorDef(pmx),

    # Ordered crossover
    "ox": OperatorVectorDef(order_cross),
    "order_cross": OperatorVectorDef(order_cross),
    "order_crossover": OperatorVectorDef(order_cross),
}
# fmt: on


def create_permutation_operator(method, encoding=None, name=None, **kwargs):

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=perm_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=False, **kwargs)
