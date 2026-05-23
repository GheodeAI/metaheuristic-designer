"""
Permutation operator registry and factory.
"""

from typing import Optional

from ...encoding import Encoding
from ...operator import OperatorFromLambda
from ..operator_functions.permutation import permute_mutation, roll_mutation, invert_mutation, pmx, order_cross
from ..operator_functions.utils import OperatorFnDef

# fmt: off
permutation_ops_map = {
    # Swap components
    "swap": OperatorFnDef(permute_mutation, forced_params={"N": 2}),
    "swap_mutation": OperatorFnDef(permute_mutation, forced_params={"N": 2}),
    "two_swap": OperatorFnDef(permute_mutation, forced_params={"N": 2}),

    # Shift
    "shift": OperatorFnDef(roll_mutation, forced_params={"N": 1}),
    "insert": OperatorFnDef(roll_mutation, forced_params={"N": 1}),
    "roll1": OperatorFnDef(roll_mutation, forced_params={"N": 1}),
    "block_shift": OperatorFnDef(roll_mutation, forced_params={"N": 1}),

    # Scramble
    "scramble": OperatorFnDef(permute_mutation),
    "perm": OperatorFnDef(permute_mutation),
    "permutate": OperatorFnDef(permute_mutation),
    "scramble_mutation": OperatorFnDef(permute_mutation),
    "permutation_mutation": OperatorFnDef(permute_mutation),
    "permute_components": OperatorFnDef(permute_mutation),

    # Reverse
    "invert": OperatorFnDef(invert_mutation),
    "reverse": OperatorFnDef(invert_mutation),
    "inversion_mutation": OperatorFnDef(invert_mutation),

    # Roll
    "roll": OperatorFnDef(roll_mutation),
    "roll_mutation": OperatorFnDef(roll_mutation),
    "cyclic_shift": OperatorFnDef(roll_mutation),

    # Partially mapped crossover
    "pmx": OperatorFnDef(pmx),
    "pmx_crossover": OperatorFnDef(pmx),
    "partially_mapped_crossover": OperatorFnDef(pmx),

    # Ordered crossover
    "ox": OperatorFnDef(order_cross),
    "order_cross": OperatorFnDef(order_cross),
    "order_crossover": OperatorFnDef(order_cross),
}
# fmt: on


def create_permutation_operator(method: str, encoding: Optional[Encoding] = None, name: Optional[str] = None, **kwargs) -> OperatorFromLambda:
    """
    Create a permutation operator by name.

    Parameters
    ----------
    method : str
        Key into :data:`permutation_ops_map`.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    name : str, optional
        Display name; defaults to *method*.
    **kwargs
        Forwarded to the operator function.

    Returns
    -------
    OperatorFromLambda
        The wrapped permutation operator.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=permutation_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=False, **kwargs)
