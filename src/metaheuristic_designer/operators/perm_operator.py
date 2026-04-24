from ..operator import OperatorFromLambda
from .operator_functions.permutation import *
from .operator_functions.utils import OperatorVectorDef

perm_ops_map = {
    "swap": OperatorVectorDef(permute_mutation, forced_params={"N": 2}),
    "insert": OperatorVectorDef(roll_mutation, forced_params={"N": 1}),
    "scramble": OperatorVectorDef(permute_mutation),
    "perm": OperatorVectorDef(permute_mutation),
    "invert": OperatorVectorDef(invert_mutation),
    "roll": OperatorVectorDef(roll_mutation),
    "pmx": OperatorVectorDef(pmx),
    "ordercross": OperatorVectorDef(order_cross),
}

def create_permutation_operator(method, encoding=None, **kwargs):
    return OperatorFromLambda(
        operator_fn=perm_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )
