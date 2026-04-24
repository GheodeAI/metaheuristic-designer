"""

"""

from .operator_functions.random_generation import *
from .operator_functions.utils import OperatorVectorDef, OperatorRandomDef
from ..operator import OperatorFromLambda

random_ops_map = {
    "random": OperatorRandomDef(random_initialize),
    "reset": OperatorRandomDef(random_reset),
    "dummy": OperatorVectorDef(dummy_op),
}

def create_random_operator(method, encoding=None, **kwargs):
    """
    """

    return OperatorFromLambda(
        operator_fn=random_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )