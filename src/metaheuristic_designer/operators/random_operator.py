""" """

from .operator_functions.random_generation import *
from .operator_functions.utils import OperatorVectorDef, OperatorRandomDef
from ..operator import OperatorFromLambda

# fmt: off
random_ops_map = {
    # Random regeneration
    "random": OperatorRandomDef(random_initialize),
    "reinitialize": OperatorRandomDef(random_initialize),
    "regenerate": OperatorRandomDef(random_initialize),
    "full_random_reset": OperatorRandomDef(random_initialize),

    # Random reset of components
    "reset": OperatorRandomDef(random_reset),
    "reset_n": OperatorRandomDef(random_reset),
    "reset_random": OperatorRandomDef(random_reset),
    "reset_components": OperatorRandomDef(random_reset),
}
# fmt: on


def create_random_operator(method, encoding=None, name=None, **kwargs):
    """
    Create operators that utilize the Initializer interface for random generation.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=random_ops_map[method.lower()], name=method, encoding=encoding, **kwargs)
