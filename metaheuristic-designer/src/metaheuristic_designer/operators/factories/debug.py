from ...operator import OperatorFromLambda
from ..operator_functions.utils import OperatorFnDef, dummy_op

debug_ops_map = {
    "debug": OperatorFnDef(dummy_op),
    "dummy": OperatorFnDef(dummy_op),
    "constant": OperatorFnDef(dummy_op),
    "set_to_value": OperatorFnDef(dummy_op),
    "zeros": OperatorFnDef(dummy_op, forced_params={"f": 0}),
    "ones": OperatorFnDef(dummy_op, forced_params={"f": 1}),
}


def create_debug_operator(method, encoding=None, name=None, **kwargs):
    """
    Create operators that utilize the Initializer interface for random generation.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=debug_ops_map[method.lower()], name=name, encoding=encoding, **kwargs)
