from ..operator import OperatorFromLambda
from .operator_functions.utils import OperatorVectorDef,dummy_op

debug_ops_map = {
    "dummy":          OperatorVectorDef(dummy_op),
    "constant":       OperatorVectorDef(dummy_op),
    "set_to_value":   OperatorVectorDef(dummy_op),

    "zeros":          OperatorVectorDef(dummy_op, forced_params={"f": 0}),
    "ones":           OperatorVectorDef(dummy_op, forced_params={"f": 1}),
}

def create_debug_operator(method, encoding=None, **kwargs):
    """
    Create operators that utilize the Initializer interface for random generation.
    """

    return OperatorFromLambda(
        operator_fn=debug_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )