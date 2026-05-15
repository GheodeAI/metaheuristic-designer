"""
Debug operator registry and factory.
"""

from typing import Optional

from ...encoding import Encoding
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


def create_debug_operator(
    method: str,
    encoding: Optional[Encoding] = None,
    name: Optional[str] = None,
    **kwargs
) -> OperatorFromLambda:
    """
    Create a debug operator by name.

    Parameters
    ----------
    method : str
        Key into :data:`debug_ops_map`.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    name : str, optional
        Display name; defaults to *method*.
    **kwargs
        Forwarded to the operator function.

    Returns
    -------
    OperatorFromLambda
        The wrapped debug operator.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=debug_ops_map[method.lower()], name=name, encoding=encoding, **kwargs)
