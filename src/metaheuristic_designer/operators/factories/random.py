"""
Random-generation operator registry and factory.
"""

from typing import Optional
from ...encoding import Encoding
from ..operator_functions.random_generation import *
from ..operator_functions import OperatorRandomDef
from ...operator import OperatorFromLambda

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


def create_random_operator(
    method: str,
    encoding: Optional[Encoding] = None,
    name: Optional[str] = None,
    **kwargs
) -> OperatorFromLambda:
    """
    Create a random operator that uses an Initializer for fresh values.

    Parameters
    ----------
    method : str
        Key into :data:`random_ops_map`.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    name : str, optional
        Display name; defaults to *method*.
    **kwargs
        Forwarded to the operator function.

    Returns
    -------
    OperatorFromLambda
        The wrapped random operator.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=random_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=True, **kwargs)
