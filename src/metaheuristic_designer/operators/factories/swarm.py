"""
Swarm operator registry and factory.
"""

from __future__ import annotations
from typing import Optional
from ..operator_functions.swarm import pso_operator_wrapper
from ..operator_functions.utils import OperatorSwarmDef
from ...operator import OperatorFromLambda

swarm_ops_map = {
    "pso": OperatorSwarmDef(pso_operator_wrapper),
    "particle_swarm": OperatorSwarmDef(pso_operator_wrapper),
}


def create_swarm_operator(
    method: str,
    name: Optional[str] = None,
    **kwargs
) -> OperatorFromLambda:
    """
    Create a swarm operator by name.

    Parameters
    ----------
    method : str
        Key into :data:`swarm_ops_map` (e.g., ``"pso"``).
    name : str, optional
        Display name; defaults to *method*.
    **kwargs
        Forwarded to the operator function.

    Returns
    -------
    OperatorFromLambda
        The wrapped swarm operator.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=swarm_ops_map[method.lower()], name=method, **kwargs)
