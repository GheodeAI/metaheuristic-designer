from __future__ import annotations
from ..operator_functions.swarm import pso_operator_wrapper, firefly_operator
from ...encodings import ParameterExtendingEncoding
from ..operator_functions.utils import OperatorSwarmDef, OperatorVectorDef
from ...operator import OperatorFromLambda

swarm_ops_map = {
    "pso": OperatorSwarmDef(pso_operator_wrapper),
    "firefly": OperatorVectorDef(firefly_operator),
}


def create_swarm_operator(method, encoding, name=None, **kwargs):

    if not isinstance(encoding, ParameterExtendingEncoding):
        raise TypeError("The encoding must inherit from ParameterExtendingEncoding.")

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=swarm_ops_map[method.lower()], name=method, encoding=encoding, **kwargs)
