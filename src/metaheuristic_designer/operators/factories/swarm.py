from __future__ import annotations
from ..operator_functions.swarm import pso_operator_wrapper
from ..operator_functions.utils import OperatorSwarmDef
from ...operator import OperatorFromLambda

swarm_ops_map = {
    "pso": OperatorSwarmDef(pso_operator_wrapper),
    "particle_swarm": OperatorSwarmDef(pso_operator_wrapper),
}


def create_swarm_operator(method, name=None, **kwargs):

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=swarm_ops_map[method.lower()], name=method, **kwargs)
