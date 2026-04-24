from __future__ import annotations
from .operator_functions.swarm import (
    pso_operator_wrapper,
    firefly_operator_wrapper,
    glowworm_operator_wrapper
)
from ..encodings import ParameterExtendingEncoding
from .operator_functions.utils import OperatorVectorDef
from ..operator import OperatorFromLambda

swarm_ops_map = {
    "pso": OperatorVectorDef(pso_operator_wrapper),
    "firefly": OperatorVectorDef(firefly_operator_wrapper),
    "glowworm": OperatorVectorDef(glowworm_operator_wrapper)
}

def create_swarm_operator(method, encoding, **kwargs):

    if not isinstance(encoding, ParameterExtendingEncoding):
        raise TypeError("The encoding must inherit from ParameterExtendingEncoding.")

    return OperatorFromLambda(
        operator_fn=swarm_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )
