from __future__ import annotations
from copy import copy
import enum
from enum import Enum
import numpy as np

from .operator_functions.mutation import *
from .operator_functions.crossover import *
from .operator_functions.permutation import *
from .operator_functions.differential_evolution import *
from .operator_functions.swarm import *
from ..operator import Operator
from ..param_scheduler import ParamScheduler
from ..encoding import Encoding
from ..encodings import ParameterExtendingEncoding
from .operator_functions.utils import OperatorVectorDef
from ..operator import OperatorFromLambda

swarm_ops_map = {
    "pso": OperatorVectorDef(pso_operator_wrapper),
    "firefly": OperatorVectorDef(firefly_wrapper),
    "glowworm": OperatorVectorDef(glowworm_wrapper)
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
