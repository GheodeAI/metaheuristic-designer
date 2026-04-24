""" 
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from .operator_functions.utils import OperatorVectorDef
from .operator_functions.differential_evolution import (
    differential_evolution_best1,
    differential_evolution_rand1,
    differential_evolution_best2,
    differential_evolution_rand2,
    differential_evolution_current_to_rand1,
    differential_evolution_current_to_best1,
    differential_evolution_current_to_pbest1
)
from ..operator import OperatorFromLambda

de_ops_map = {
    "de/rand/1": OperatorVectorDef(differential_evolution_rand1),
    "de/best/1": OperatorVectorDef(differential_evolution_best1), 
    "de/rand/2": OperatorVectorDef(differential_evolution_rand2),
    "de/best/2": OperatorVectorDef(differential_evolution_best2),
    "de/current-to-rand/1": OperatorVectorDef(differential_evolution_current_to_rand1),
    "de/current-to-best/1": OperatorVectorDef(differential_evolution_current_to_best1),
    "de/current-to-pbest/1": OperatorVectorDef(differential_evolution_current_to_pbest1),
}

def create_differential_evolution_operator(method, encoding=None, vectorized=True, **kwargs):
    """

    Parameters
    ----------
    method
        _description_
    encoding, optional
        _description_, by default None

    Returns
    -------
        _description_
    """

    return OperatorFromLambda(
        operator_fn=de_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )
