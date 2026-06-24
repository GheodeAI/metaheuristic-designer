"""
Differential Evolution operator registry and factory.
"""

from typing import Optional

from ...encoding import Encoding

from ..operator_functions.utils import OperatorFnDef
from ..operator_functions.differential_evolution import (
    differential_evolution_best1,
    differential_evolution_rand1,
    differential_evolution_best2,
    differential_evolution_rand2,
    differential_evolution_current_to_rand1,
    differential_evolution_current_to_best1,
    differential_evolution_current_to_pbest1,
)
from ...operator import OperatorFromLambda

# fmt: off
de_ops_map = {
    "de/rand/1": OperatorFnDef(differential_evolution_rand1),
    "de_rand_1": OperatorFnDef(differential_evolution_rand1),

    "de/best/1": OperatorFnDef(differential_evolution_best1),
    "de_best_1": OperatorFnDef(differential_evolution_best1),
    
    "de/rand/2": OperatorFnDef(differential_evolution_rand2),
    "de_rand_2": OperatorFnDef(differential_evolution_rand2),
    
    "de/best/2": OperatorFnDef(differential_evolution_best2),
    "de_best_2": OperatorFnDef(differential_evolution_best2),
    
    "de/current-to-rand/1": OperatorFnDef(differential_evolution_current_to_rand1),
    "de_current_to_rand_1": OperatorFnDef(differential_evolution_current_to_rand1),
    
    "de/current-to-best/1": OperatorFnDef(differential_evolution_current_to_best1),
    "de_current_to_best_1": OperatorFnDef(differential_evolution_current_to_best1),

    "de/current-to-pbest/1": OperatorFnDef(differential_evolution_current_to_pbest1),
    "de_current_to_pbest_1": OperatorFnDef(differential_evolution_current_to_pbest1),
}
# fmt: on


def create_differential_evolution_operator(
    method: str, encoding: Optional[Encoding] = None, vectorized: bool = True, name: Optional[str] = None, **kwargs
) -> OperatorFromLambda:
    """
    Create a DE operator by name.

    Parameters
    ----------
    method : str
        DE variant string, e.g., ``"de/rand/1"``.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    vectorized : bool, optional
        Unused; kept for interface compatibility.
    name : str, optional
        Display name; defaults to *method*.
    \\*\\*kwargs
        Forwarded to the DE operator function (e.g., ``F``, ``Cr``).

    Returns
    -------
    OperatorFromLambda
        The wrapped DE operator.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=de_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=True, **kwargs)
