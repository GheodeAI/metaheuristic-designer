"""
"""

import logging
from .BO_operator import BOOperator
from .random_operator import random_ops_map
from .crossover_operator import crossover_fn_map
from .mutation_operator import mutation_ops_map
from .differential_evolution_operator import de_ops_map
from .perm_operator import perm_ops_map
from .swarm_operator import swarm_ops_map
from ..operator import OperatorFromLambda

logger = logging.getLogger(__name__)

all_ops_map = {
    "random": random_ops_map,
    "mutation": mutation_ops_map,
    "crossover": crossover_fn_map,
    "permutation": perm_ops_map,
    "DE": de_ops_map,
    "swarm": swarm_ops_map,
}

def create_operator(method, encoding=None, random_state=None, **kwargs) :
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

    Raises
    ------
    ValueError
        When the method indicated is not one of the available ones in the registry.
    """

    method_lower = method.lower()
    new_operator = None
    if method == "bo":
        new_operator = BOOperator(encoding=encoding, **kwargs)
    else:
        for op_reg_name, op_map in all_ops_map.items():
            if method_lower in op_map:
                new_operator = OperatorFromLambda(
                    operator_fn=op_map[method_lower],
                    name=method,
                    vectorized=True,
                    encoding=encoding,
                    random_state=random_state,
                    **kwargs
                )
                logger.debug("Created operator from %s registry.", op_reg_name)
                break

    if new_operator is None:
        raise ValueError(f"Operator {method} not found in the operator registry.")

    return new_operator
