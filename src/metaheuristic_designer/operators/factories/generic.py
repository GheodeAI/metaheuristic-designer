"""
Top-level operator factory and registry.
"""

import logging
from typing import Callable, Optional

from metaheuristic_designer.encoding import Encoding
from ...operator import OperatorFromLambda, NullOperator
from .debug import debug_ops_map
from .random import random_ops_map
from .crossover import crossover_ops_map
from .mutation import mutation_ops_map
from .differential_evolution import de_ops_map
from .permutation import permutation_ops_map
from .swarm import swarm_ops_map
from ..BO_operator import BOOperator
from ...utils import RNGLike, null_aliases

logger = logging.getLogger(__name__)

all_ops_map = {
    "random": random_ops_map,
    "mutation": mutation_ops_map,
    "crossover": crossover_ops_map,
    "permutation": permutation_ops_map,
    "de": de_ops_map,
    "swarm": swarm_ops_map,
    "debug": debug_ops_map,
    "custom": {},
}

bo_aliases = {"bo", "bayesian_optimization"}
order_preserving_registries = {"random", "mutation", "de", "swarm", "debug"}
order_preserving_operators = {}


def create_operator(
    method: str, encoding: Optional[Encoding] = None, rng: Optional[RNGLike] = None, name: Optional[str] = None, **kwargs
) -> OperatorFromLambda:
    """
    Create an operator by name from any registry.

    The *method* string can be a simple key (e.g., ``"gauss"``) or
    dot-separated ``"registry.key"`` (e.g., ``"crossover.one_point"``).

    Parameters
    ----------
    method : str
        Operator key, possibly with registry prefix.
    encoding : Encoding, optional
        Encoding applied to the genotype after the operator runs.
    rng : RNGLike, optional
        Random number generator.
    name : str, optional
        Display name; defaults to *method*.
    **kwargs
        Parameters forwarded to the underlying operator function.

    Returns
    -------
    OperatorFromLambda
        The wrapped operator.

    Raises
    ------
    ValueError
        If the operator cannot be found.
    """

    if name is None:
        name = method

    method_lower = method.lower()
    new_operator = None
    if method_lower in bo_aliases:
        new_operator = BOOperator(name=name, encoding=encoding, rng=rng, **kwargs)
        logger.debug("Created bayesian optimization operator.")
    elif method_lower in null_aliases:
        new_operator = NullOperator(name=name)
        logger.debug("Created null operator.")
    elif "." in method_lower:
        op_reg_name, op_name, *_ = method_lower.split(".")
        if op_reg_name not in all_ops_map:
            raise ValueError(f"Operator registry {op_reg_name} doesn't exist, try one of {all_ops_map.keys()}")

        op_map = all_ops_map[op_reg_name]
        if op_name in op_map:
            preserves_order = (op_reg_name in order_preserving_registries) or (op_name in order_preserving_operators)
            new_operator = OperatorFromLambda(
                operator_fn=op_map[op_name], name=name, encoding=encoding, preserves_order=preserves_order, rng=rng, **kwargs
            )
        else:
            raise ValueError(f"Operator {op_name} not found in the operator registry {op_reg_name}.")
        logger.debug("Created operator from %s registry.", op_reg_name)
    else:
        possible_collision = None
        for op_reg_name, op_map in all_ops_map.items():
            if method_lower in op_map:
                if new_operator is None:
                    preserves_order = (op_reg_name in order_preserving_registries) or (method_lower in order_preserving_operators)
                    new_operator = OperatorFromLambda(
                        operator_fn=op_map[method_lower],
                        name=name,
                        encoding=encoding,
                        preserves_order=preserves_order,
                        rng=rng,
                        **kwargs,
                    )
                    possible_collision = op_reg_name
                    logger.debug("Created operator from %s registry.", op_reg_name)
                else:
                    logger.warning(
                        "Found a name collision on operator %s between registries %s and %s.", method_lower, possible_collision, op_reg_name
                    )

    if new_operator is None:
        raise ValueError(f"Operator {method} not found in the operator registry.")

    return new_operator


def add_operator_entry(operator_fn: callable, operator_name: str, operator_registry: str = "custom", preserves_order=False):
    """Register a new operator so it can be created by :func:`create_operator`.

    Parameters
    ----------
    operator_fn : callable
        A callable that follows the operator signature expected by
        :class:`OperatorFromLambda`.  Usually wrapped with
        :class:`OperatorFnDef`, :class:`OperatorRandomDef`, etc.
    operator_name : str
        Key under which the operator is registered.
    operator_registry : str, optional
        Registry name (default ``"custom"``).  If the registry does
        not exist, it is created.
    preserves_order : bool, optional
        If ``True``, the operator is marked as order-preserving,
        meaning individuals retain their position when applying
        it.  Default ``False``.
    """

    OperatorFromLambda._validate_function(operator_fn)

    if operator_registry not in all_ops_map:
        all_ops_map[operator_registry] = {}
        logger.info('Added a new operator registry named "%s"', operator_registry)

    op_reg_map = all_ops_map[operator_registry]
    if operator_name in op_reg_map:
        logger.warning('Overwritten operator "%s" in registry "%s"', operator_name, operator_registry)
    op_reg_map[operator_name] = operator_fn

    if preserves_order:
        order_preserving_operators.add(operator_name)

    logger.info('Added a new operator "%s" in registry "%s"', operator_name, operator_registry)


def list_operators() -> list[str]:
    """Return a list of all registered operator keys.

    Each key is formatted as ``"registry.operator_name"`` and can be
    passed to :func:`create_operator`.

    Returns
    -------
    list of str
        Fully qualified operator names.
    """

    all_ops_list = []
    for registry_name, registry_map in all_ops_map.items():
        for op_name in registry_map.keys():
            all_ops_list.append(f"{registry_name}.{op_name}")
    return all_ops_list
