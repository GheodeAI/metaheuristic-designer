"""
Operator factory package, provides registries and creation functions.
"""

from .generic import all_ops_map, bo_aliases, null_aliases, create_operator, add_operator_entry, list_operators
from .random import random_ops_map, create_random_operator
from .mutation import mutation_ops_map, create_mutation_operator
from .crossover import crossover_ops_map, create_crossover_operator
from .permutation import permutation_ops_map, create_permutation_operator
from .differential_evolution import de_ops_map, create_differential_evolution_operator
from .swarm import swarm_ops_map, create_swarm_operator
from .debug import debug_ops_map, create_debug_operator

__all__ = [
    "list_operators",
    "add_operator_entry",
    "all_ops_map",
    "bo_aliases",
    "create_crossover_operator",
    "create_debug_operator",
    "create_differential_evolution_operator",
    "create_mutation_operator",
    "create_operator",
    "create_permutation_operator",
    "create_random_operator",
    "create_swarm_operator",
    "crossover_ops_map",
    "de_ops_map",
    "debug_ops_map",
    "mutation_ops_map",
    "null_aliases",
    "permutation_ops_map",
    "random_ops_map",
    "swarm_ops_map",
]
