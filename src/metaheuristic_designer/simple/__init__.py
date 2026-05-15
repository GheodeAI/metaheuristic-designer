"""
Ready-to-run wrappers that build complete algorithms from a few hyperparameters.
"""

from .genetic_algorithm import genetic_algorithm_binary, genetic_algorithm_permutation, genetic_algorithm_discrete, genetic_algorithm_real
from .hill_climb import hill_climb_binary, hill_climb_permutation, hill_climb_discrete, hill_climb_real
from .local_search import local_search_binary, local_search_permutation, local_search_discrete, local_search_real
from .particle_swarm import particle_swarm_binary, particle_swarm_discrete, particle_swarm_real
from .random_search import random_search_binary, random_search_permutation, random_search_discrete, random_search_real
from .evolution_strategy import evolution_strategy_binary, evolution_strategy_permutation, evolution_strategy_discrete, evolution_strategy_real
from .differential_evolution import differential_evolution_binary, differential_evolution_discrete, differential_evolution_real
from .simulated_annealing import simulated_annealing_binary, simulated_annealing_permutation, simulated_annealing_discrete, simulated_annealing_real
from .bayesian_optimization import bayesian_optimization_binary, bayesian_optimization_discrete, bayesian_optimization_real

__all__ = [
    "genetic_algorithm_binary",
    "genetic_algorithm_permutation",
    "genetic_algorithm_discrete",
    "genetic_algorithm_real",
    "hill_climb_binary",
    "hill_climb_permutation",
    "hill_climb_discrete",
    "hill_climb_real",
    "local_search_binary",
    "local_search_permutation",
    "local_search_discrete",
    "local_search_real",
    "particle_swarm_binary",
    "particle_swarm_discrete",
    "particle_swarm_real",
    "random_search_binary",
    "random_search_permutation",
    "random_search_discrete",
    "random_search_real",
    "evolution_strategy_binary",
    "evolution_strategy_permutation",
    "evolution_strategy_discrete",
    "evolution_strategy_real",
    "differential_evolution_binary",
    "differential_evolution_discrete",
    "differential_evolution_real",
    "simulated_annealing_binary",
    "simulated_annealing_permutation",
    "simulated_annealing_discrete",
    "simulated_annealing_real",
    "bayesian_optimization_binary",
    "bayesian_optimization_discrete",
    "bayesian_optimization_real",
]
