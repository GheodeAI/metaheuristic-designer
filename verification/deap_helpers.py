# experiments/deap_helpers.py
"""DEAP canonical algorithm factories - GA and ES only.

DE (Differential Evolution) is omitted because the required `cxDE`
operator is not available in your installed DEAP version.  PSO is
omitted because it requires a custom update loop that is incompatible
with the generic DEAPWrapper built around `eaSimple`.

Both DE and PSO are covered by the PyGMO and Nevergrad wrappers in
the benchmark suite.
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms

from metaheuristic_designer.analysis.external_wrappers.deap_wrapper import DEAPWrapper


# -------------------------------------------------------------------
# Shared toolbox builder
# -------------------------------------------------------------------

def _make_toolbox(objfunc, seed, register_operators, pop_size):
    """Create a standard DEAP toolbox, initial population, stats, hof."""
    random.seed(seed)
    np.random.seed(seed)

    dim = objfunc.dimension

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -5.0, 5.0)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual,
        toolbox.attr_float, n=dim,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        lambda ind: (objfunc.objective(np.array(ind)),),
    )

    register_operators(toolbox, dim)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)

    return toolbox, pop, stats, hof


# -------------------------------------------------------------------
# Canonical Genetic Algorithm factory
# -------------------------------------------------------------------

def canonical_deap_ga(
    objfunc,
    seed,
    budget,
    pop_size=100,
    crossover_prob=0.7,
    mutation_prob=0.3,
    tournament_size=3,
    mutation_sigma=0.1,
    sbx_eta=20.0,
    lower_bound=-5.0,
    upper_bound=5.0,
):
    """Return a DEAP GA (eaSimple) with textbook parameters."""
    ngen = max(1, budget // pop_size)

    def _register_ga_ops(toolbox, dim):
        toolbox.register(
            "mate", tools.cxBlend, alpha=0.5
        )
        toolbox.register(
            "mutate", tools.mutGaussian,
            mu=0.0, sigma=mutation_sigma, indpb=1.0 / dim,
        )
        toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    return DEAPWrapper(
        objfunc,
        build_fn=lambda o, s: _make_toolbox(o, s, _register_ga_ops, pop_size),
        ngen=ngen,
        seed=seed,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        algorithm=algorithms.eaSimple,
    )


# -------------------------------------------------------------------
# Canonical Evolution Strategy (μ+λ) factory
# -------------------------------------------------------------------

def canonical_deap_es(
    objfunc,
    seed,
    budget,
    pop_size=500,
    mutation_sigma=0.1,
):
    """Return a (μ+λ)-ES via DEAP (eaSimple with crossover disabled)."""
    ngen = max(1, budget // pop_size)

    def _register_es_ops(toolbox, dim):
        toolbox.register(
            "mate", tools.cxSimulatedBinaryBounded,
            eta=20.0, low=-5.0, up=5.0,
        )
        toolbox.register(
            "mutate", tools.mutGaussian,
            mu=0.0, sigma=mutation_sigma, indpb=1.0,
        )
        toolbox.register("select", tools.selTournament, tournsize=2)

    return DEAPWrapper(
        objfunc,
        build_fn=lambda o, s: _make_toolbox(o, s, _register_es_ops, pop_size),
        ngen=ngen,
        seed=seed,
        cxpb=0.0,
        mutpb=1.0,
        algorithm=algorithms.eaSimple,
    )