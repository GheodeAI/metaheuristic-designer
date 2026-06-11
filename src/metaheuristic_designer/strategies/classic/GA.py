"""
Genetic Algorithm strategy.
"""

from __future__ import annotations
from typing import Optional

from ...initializer import Initializer
from ...operator import Operator, NullOperator
from ...survivor_selection_base import SurvivorSelection
from ...parent_selection_base import ParentSelection
from ...operators import CompositeOperator, BranchOperator
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_rng, RNGLike
from ..population_based_strategy import PopulationBasedStrategy


class GA(PopulationBasedStrategy):
    """
    Genetic Algorithm.

    Combines crossover (applied with probability *crossover_prob*)
    and mutation (applied per individual with probability
    *mutation_prob*) via a :class:`BranchOperator`.  The population
    size is constant.

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    mutation_op : Operator
        Mutation operator (will be applied probabilistically).
    crossover_op : Operator
        Crossover operator (applied pairwise).
    parent_sel : ParentSelection
        Parent selection method.
    survivor_sel : SurvivorSelection
        Survivor selection method.
    name : str, optional
        Display name (default ``"GA"``).
    mutation_prob : float or SchedulableParameter, optional
        Individual-level probability of mutation (default 0.1).
    crossover_prob : float or SchedulableParameter, optional
        Pair-level probability of crossover (default 0.9). If the
        crossover operator supports it, this value is injected
        via ``update_kwargs``.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :class:`StaticPopulation`.
    """

    def __init__(
        self,
        initializer: Initializer,
        mutation_op: Operator,
        crossover_op: Operator,
        parent_sel: ParentSelection,
        survivor_sel: SurvivorSelection,
        name: str = "GA",
        mutation_prob: float | SchedulableParameter = 0.1,
        crossover_prob: float | SchedulableParameter = 0.9,
        rng: Optional[RNGLike] = None,
        **kwargs,
    ):
        # We need to do the check earlier since it will be injected into the operators
        # and we want everything to share the random state if possible.
        rng = check_rng(rng)

        prob_mut_op = BranchOperator([mutation_op, NullOperator()], method="Random", p=mutation_prob, rng=rng)

        # Override the crossover probability if possible
        if hasattr(crossover_op.params, "crossover_prob"):
            crossover_op.update_kwargs(crossover_prob=crossover_prob)

        evolve_op = CompositeOperator([crossover_op, prob_mut_op])

        super().__init__(initializer, operator=evolve_op, parent_sel=parent_sel, survivor_sel=survivor_sel, name=name, rng=rng, **kwargs)
