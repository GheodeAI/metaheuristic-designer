"""
Evolution Strategy.
"""

from __future__ import annotations
from typing import Optional

from ...initializer import Initializer
from ...operator import Operator
from ...survivor_selection_base import SurvivorSelection
from ...parent_selection_base import ParentSelection
from ...operators import CompositeOperator
from ..shuffled_population_strategy import ShuffledPopulationStrategy
from ...utils import RNGLike, check_random_state


class ES(ShuffledPopulationStrategy):
    """
    Evolution Strategy (μ+λ or μ,λ).

    Applies mutation (and optionally crossover) to the selected
    parents, then selects survivors.  By default, no parent
    selection is performed (all individuals are used).

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    mutation_op : Operator
        Mutation operator.
    crossover_op : Operator, optional
        Crossover operator.  If ``None``, only mutation is applied.
    parent_sel : ParentSelection, optional
        Parent selection (default: use the whole population).
    survivor_sel : SurvivorSelection, optional
        Survivor selection (default: generational).
    offspring_size : int, optional
        Number of offspring per generation.
    name : str, optional
        Display name (default ``"ES"``).
    **kwargs
        Forwarded to :class:`VariablePopulation`.
    """

    def __init__(
        self,
        initializer: Initializer,
        mutation_op: Operator,
        crossover_op: Optional[Operator] = None,
        parent_sel: Optional[ParentSelection] = None,
        survivor_sel: Optional[SurvivorSelection] = None,
        offspring_size: Optional[int] = None,
        name: str = "ES",
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        random_state = check_random_state(random_state)
        if crossover_op is None:
            evolve_op = mutation_op
        else:
            evolve_op = CompositeOperator([mutation_op, crossover_op])

        super().__init__(
            initializer,
            operator=evolve_op,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            offspring_size=offspring_size,
            name=name,
            random_state=random_state,
            **kwargs,
        )
