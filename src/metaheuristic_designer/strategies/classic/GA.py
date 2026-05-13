from __future__ import annotations
from typing import Optional

from ...initializer import Initializer
from ...operator import Operator, NullOperator
from ...survivor_selection_base import SurvivorSelection
from ...parent_selection_base import ParentSelection
from ...operators import CompositeOperator, BranchOperator
from ..static_population import StaticPopulation
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_random_state, RNGLike


class GA(StaticPopulation):
    """
    Genetic algorithm
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
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        # We need to do the check earlier since it will be injected into the operators
        # and we want everything to share the random state if possible.
        random_state = check_random_state(random_state)

        prob_mut_op = BranchOperator([mutation_op, NullOperator()], method="Random", p=mutation_prob, random_state=random_state)

        # Override the crossover probability if possible
        if hasattr(crossover_op.params, "crossover_prob"):
            crossover_op.update_kwargs(crossover_prob=crossover_prob)

        evolve_op = CompositeOperator([crossover_op, prob_mut_op])

        super().__init__(
            initializer, operator=evolve_op, parent_sel=parent_sel, survivor_sel=survivor_sel, name=name, random_state=random_state, **kwargs
        )
