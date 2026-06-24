"""
Bayesian Optimization strategy.
"""

from __future__ import annotations
from typing import Optional


from ...initializer import Initializer
from ...objective_function import ObjectiveFunc
from ...parent_selection_base import ParentSelection
from ...operators.BO_operator import BOOperator
from ..population_based_strategy import PopulationBasedStrategy
from ...utils import RNGLike


class BayesianOptimization(PopulationBasedStrategy):
    """
    Bayesian Optimization using a Gaussian Process surrogate.

    This strategy replaces the usual perturbation operator with a
    :class:`BOOperator`, which fits a GP model to the current
    population and uses an acquisition function to propose new
    candidates.

    Parameters
    ----------
    initializer : Initializer
        Population initializer (provides the starting points).
    parent_sel : ParentSelection, optional
        Parent selection method (default: identity).
    name : str, optional
        Display name (default ``"Bayesian Optimization"``).
    \\*\\*kwargs
        Forwarded to :class:`BOOperator` (e.g., ``batch_size``,
        ``max_samples``, ``kernel``).
    """

    def __init__(
        self,
        initializer: Initializer,
        objfunc: ObjectiveFunc,
        parent_sel: ParentSelection = None,
        name: str = "Bayesian Optimization",
        rng: Optional[RNGLike] = None,
        **kwargs,
    ):
        super().__init__(
            initializer,
            operator=BOOperator(objfunc=objfunc, initializer=initializer, rng=rng, **kwargs),
            parent_sel=parent_sel,
            name=name,
            rng=rng,
            **kwargs,
        )
