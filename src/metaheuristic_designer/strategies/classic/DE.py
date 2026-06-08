"""
Differential Evolution strategy.
"""

from __future__ import annotations
from typing import Optional
from ...initializer import Initializer
from ...operators import create_differential_evolution_operator
from ...survivor_selection import SurvivorSelection, create_survivor_selection
from ...schedulable_parameter import SchedulableParameter
from ...utils import check_rng, RNGLike
from ..population_based_strategy import PopulationBasedStrategy


class DE(PopulationBasedStrategy):
    """
    Differential Evolution algorithm.

    Uses a DE mutation operator (e.g., ``"DE/best/1"``) and
    one-to-one survivor selection by default.  The population size
    stays constant, and every individual is perturbed each generation.

    Parameters
    ----------
    initializer : Initializer
        Population initializer.
    de_operator_name : str, optional
        DE variant (default ``"DE/best/1"``).
    survivor_sel : SurvivorSelection, optional
        Survivor selection; defaults to one-to-one competition.
    name : str, optional
        Display name (default ``"DE"``).
    rng : RNGLike, optional
        Random number generator.
    F : float or SchedulableParameter, optional
        Scale factor (default 0.8).
    Cr : float or SchedulableParameter, optional
        Crossover probability (default 0.9).
    p : float or SchedulableParameter, optional
        Elite fraction for ``/pbest/`` variants (default 0.1).
    **kwargs
        Forwarded to :class:`StaticPopulation`.
    """

    def __init__(
        self,
        initializer: Initializer,
        de_operator_name: str = "DE/best/1",
        survivor_sel: Optional[SurvivorSelection] = None,
        name: str = "DE",
        rng: Optional[RNGLike] = None,
        F: float | SchedulableParameter = 0.8,
        Cr: float | SchedulableParameter = 0.9,
        p: float | SchedulableParameter = 0.1,
        **kwargs,
    ):
        # We need to do the check earlier since it will be injected into the operator
        # and we want everything to share the random state if possible.
        rng = check_rng(rng)

        if survivor_sel is None:
            survivor_sel = create_survivor_selection("one_to_one")

        super().__init__(
            initializer,
            operator=create_differential_evolution_operator(de_operator_name, rng=rng, F=F, Cr=Cr, p=p),
            survivor_sel=survivor_sel,
            name=name,
            rng=rng,
            **kwargs,
        )
