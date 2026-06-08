"""
Parent selection registry and factory.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Optional
from ..parent_selection_base import ParentSelection, ParentSelectionFromLambda, NullParentSelection
from ..population import Population
from .parent_selection_functions import repeating_selection, prob_tournament, select_best, roulette, shuffle_population, uniform_selection, sus
from ..utils import RNGLike, null_aliases

logger = logging.getLogger(__name__)


@dataclass
class ParentSelectionDef:
    """Wrapper that turns a raw parent-selection function into a callable.

    Parameters
    ----------
    selection_fn : callable
        Function ``(fitness, amount, rng, **kwargs) -> indices``.
    params : dict, optional
        Default keyword arguments merged with user-supplied ones.
    forced_params : dict, optional
        Keyword arguments that always override user-supplied ones.
    """

    selection_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, amount: int = None, rng=None, **kwargs) -> Population:
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return self.selection_fn(population.fitness, amount, rng, **modified_kwargs)


# fmt: off
parent_sel_map = {
    # Tournament
    "tournament":               ParentSelectionDef(prob_tournament, forced_params={"prob": 1.0}, params={"tournament_size": 3}),
    "tournament_selection":     ParentSelectionDef(prob_tournament, forced_params={"prob": 1.0}, params={"tournament_size": 3}),

    # Probabilistic tournament
    "probabilistic_tournament": ParentSelectionDef(prob_tournament, params={"tournament_size": 3, "prob": 0.5}),

    # Keep best
    "best":                     ParentSelectionDef(select_best),
    "truncation":               ParentSelectionDef(select_best),
    "select_best":              ParentSelectionDef(select_best),

    # Random selection (with replacement)
    "random":                   ParentSelectionDef(uniform_selection),
    "random_with_replacement":  ParentSelectionDef(uniform_selection),
    "uniform":                  ParentSelectionDef(uniform_selection),

    # Random selection (without replacement)
    "random_without_replacement": ParentSelectionDef(shuffle_population),
    "random_subset":            ParentSelectionDef(shuffle_population),
    "shuffle":                  ParentSelectionDef(shuffle_population),
    "permute":                  ParentSelectionDef(shuffle_population),

    # Repeat
    "repeat":                   ParentSelectionDef(repeating_selection),
    "replicate":                ParentSelectionDef(repeating_selection),

    # Roulette
    "roulette":                 ParentSelectionDef(roulette),
    "fitness_proportional":     ParentSelectionDef(roulette, forced_params={"method": "fit_prop"}),
    "fit_prop":                 ParentSelectionDef(roulette, forced_params={"method": "fit_prop"}),
    "proportional":             ParentSelectionDef(roulette, forced_params={"method": "fit_prop"}),
    "pro":                      ParentSelectionDef(roulette, forced_params={"method": "fit_prop"}),
    "std_roulette":             ParentSelectionDef(roulette, forced_params={"method": "sigma_scale"}),
    "sigma_scaling":            ParentSelectionDef(roulette, forced_params={"method": "sigma_scale"}),
    "rank_roulette":            ParentSelectionDef(roulette, forced_params={"method": "lin_rank"}),
    "linear_rank":              ParentSelectionDef(roulette, forced_params={"method": "lin_rank"}),
    "exp_rank_roulette":        ParentSelectionDef(roulette, forced_params={"method": "exp_rank"}),
    "exponential_rank":         ParentSelectionDef(roulette, forced_params={"method": "exp_rank"}),

    # Stochastic Universal Sampling (SUS)
    "sus":                      ParentSelectionDef(sus),
    "stochastic_universal_sampling": ParentSelectionDef(sus),
    "sus_fitness_proportional": ParentSelectionDef(sus, forced_params={"method": "fit_prop"}),
    "sus_fit_prop":             ParentSelectionDef(sus, forced_params={"method": "fit_prop"}),
    "sus_proportional":         ParentSelectionDef(sus, forced_params={"method": "fit_prop"}),
    "sus_prop":                 ParentSelectionDef(sus, forced_params={"method": "fit_prop"}),
    "sus_std":                  ParentSelectionDef(sus, forced_params={"method": "sigma_scale"}),
    "sus_sigma":                ParentSelectionDef(sus, forced_params={"method": "sigma_scale"}),
    "sus_rank":                 ParentSelectionDef(sus, forced_params={"method": "lin_rank"}),
    "sus_exp":                  ParentSelectionDef(sus, forced_params={"method": "exp_rank"}),
    "sus_exponential":          ParentSelectionDef(sus, forced_params={"method": "exp_rank"}),
}


def create_parent_selection(method: str, name: Optional[str] = None, amount: Optional[int] = None, rng: Optional[RNGLike] = None, **kwargs) -> ParentSelection:
    """Create a parent selection method by name.

    Parameters
    ----------
    method : str
        Key into :data:`parent_sel_map`, or a null alias.
    name : str, optional
        Display name for the selection method.
    amount : int, optional
        Default number of parents to select.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Additional parameters forwarded to the selection function.

    Returns
    -------
    ParentSelectionFromLambda or NullParentSelection
        The wrapped selection method.
    """

    if name is None:
        name = method

    if method in null_aliases:
        return NullParentSelection(name=name, **kwargs)

    return ParentSelectionFromLambda(selection_fn=parent_sel_map[method.lower()], name=name, amount=amount, rng=rng, **kwargs)


def add_parent_selection_entry(selection_fn: callable, selection_method_name: str):
    """Register a new parent selection method.

    Parameters
    ----------
    selection_fn : callable
        A function with the parent selection signature.
    selection_method_name : str
        Name under which to register the method.  If it already exists,
        a warning is logged.
    """

    ParentSelectionFromLambda._validate_function(selection_fn)

    if selection_method_name in parent_sel_map:
        logger.warning('Overwritten parent selection method "%s".', selection_method_name)
    parent_sel_map[selection_method_name] = selection_fn

    logger.info('Added a new parent selection method "%s".', selection_method_name)

def list_parent_selection_methods() -> list[str]:
    """Return a list of all registered parent selection method names.

    Returns
    -------
    list of str
    """

    return list(parent_sel_map.keys())
