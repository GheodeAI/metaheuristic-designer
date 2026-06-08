"""
Survivor selection registry and factory.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
from ..population import Population
from ..survivor_selection_base import SurvivorSelectionFromLambda, NullSurvivorSelection
from .survivor_selection_functions import (
    generational,
    elitism,
    cond_elitism,
    one_to_one,
    prob_one_to_one,
    many_to_one,
    prob_many_to_one,
    keep_best,
    keep_best_offspring,
)
from ..utils import RNGLike, null_aliases

logger = logging.getLogger(__name__)


@dataclass
class SurvivorSelectionDef:
    """Wrapper that turns a raw survivor-selection function into a callable.

    Parameters
    ----------
    selection_fn : callable
        Function ``(parent_fitness, offspring_fitness, rng, **kwargs) -> indices``.
    params : dict, optional
        Default keyword arguments merged with user-supplied ones.
    forced_params : dict, optional
        Keyword arguments that always override user-supplied ones.
    preserves_order : bool, optional
        If ``True``, the selection method keeps individuals in the same order.
        Default ``False``.
    """

    selection_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)
    preserves_order: bool = False

    def __call__(self, population: Population, offspring: Population, rng=None, **kwargs) -> Population:
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return self.selection_fn(population.fitness, offspring.fitness, rng, **modified_kwargs)


# fmt: off
surv_method_map = {
    # Elitism
    "elitism":                      SurvivorSelectionDef(elitism),

    # Conditional elitism
    "cond_elitism":                 SurvivorSelectionDef(cond_elitism),
    "conditional_elitism":          SurvivorSelectionDef(cond_elitism),

    # Generational
    "generational":                 SurvivorSelectionDef(generational, preserves_order=True),
    "nothing":                      SurvivorSelectionDef(generational, preserves_order=True),

    # Hill climb
    "one_to_one":                   SurvivorSelectionDef(one_to_one, preserves_order=True),
    "hillclimb":                    SurvivorSelectionDef(one_to_one, preserves_order=True),
    "hill_climb":                   SurvivorSelectionDef(one_to_one, preserves_order=True),

    # Probabilistic hill climb
    "prob_one_to_one":              SurvivorSelectionDef(prob_one_to_one, preserves_order=True),
    "prob_hillclimb":               SurvivorSelectionDef(prob_one_to_one, preserves_order=True),
    "prob_hill_climb":              SurvivorSelectionDef(prob_one_to_one, preserves_order=True),
    "probabilistic_one_to_one":     SurvivorSelectionDef(prob_one_to_one, preserves_order=True),
    "probabilistic_hillclimb":      SurvivorSelectionDef(prob_one_to_one, preserves_order=True),
    "probabilistic_hill_climb":     SurvivorSelectionDef(prob_one_to_one, preserves_order=True),

    # Local search
    "many_to_one":                  SurvivorSelectionDef(many_to_one, preserves_order=True),
    "local_search":                 SurvivorSelectionDef(many_to_one, preserves_order=True),

    # Probabilistic Local search
    "prob_many_to_one":             SurvivorSelectionDef(prob_many_to_one, preserves_order=True),
    "prob_local_search":            SurvivorSelectionDef(prob_many_to_one, preserves_order=True),
    "probabilistic_many_to_one":    SurvivorSelectionDef(prob_many_to_one, preserves_order=True),
    "probabilistic_local_search":   SurvivorSelectionDef(prob_many_to_one, preserves_order=True),

    # (mu + lambda)
    "(m+n)":                        SurvivorSelectionDef(keep_best),
    "(mu+lambda)":                  SurvivorSelectionDef(keep_best),
    "mu+lambda":                    SurvivorSelectionDef(keep_best),
    "keep_best":                    SurvivorSelectionDef(keep_best),

    # (mu, lambda)
    "(m,n)":                        SurvivorSelectionDef(keep_best_offspring),
    "(mu,lambda)":                  SurvivorSelectionDef(keep_best_offspring),
    "mu,lambda":                    SurvivorSelectionDef(keep_best_offspring),
    "keep_offspring":               SurvivorSelectionDef(keep_best_offspring),
    "keep_best_offspring":          SurvivorSelectionDef(keep_best_offspring),
}
# fmt: on

order_preserving_selections = {}


def create_survivor_selection(method: str, name: Optional[str] = None, rng: Optional[RNGLike] = None, **kwargs):
    """Create a survivor selection method by name.

    Parameters
    ----------
    method : str
        Key into :data:`surv_method_map`, or a null alias.
    name : str, optional
        Display name for the selection method.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Additional parameters forwarded to the selection function.

    Returns
    -------
    SurvivorSelectionFromLambda or NullSurvivorSelection
        The wrapped selection method.
    """

    if name is None:
        name = method

    if method in null_aliases:
        return NullSurvivorSelection(name=name, **kwargs)

    selection_fn_wrapper = surv_method_map[method.lower()]
    preserves_order = selection_fn_wrapper.preserves_order or (method.lower() in order_preserving_selections)
    return SurvivorSelectionFromLambda(
        selection_fn=selection_fn_wrapper, name=name, preserves_order=preserves_order, rng=rng, **kwargs
    )


def add_survivor_selection_entry(selection_fn: callable, selection_method_name: str, preserves_order: bool = False):
    """Register a new survivor selection method.

    Parameters
    ----------
    selection_fn : callable
        A function with the survivor selection signature.
    selection_method_name : str
        Name under which to register the method.  If it already exists,
        a warning is logged.
    preserves_order : bool, optional
        Whether the method preserves the order of individuals.
    """

    SurvivorSelectionFromLambda._validate_function(selection_fn)

    if selection_method_name in surv_method_map:
        logger.warning('Overwritten survivor selection method "%s".', selection_method_name)
    surv_method_map[selection_method_name] = selection_fn

    if preserves_order:
        order_preserving_selections.add(selection_method_name)

    logger.info('Added a new survivor selection method "%s".', selection_method_name)


def list_survivor_selection_methods() -> list[str]:
    """Return a list of all registered survivor selection method names.

    Returns
    -------
    list of str
    """

    return list(surv_method_map.keys())
