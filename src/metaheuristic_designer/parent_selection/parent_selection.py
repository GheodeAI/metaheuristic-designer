from __future__ import annotations
from dataclasses import dataclass, field
from enum import show_flag_values
import logging
from ..parent_selection_base import ParentSelectionFromLambda, NullParentSelection
from ..population import Population
from .parent_selection_functions import SelectionDist, prob_tournament, select_best, roulette, shuffle_population, uniform_selection, sus
from ..utils import null_aliases

logger = logging.getLogger(__name__)


@dataclass
class ParentSelectionDef:
    """ """

    selection_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, amount: int = None, random_state=None, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return self.selection_fn(population.fitness, amount, random_state, **modified_kwargs)


# fmt: off
parent_sel_map = {
    # Tournament
    "tournament":               ParentSelectionDef(prob_tournament, forced_params={"prob": 1.0}, params={"tournament_size": 3}),
    "tournament_selection":     ParentSelectionDef(prob_tournament, forced_params={"prob": 1.0}, params={"tournament_size": 3}),

    # Probabilistic Torunament
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

    # Roulette
    "roulette":                 ParentSelectionDef(roulette),
    "fitness_proportional":     ParentSelectionDef(roulette, forced_params={"method": SelectionDist.FIT_PROP}),
    "fit_prop":                 ParentSelectionDef(roulette, forced_params={"method": SelectionDist.FIT_PROP}),
    "proportional":             ParentSelectionDef(roulette, forced_params={"method": SelectionDist.FIT_PROP}),
    "pro":                      ParentSelectionDef(roulette, forced_params={"method": SelectionDist.FIT_PROP}),
    "std_roulette":             ParentSelectionDef(roulette, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "sigma_scaling":            ParentSelectionDef(roulette, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "rank_roulette":            ParentSelectionDef(roulette, forced_params={"method": SelectionDist.LIN_RANK}),
    "linear_rank":              ParentSelectionDef(roulette, forced_params={"method": SelectionDist.LIN_RANK}),
    "exp_rank_roulette":        ParentSelectionDef(roulette, forced_params={"method": SelectionDist.EXP_RANK}),
    "exponential_rank":         ParentSelectionDef(roulette, forced_params={"method": SelectionDist.EXP_RANK}),

    # Stochastic Universal Sampling (SUS)
    "sus":                      ParentSelectionDef(sus),
    "stochastic_universal_sampling": ParentSelectionDef(sus),
    "sus_fitness_proportional": ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_fit_prop":             ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_proportional":         ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_prop":                 ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_std":                  ParentSelectionDef(sus, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "sus_sigma":                ParentSelectionDef(sus, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "sus_rank":                 ParentSelectionDef(sus, forced_params={"method": SelectionDist.LIN_RANK}),
    "sus_exp":                  ParentSelectionDef(sus, forced_params={"method": SelectionDist.EXP_RANK}),
    "sus_exponential":          ParentSelectionDef(sus, forced_params={"method": SelectionDist.EXP_RANK}),
}


def create_parent_selection(method, name=None, amount=None, random_state=None, **kwargs):
    if name is None:
        name = method

    if method in null_aliases:
        return NullParentSelection(name=name, **kwargs)

    return ParentSelectionFromLambda(selection_fn=parent_sel_map[method.lower()], name=name, amount=amount, random_state=random_state, **kwargs)


def add_parent_selection_entry(selection_fn: callable, selection_method_name: str):
    """
    Adds an operator so it can be generated by a the operator factory.

    Parameters
    ----------
    operator_fn
        Callable that gets a population and a random_state to perturb the population. It is highly recommended to
        use one of the available wrappers OperatorVectorDef, OperatorRandomDef, ...
    operator_name
        Name to give the operator in the registry.
    operator_registry, optional
        Name of the registry to add this operator to. New names simply generate a new registry, by default "custom"
    """

    ParentSelectionFromLambda._validate_function(selection_fn)

    if selection_method_name in parent_sel_map:
        logger.warning('Overwritten parent selection method "%s".', selection_method_name)
    parent_sel_map[selection_method_name] = selection_fn

    logger.info('Added a new parent selection method "%s".', selection_method_name)

def list_parent_selection_methods():
    return list(parent_sel_map.keys())
