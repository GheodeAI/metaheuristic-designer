from __future__ import annotations
from dataclasses import dataclass, field
from ..parent_selection import ParentSelectionFromLambda, NullParentSelection
from ..population import Population
from .parent_selection_functions import SelectionDist, prob_tournament, select_best, roulette, uniform_selection, sus
from ..utils import null_aliases


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

        return self.selection_fn(population.fitness, amount, random_state, **kwargs)


parent_sel_map = {
    "tournament": ParentSelectionDef(prob_tournament, forced_params={"prob": 1.0}, params={"tournament_size": 3}),
    "tournament_selection": ParentSelectionDef(prob_tournament, forced_params={"prob": 1.0}, params={"tournament_size": 3}),
    "probabilistic_tournament": ParentSelectionDef(prob_tournament, params={"tournament_size": 3, "prob": 0.5}),
    "best": ParentSelectionDef(select_best),
    "truncation": ParentSelectionDef(select_best),
    "select_best": ParentSelectionDef(select_best),
    "random": ParentSelectionDef(uniform_selection),
    "uniform": ParentSelectionDef(uniform_selection),
    "roulette": ParentSelectionDef(roulette),
    "fitness_proportional": ParentSelectionDef(roulette, forced_params={"method": SelectionDist.FIT_PROP}),
    "std_roulette": ParentSelectionDef(roulette, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "sigma_scaling": ParentSelectionDef(roulette, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "rank_roulette": ParentSelectionDef(roulette, forced_params={"method": SelectionDist.LIN_RANK}),
    "linear_rank": ParentSelectionDef(roulette, forced_params={"method": SelectionDist.LIN_RANK}),
    "exp_rank_roulette": ParentSelectionDef(roulette, forced_params={"method": SelectionDist.EXP_RANK}),
    "exponential_rank": ParentSelectionDef(roulette, forced_params={"method": SelectionDist.EXP_RANK}),
    "sus": ParentSelectionDef(sus),
    "stochastic_universal_sampling": ParentSelectionDef(sus),
    "sus_fitness_proportional": ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_fit_prop": ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_proportional": ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_prop": ParentSelectionDef(sus, forced_params={"method": SelectionDist.FIT_PROP}),
    "sus_std": ParentSelectionDef(sus, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "sus_sigma": ParentSelectionDef(sus, forced_params={"method": SelectionDist.SIGMA_SCALE}),
    "sus_rank": ParentSelectionDef(sus, forced_params={"method": SelectionDist.LIN_RANK}),
    "sus_exp": ParentSelectionDef(sus, forced_params={"method": SelectionDist.EXP_RANK}),
    "sus_exponential": ParentSelectionDef(sus, forced_params={"method": SelectionDist.EXP_RANK}),
}


def create_parent_selection(method, name=None, amount=None, random_state=None, **kwargs):
    if name is None:
        name = method

    if method in null_aliases:
        return NullParentSelection(name=name, **kwargs)

    return ParentSelectionFromLambda(selection_fn=parent_sel_map[method.lower()], name=name, amount=amount, random_state=random_state, **kwargs)
