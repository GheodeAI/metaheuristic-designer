"""
Parent selection registry and factory.
"""

from ..parent_selection_base import ParentSelection, ParentSelectionFromLambda, NullParentSelection
from .parent_selection import parent_sel_map, ParentSelectionDef, create_parent_selection, add_parent_selection_entry, list_parent_selection_methods
from .parent_selection_functions import (
    create_scaling_fn,
    select_best,
    prob_tournament,
    shuffle_population,
    uniform_selection,
    roulette,
    sus,
)

__all__ = [
    "NullParentSelection",
    "ParentSelection",
    "ParentSelectionDef",
    "ParentSelectionFromLambda",
    "list_parent_selection_methods",
    "add_parent_selection_entry",
    "create_parent_selection",
    "create_scaling_fn",
    "parent_sel_map",
    "select_best",
    "shuffle_population",
    "prob_tournament",
    "uniform_selection",
    "roulette",
    "sus",
]
