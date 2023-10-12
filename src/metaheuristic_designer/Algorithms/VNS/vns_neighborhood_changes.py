import enum
from enum import Enum


class NeighborhoodChange(Enum):
    SEQ = enum.auto()
    CYCLIC = enum.auto()
    PIPE = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in neigh_change_map:
            raise ValueError(f'Neighborhood change method "{str_input}" not defined')

        return neigh_change_map[str_input]


neigh_change_map = {
    "sequential": NeighborhoodChange.SEQ,
    "seq": NeighborhoodChange.SEQ,
    "cyclic": NeighborhoodChange.CYCLIC,
    "pipe": NeighborhoodChange.PIPE,
}


def next_neighborhood(new_indiv, prev_indiv, chosen_idx, method):
    """
    Methods from:
        Hansen, Pierre, et al. "Variable neighborhood search: basics and variants." EURO Journal on Computational Optimization 5.3 (2017): 423-454.
    """
    next_idx = None

    if method == NeighborhoodChange.SEQ:
        next_idx = sequential_n_change(new_indiv, prev_indiv, chosen_idx)
    elif method == NeighborhoodChange.CYCLIC:
        next_idx = cyclic_n_change(new_indiv, prev_indiv, chosen_idx)
    elif method == NeighborhoodChange.PIPE:
        next_idx = pipe_n_change(new_indiv, prev_indiv, chosen_idx)

    return next_idx


def sequential_n_change(new_indiv, prev_indiv, chosen_idx):
    next_idx = None

    if new_indiv.fitness > prev_indiv.fitness:
        next_idx = 0
    else:
        next_idx = chosen_idx + 1

    return next_idx


def cyclic_n_change(new_indiv, prev_indiv, chosen_idx):
    return chosen_idx + 1


def pipe_n_change(new_indiv, prev_indiv, chosen_idx):
    next_idx = None

    if new_indiv.fitness > prev_indiv.fitness:
        next_idx = chosen_idx
    else:
        next_idx = chosen_idx + 1

    return next_idx
