from copy import copy
import numpy as np
import random


def dominates(indiv1_fitness, indiv2_fitness):
    return np.all(indiv1_fitness >= indiv2_fitness) & np.any(indiv1_fitness > indiv2_fitness)


def dominates_equals(indiv1_fitness, indiv2_fitness):
    return np.all(indiv1_fitness >= indiv2_fitness)

def dominates_vec(indiv1_fitness, indiv2_fitness):
    is_geq = np.all(indiv1_fitness >= indiv2_fitness, axis=0)
    is_greater = np.any(indiv1_fitness > indiv2_fitness, axis=0)
    return is_geq & is_greater

def dominates_equals(indiv1_fitness, indiv2_fitness):
    return np.all(indiv1_fitness >= indiv2_fitness, axis=0)


def non_dominated_ranking(population_fitness):
    idx_list = list(range(population_fitness.shape[0]))
    if len(indiv_list) > 0:
        best_ranked = []
        not_ranked_idx = []
        for idx_curr, fitness_curr in enumerate(population_fitness):
            remaining_mask = np.ones(population_fitness.shape[0], dtype=bool)
            remaining_mask[idx_curr] = False
            remaining = population_fitness[remaining_mask]

            is_dominated = False
            for fitness in remaining_pop:
                is_dominated = is_dominated or dominates(fitness, fitness_curr)
                if is_dominated:
                    break
            
            if is_dominated:
                not_ranked_idx.append(idx_curr)
            else:
                best_ranked.append(indiv_curr)
        
        not_ranked = [indiv_list[i] for i in not_ranked_idx]
        return [best_ranked] + non_dominated_ranking(not_ranked)
    else:
        return []

def fast_non_dominated_ranking(population_fitness):
    S = []
    N = []
    ranks = np.zeros(population_fitness.shape[0])
    F = [[]]
    for p, fit_p in population_fitness:
        S_p = []
        N_p = 0
        for q, fit_q in population_fitness:
            if dominates(fit_p, fit_q):
                s_p.append(f_q_idx)
            elif np.any(fit_p != fit_q):
                n_p += 1
        if n_p == 0:
            ranks[f_p_idx] = 1
            F[0].append(p)
        S.append(S_p)
        n.append(n_p)
    
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p_idx, p in enumerate(fronts[i]):
            for q_idx, q in enumerate(s[i]):
                n[i] -= 1
                if n[i] == 0:
                    rank[q] = i+1 
                    Q.append(q)
        
        i += 1
        F[i] = Q
    
    return fronts



def crowding_distance_selection(ranks, amount):
    sorted_ranks = []
    k = amount
    for rank in ranks:
        if k < 1:
            break

        if len(rank) > 1:
            fitness_mat = np.array([indiv.fitness for indiv in rank])
            n_obj = fitness_mat.shape[1]

            distances = np.zeros((len(rank), n_obj))
            for idx_indiv, fit in enumerate(fitness_mat):
                other_fit = np.delete(fitness_mat, idx_indiv, axis = 0)

                other_fit_low = other_fit.copy()
                other_fit_low[other_fit_low >= fit] = -np.inf
                lower = other_fit_low.max(axis=0)

                other_fit_up = other_fit.copy()
                other_fit_up[other_fit_up <= fit] = np.inf
                upper = other_fit_up.min(axis=0)

                dist = upper - lower
                dist[np.isnan(dist)] = 0
                distances[idx_indiv, :] = dist

            indiv_order = np.argsort(distances.mean(axis=1))
            rank_sorted = [rank[i] for i in indiv_order]
            sorted_ranks.append(rank_sorted[:k])
        else:
            sorted_ranks.append(rank)
        
        k -= len(rank)
    
    return sum(sorted_ranks, [])


def non_dominated_sorting(popul, offspring, amount):
    indiv_list = popul + offspring
    ranks = non_dominated_ranking(indiv_list)
    return crowding_distance_selection(ranks, amount)


def argsort(seq):
    """
    Implementation of argsort for python-style lists.
    Source: https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

    Parameters
    ----------
    seq: Iterable
        Iterable for which we want to obtain the order of.

    Returns
    -------
    order: List
        The positions of the original elements of the list in order.
    """

    return sorted(range(len(seq)), key=seq.__getitem__)

