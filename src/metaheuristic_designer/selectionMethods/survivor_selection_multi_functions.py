from copy import copy
import random

def dominates(indiv1, indiv2):
    fit_vec1 = indiv1.fitness
    fit_vec2 = indiv2.fitness
    return np.all(fit_vec1 >= fit_vec2) and np.any(fit_vec1 > fit_vec2)


def dominates_equals(indiv1, indiv2):
    fit_vec1 = indiv1.fitness
    fit_vec2 = indiv2.fitness
    return np.all(fit_vec1 >= fit_vec2)


def non_dominated_ranking(indiv_list):
    if len(indiv_list) > 0:
        not_ranked_idx = []
        for idx_curr, indiv_curr in enumerate(indiv_list):
            is_dominant = True
            for idx_pop, indiv_pop in indiv_list:
                is_dominant = is_dominant and dominates_equals(indiv_curr, indiv_pop)
                if not is_dominant:
                    break
            
            if is_dominant:
                best_ranked.append(indiv_curr)
            else:
                not_ranked_idx.append(idx_curr)
        
        not_ranked = [indiv_list[i] for i in not_ranked_idx]
        return [best_ranked] + non_dominated_ranking(not_ranked)
    else:
        return []


def crowding_distance_selection(ranks):
    sorted_ranks = []
    k = len(ranks[0])
    for rank in ranks:
        if len(rank) > 1:
            fitness_mat = np.array([indiv.fitness for indiv in rank])
            n_obj = fitness_list[0].size

            distances = np.zeros((len(rank), n_obj))
            for idx_indiv, fit in enumerate(fitness_list):
                other_fit = np.delete(fitness_mat, idx_indiv, axis = 0)

                other_fit_low = other_fit.copy()
                other_fit_low[other_fit_low >= fit] = -np.inf
                lower = other_fit_low.max(axis=0)

                other_fit_up = other_fit.copy()
                other_fit_up[other_fit_up <= fit] = np.inf
                upper = other_fit_up.min(axis=0)

                dist = upper - lower
                dist[~np.isnan(dist)] = 0
                distances[idx_indiv, :] = dist

            indiv_order = argsort(distances.mean(axis=1))
            rank_sorted = [rank[i] for i in indiv_order]
            sorted_ranks.append(rank_sorted[:k])
        else:
            sorted_ranks.append(rank)
    
    return sum(sorted_ranks, [])


def non_dominated_sorting(indiv_list):
    ranks = non_dominated_ranking(indiv_list)
    return crowding_distance_selection(ranks)


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

