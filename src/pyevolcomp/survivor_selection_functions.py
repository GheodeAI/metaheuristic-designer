from copy import copy
import random

def one_to_one(popul, offspring):
    new_population = []
    for parent, child in zip(popul, offspring):
        if child.fitness > parent.fitness:
            new_population.append(child)
        else:
            new_population.append(parent)

    if len(offspring) < len(popul):
        n_leftover = len(offspring) - len(popul)
        new_population += popul[n_leftover:]

    return new_population


def elitism(popul, offspring, amount):
    selected_offspring = sorted(offspring, reverse=True, key=lambda x: x.fitness)[:len(popul) - amount]
    best_parents = sorted(popul, reverse=True, key=lambda x: x.fitness)[:amount]

    return best_parents + selected_offspring


def cond_elitism(popul, offspring, amount):
    best_parents = sorted(popul, reverse=True, key=lambda x: x.fitness)[:amount]
    new_offspring = sorted(offspring, reverse=True, key=lambda x: x.fitness)[:len(popul)]
    best_offspring = new_offspring[:amount]

    for idx, val in enumerate(best_parents):
        if val.fitness > best_offspring[0].fitness:
            new_offspring.pop()
            new_offspring = [val] + new_offspring

    return new_offspring


def lamb_plus_mu(popul, offspring):
    population = popul + offspring
    return sorted(population, reverse=True, key=lambda x: x.fitness)[:len(popul)]


def lamb_comma_mu(popul, offspring):
    return sorted(offspring, reverse=True, key=lambda x: x.fitness)[:len(popul)]


def argsort(seq):
    # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def _cro_set_larvae(population, offspring, attempts, maxpopsize):
    new_population = copy(population)
    for larva in offspring:
        attempts_left = attempts
        setted = False

        while attempts_left > 0 and not setted:
            idx = random.randrange(0, maxpopsize)

            if setted := (idx >= len(new_population)):
                new_population.append(larva)
            elif setted := (larva.fitness > new_population[idx].fitness):
                new_population[idx] = larva

            attempts_left -= 1
    
    return new_population


def _cro_depredation(population, Fd, Pd):
    amount = int(len(population)*Fd)

    fitness_values = [coral.fitness for coral in population]
    affected_corals = argsort(fitness_values)[:amount]

    alive_count = len(population)
    dead_list = [False] * len(population)

    for idx, val in enumerate(affected_corals):
        if alive_count <= 2:
            break
        
        dies = random.random() <= Pd
        dead_list[idx] = dies
        if dies:
            alive_count -= 1

    return [c for idx, c in enumerate(population) if not dead_list[idx]]


def cro_selection(popul, offspring, Fd, Pd, attempts, maxpopsize):
    setted_corals = _cro_set_larvae(popul, offspring, attempts, maxpopsize)
    reduced_population = _cro_depredation(setted_corals, Fd, Pd)
    return reduced_population
