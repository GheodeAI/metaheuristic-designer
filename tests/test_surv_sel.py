import pytest
import numpy as np
from metaheuristic_designer import Individual
from metaheuristic_designer.selectionMethods import SurvivorSelection
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100

example_populaton = [Individual(None, None) for i in range(pop_size)]
example_offspring = [Individual(None, None) for i in range(pop_size)]
example_offspring_small = [Individual(None, None) for i in range(pop_size // 2)]
example_offspring_big = [Individual(None, None) for i in range(pop_size * 2)]

for idx, ind in enumerate(example_populaton):
    example_populaton[idx].fitness = 10

for idx, ind in enumerate(example_offspring):
    example_offspring[idx].fitness = 100

for idx, ind in enumerate(example_offspring_small):
    example_offspring_small[idx].fitness = 100

for idx, ind in enumerate(example_offspring_big):
    example_offspring_big[idx].fitness = 100


def test_error():
    with pytest.raises(ValueError):
        surv_selection = SurvivorSelection("not_a_method")


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
@pytest.mark.parametrize("keep_amount", [1, 5, 20])
def test_elitism(population, offspring, keep_amount):
    surv_selection = SurvivorSelection("Elitism", {"amount": keep_amount})
    survivors = surv_selection.select(population, offspring)
    assert len(population) >= len(survivors)
    assert set(population[:keep_amount]) == set(survivors[:keep_amount])
    pop_fit_avg = sum([i.fitness for i in population]) / len(population)
    surv_fit_avg = sum([i.fitness for i in survivors]) / len(survivors)
    assert pop_fit_avg < surv_fit_avg


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
@pytest.mark.parametrize("keep_amount", [1, 5, 20])
def test_condelitism(population, offspring, keep_amount):
    surv_selection = SurvivorSelection("CondElitism", {"amount": keep_amount})
    survivors = surv_selection.select(population, offspring)
    assert len(population) >= len(survivors)
    pop_fit_avg = sum([i.fitness for i in population]) / len(population)
    surv_fit_avg = sum([i.fitness for i in survivors]) / len(survivors)
    assert pop_fit_avg < surv_fit_avg


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
def test_one_to_one(population, offspring):
    surv_selection = SurvivorSelection("One-to-one")
    survivors = surv_selection.select(population, offspring)
    assert len(population) >= len(survivors)
    assert set(population[len(offspring) :]) == set(survivors[len(offspring) :])
    pop_fit_avg = sum([i.fitness for i in population]) / len(population)
    surv_fit_avg = sum([i.fitness for i in survivors]) / len(survivors)
    assert pop_fit_avg < surv_fit_avg


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
@pytest.mark.parametrize("p", [0, 0.1, 0.25, 0.5, 0.75, 1])
def test_prob_one_to_one(population, offspring, p):
    surv_selection = SurvivorSelection("Prob-One-to-one", {"p": p})
    survivors = surv_selection.select(population, offspring)
    assert len(population) >= len(survivors)
    assert set(population[len(offspring) :]) == set(survivors[len(offspring) :])
    pop_fit_avg = sum([i.fitness for i in population]) / len(population)
    surv_fit_avg = sum([i.fitness for i in survivors]) / len(survivors)
    assert pop_fit_avg <= surv_fit_avg


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
def test_generational(population, offspring):
    surv_selection = SurvivorSelection("Generational")
    survivors = surv_selection.select(population, offspring)
    assert survivors == offspring


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
def test_m_plus_n(population, offspring):
    surv_selection = SurvivorSelection("(m+n)")
    survivors = surv_selection.select(population, offspring)
    assert len(population) >= len(survivors)

    pop_fit_avg = sum([i.fitness for i in population]) / len(population)
    surv_fit_avg = sum([i.fitness for i in survivors]) / len(survivors)
    assert pop_fit_avg < surv_fit_avg


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
def test_m_comma_n(population, offspring):
    surv_selection = SurvivorSelection("(m,n)")
    survivors = surv_selection.select(population, offspring)
    assert len(population) >= len(survivors)

    for parent in population:
        assert parent not in survivors

    pop_fit_avg = sum([i.fitness for i in population]) / len(population)
    surv_fit_avg = sum([i.fitness for i in survivors]) / len(survivors)
    assert pop_fit_avg < surv_fit_avg


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize(
    "offspring", [example_offspring, example_offspring_small, example_offspring_big]
)
@pytest.mark.parametrize("fd", np.linspace(0, 1, 10))
@pytest.mark.parametrize("pd", np.linspace(0, 1, 10))
@pytest.mark.parametrize("attempts", [1, 3, 5, 10])
def test_cro_selection(population, offspring, fd, pd, attempts):
    surv_selection = SurvivorSelection(
        "CRO",
        {
            "Fd": fd,
            "Pd": pd,
            "attempts": attempts,
            "maxPopSize": len(example_populaton),
        },
    )
    survivors = surv_selection.select(population, offspring)
    assert len(population) >= len(survivors)

    pop_fit_avg = sum([i.fitness for i in population]) / len(population)
    surv_fit_avg = sum([i.fitness for i in survivors]) / len(survivors)
    if fd != 1 and pd != 1:
        assert pop_fit_avg < surv_fit_avg
