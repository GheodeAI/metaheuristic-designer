import pytest

from metaheuristic_designer import Individual
from metaheuristic_designer.selectionMethods import ParentSelection
import metaheuristic_designer as mhd

mhd.reset_seed(0)

pop_size = 100

example_populaton1 = [Individual(None, None) for i in range(pop_size)]
example_populaton2 = [Individual(None, None) for i in range(pop_size)]

for idx, ind in enumerate(example_populaton1):
    example_populaton1[idx].fitness = idx

for idx, ind in enumerate(example_populaton2):
    example_populaton2[idx].fitness = 1


def test_error():
    with pytest.raises(ValueError):
        surv_selection = ParentSelection("not_a_method")


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
def test_nothing(population):
    parent_sel = ParentSelection("Nothing")
    parents = parent_sel.select(population)

    assert parents == population


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
@pytest.mark.parametrize("amount", [1, 5, 20])
def test_best(population, amount):
    parent_sel = ParentSelection("Best", {"amount": amount})
    parents = parent_sel.select(population)
    id_list = [i.id for i in population]

    assert len(parents) == amount

    for indiv in parents:
        assert indiv.id in id_list

    fit_list = [i.fitness for i in population]
    parent_fit_list = [i.fitness for i in parents]
    assert max(fit_list) == max(parent_fit_list)


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
@pytest.mark.parametrize("amount", [1, 5, 20])
@pytest.mark.parametrize("prob", [0.01, 0.1, 0.2, 0.5])
@pytest.mark.parametrize("dummy_var", range(10))  # makes the test repeat 10 times
def test_tournament(population, amount, prob, dummy_var):
    parent_sel = ParentSelection("Tournament", {"amount": amount, "p": prob})
    parents = parent_sel.select(population)
    id_list = [i.id for i in population]

    for indiv in parents:
        assert indiv.id in id_list


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
@pytest.mark.parametrize("amount", [1, 5, 20])
def test_random(population, amount):
    parent_sel = ParentSelection("Random", {"amount": amount})
    parents = parent_sel.select(population)
    id_list = [i.id for i in population]

    for indiv in parents:
        assert indiv.id in id_list


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
@pytest.mark.parametrize("amount", [1, 5, 20])
@pytest.mark.parametrize(
    "method", ["FitnessProp", "SigmaScaling", "LinRank", "ExpRank"]
)
@pytest.mark.parametrize("f", [0, 0.5, 1, 1.5, 2])
@pytest.mark.parametrize("dummy_var", range(5))  # makes the test repeat 5 times
def test_roullete(population, amount, method, f, dummy_var):
    parent_sel = ParentSelection(
        "Roulette", {"amount": amount, "method": method, "f": f}
    )
    parents = parent_sel.select(population)
    id_list = [i.id for i in population]

    for indiv in parents:
        assert indiv.id in id_list


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
@pytest.mark.parametrize("amount", [1, 5, 20])
@pytest.mark.parametrize(
    "method", ["FitnessProp", "SigmaScaling", "LinRank", "ExpRank"]
)
@pytest.mark.parametrize("f", [0, 0.5, 1, 1.5, 2])
@pytest.mark.parametrize("dummy_var", range(5))  # makes the test repeat 5 times
def test_sus(population, amount, method, f, dummy_var):
    parent_sel = ParentSelection("Sus", {"amount": amount, "method": method, "f": f})
    parents = parent_sel.select(population)
    id_list = [i.id for i in population]

    for indiv in parents:
        assert indiv.id in id_list
