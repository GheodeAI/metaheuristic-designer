import pytest

from pyevolcomp import Individual, ParentSelection

pop_size = 100

example_populaton1 = [Individual(None, None) for i in range(pop_size)]
example_populaton2 = [Individual(None, None) for i in range(pop_size)]

for idx, ind in enumerate(example_populaton1):
    example_populaton1[idx].fitness = idx

for idx, ind in enumerate(example_populaton2):
    example_populaton2[idx].fitness = 1


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
def test_nothing(population):
    parent_sel = ParentSelection("Nothing")
    parents, idxs = parent_sel.select(population)

    assert parents == population
    assert idxs == range(len(parents))


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
@pytest.mark.parametrize("amount", [1, 5, 20])
def test_best(population, amount):
    parent_sel = ParentSelection("Best", {"amount": amount})
    parents, idxs = parent_sel.select(population)

    assert len(parents) == amount
    for i in idxs:
        assert i < len(population)
    fit_list = [i.fitness for i in population]
    parent_fit_list = [i.fitness for i in parents]
    assert max(fit_list) == max(parent_fit_list)


@pytest.mark.parametrize("population", [example_populaton1, example_populaton2])
@pytest.mark.parametrize("amount", [1, 5, 20])
@pytest.mark.parametrize("prob", [0.01, 0.1, 0.2, 0.5])
@pytest.mark.parametrize("dummy_var", range(10)) # makes the test repeat 10 times
def test_tournament(population, amount, prob, dummy_var):
    parent_sel = ParentSelection("Tournament", {"amount": amount, "p": prob})
    parents, idxs = parent_sel.select(population)

    for i in idxs:
        assert i < len(population)
