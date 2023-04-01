import pytest

from pyevolcomp import Individual, ParentSelection

pop_size = 100

example_populaton = [Individual(None, None) for i in range(pop_size)]

for idx, ind in enumerate(example_populaton):
    example_populaton[idx].fitness = idx


@pytest.mark.parametrize("population", [example_populaton])
def test_nothing(population):
    parent_sel = ParentSelection("Nothing")
    parents, idxs = parent_sel.select(population)

    assert parents == population
    assert idxs == range(len(parents))


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize("amount", [1, 5, 20])
def test_best(population, amount):
    parent_sel = ParentSelection("Best", {"amount": amount})
    parents, idxs = parent_sel.select(population)

    assert len(parents) == amount
    for i in idxs:
        assert i < len(population)
    fit_list = [i.fitness for i in population]
    assert fit_list.index(max(fit_list)) in idxs


@pytest.mark.parametrize("population", [example_populaton])
@pytest.mark.parametrize("amount", [1, 5, 20])
@pytest.mark.parametrize("prob", [0.01, 0.1, 0.2, 0.5])
@pytest.mark.parametrize("dummy_var", range(10)) # makes the test repeat 10 times
def test_tournament(population, amount, prob, dummy_var):
    parent_sel = ParentSelection("Tournament", {"amount": amount, "p": prob})
    parents, idxs = parent_sel.select(population)

    for i in idxs:
        assert i < len(population)
