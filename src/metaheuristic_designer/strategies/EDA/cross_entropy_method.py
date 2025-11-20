from ...selection_methods import (
    ParentSelection,
    NullSurvivorSelection,
)
from ...operators import VectorOperator
from ..static_population import StaticPopulation


class CrossEntropyMethod(StaticPopulation):
    def __init__(self, initializer, params=None, name="CrossEntropyMethod"):
        if params is None:
            params = {}

        operator = VectorOperator(
            "RandSample",
            {"distrib": "Normal", "loc": "calculated", "scale": "calculated"},
        )
        n = params.get("n", initializer.pop_size)
        parent_sel = ParentSelection("Best", {"amount": n})
        survivor_sel = NullSurvivorSelection()

        super().__init__(
            initializer=initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )
