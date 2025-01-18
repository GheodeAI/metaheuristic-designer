from ...selectionMethods import (
    ParentSelection,
    SurvivorSelectionNull,
)
from ...operators import OperatorVector
from ..StaticPopulation import StaticPopulation


class CrossEntropyMethod(StaticPopulation):
    def __init__(self, initializer, params=None, name="CrossEntropyMethod"):
        if params is None:
            params = {}
        
        operator = OperatorVector(
            "RandSample",
            {"distrib": "Normal", "loc": "calculated", "scale": "calculated"},
        )
        n = params.get("n", initializer.pop_size)
        parent_sel = ParentSelection("Best", {"amount": n})
        survivor_sel = SurvivorSelectionNull()

        super().__init__(
            initializer=initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )
