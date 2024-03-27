import math
from .GaussianUMDA import GaussianUMDA
from ...selectionMethods import ParentSelection, SurvivorSelection, SurvivorSelectionNull
from ...operators import OperatorReal
from ..StaticPopulation import StaticPopulation

class CrossEntropyMethod(StaticPopulation):
    def __init__(self, initializer, params={}, name="CrossEntropyMethod"):
        operator = OperatorReal("RandSample", {"distrib": "Normal", "loc": "calculated", "scale": "calculated"})
        n = params.get("n", math.ceil(initializer.pop_size/5))
        parent_sel = ParentSelection("Best", {"amount": n})
        survivor_sel=SurvivorSelectionNull()

        super().__init__(
            initializer=initializer,
            operator=operator,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name
        )