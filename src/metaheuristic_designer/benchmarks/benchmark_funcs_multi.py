import numpy as np
from ..multiObjfunc import MultiObjectiveVectorFunc

class Shaffer1(MultiObjectiveVectorFunc):
    def __init__(self, name="Shaffer nº1"):
        super().__init__(
            n_objectives=2,
            vecsize=1,
            low_lim=-100,
            up_lim=100,
            modes="min",
            name=name
        )

    def objective(self, vector):
        return np.array([vector**2, (vector-2)**2]).squeeze()
