import numpy as np
from ..multiObjfunc import MultiObjectiveVectorFunc


class FonsecaFleming(MultiObjectiveVectorFunc):
    def __init__(self, name="Fonseca Fleming function"):
        super().__init__(n_objectives=2, vecsize=3, low_lim=-4, up_lim=4, modes="min", name=name)

    def objective(self, vector):
        n = vector.size
        return np.array(
            [
                1 - np.exp(-(vector - (1 / np.sqrt(np.arange(n)+1)) ** 2).sum()),
                1 - np.exp(-(vector + (1 / np.sqrt(np.arange(n)+1)) ** 2).sum()),
            ]
        ).squeeze()


class Kursawe(MultiObjectiveVectorFunc):
    def __init__(self, name="Kursawe function"):
        super().__init__(n_objectives=2, vecsize=3, low_lim=-5, up_lim=5, modes="min", name=name)

    def objective(self, vector):
        n = vector.size
        return np.array(
            [
                np.sum((-10 * np.exp(-0.2 * np.sqrt(vector[[0, 1]] ** 2 + vector[[1, 2]] ** 2)))),
                np.sum(np.abs(vector) ** 0.8 + 5 * np.sin(vector**3)),
            ]
        ).squeeze()


class Shaffer1(MultiObjectiveVectorFunc):
    def __init__(self, name="Shaffer nº1"):
        super().__init__(n_objectives=2, vecsize=1, low_lim=-10, up_lim=10, modes="min", name=name)

    def objective(self, vector):
        return np.array([vector**2, (vector - 2) ** 2]).squeeze()
