import time
import numpy as np
from ..objective_function import ObjectiveFunc, VectorObjectiveFunc

__all__ = [
    "MaxOnes",
    "DiophantineEq",
    "SleepTest",
    "MaxOnes",
    "Sphere",
    "HighCondElliptic",
    "BentCigar",
    "Discus",
    "Rosenbrock",
    "Ackley",
    "Weierstrass",
    "Griewank",
    "Rastrigin",
    "ModSchwefel",
    "Katsuura",
    "HappyCat",
    "HGBat",
    "ExpandedGriewankPlusRosenbrock",
    "ExpandedShafferF6",
    "SumPowell",
    "N4XinSheYang",
]

class MaxOnes(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="max", constraint_handler=None):
        super().__init__(constraint_handler=constraint_handler, vecsize=vecsize, low_lim=0, up_lim=1, mode=mode, name="Max ones")

    def objective(self, solution):
        return solution.sum()


class DiophantineEq(ObjectiveFunc):
    def __init__(self, vecsize, coeff, target, mode="min"):
        self.coeff = coeff
        self.target = target
        super().__init__(mode=mode, name="Diophantine equation")

    def objective(self, solution):
        return abs((solution * self.coeff).sum() - self.target)


class SleepTest(VectorObjectiveFunc):
    def __init__(self, vecsize, sleep_time=2, mode="min"):
        self.sleep_time = sleep_time
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Sphere function")

    def objective(self, solution):
        time.sleep(self.sleep_time)
        return _sphere(solution)


### Benchmark functions
class Sphere(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min", constraint_handler=None):
        super().__init__(constraint_handler=constraint_handler, vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Sphere function")

    def objective(self, solution):
        return _sphere(solution)


class HighCondElliptic(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-5.12, up_lim=5.12, name="High condition elliptic function")

    def objective(self, solution):
        return _high_cond_elipt_f(solution)


class BentCigar(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Bent Cigar function")

    def objective(self, solution):
        return _bent_cigar(solution)


class Discus(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-5.12, up_lim=5.12, name="Discus function")

    def objective(self, solution):
        return _discus(solution)


class Rosenbrock(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Rosenbrock function")

    def objective(self, solution):
        return _rosenbrock(solution)


class Ackley(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-5.12, up_lim=5.12, name="Ackley function")

    def objective(self, solution):
        return _ackley(solution)


class Weierstrass(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Weierstrass function")

    def objective(self, solution):
        return _weierstrass(solution)


class Griewank(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Griewank function")

    def objective(self, solution):
        return _griewank(solution)


class Rastrigin(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-5.12, up_lim=5.12, name="Rastrigin function")

    def objective(self, solution):
        return _rastrigin(solution)


class ModSchwefel(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Modified Schweafel function")

    def objective(self, solution):
        return _mod_schwefel(solution)


class Katsuura(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Katsuura function")

    def objective(self, solution):
        return _katsuura(solution)


class HappyCat(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-2, up_lim=2, name="Happy Cat function")

    def objective(self, solution):
        return _happy_cat(solution)


class HGBat(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-2, up_lim=2, name="HGBat function")

    def objective(self, solution):
        return _hgbat(solution)


class ExpandedGriewankPlusRosenbrock(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Expanded Griewank + Rosenbrock")

    def objective(self, solution):
        return _exp_griewank_plus_rosenbrock(solution)


class ExpandedShafferF6(VectorObjectiveFunc):
    def __init__(self, vecsize, mode="min"):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=-100, up_lim=100, name="Expanded Shaffer F6 function")

    def objective(self, solution):
        return _exp_shafferF6(solution)


class SumPowell(VectorObjectiveFunc):
    """
    Sum of Powell function
    """

    def __init__(self, vecsize, mode="min", lim_min=-1, lim_max=1):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=lim_min, up_lim=lim_max, name="Sum Powell")

    def objective(self, solution):
        return _sum_powell(solution)


class N4XinSheYang(VectorObjectiveFunc):
    """
    N4 Xin-She Yang function
    """

    def __init__(self, vecsize, mode="min", lim_min=-10, lim_max=10):
        super().__init__(vecsize=vecsize, mode=mode, low_lim=lim_min, up_lim=lim_max, name="N4 Xin-She Yang")

    def objective(self, solution):
        return _n4xinshe_yang(solution)


def _sphere(solution):
    return (solution**2).sum()


# @jit(nopython=True)
def _high_cond_elipt_f(vect):
    c = 1.0e6 ** ((np.arange(vect.shape[0]) / (vect.shape[0] - 1)))
    return np.sum(c * vect * vect)


# @jit(nopython=True)
def _bent_cigar(solution):
    return solution[0] ** 2 + 1e6 * (solution[1:] ** 2).sum()


# @jit(nopython=True)
def _discus(solution):
    return 1e6 * solution[0] ** 2 + (solution[1:] ** 2).sum()


# @jit(nopython=True)
def _rosenbrock(solution):
    term1 = solution[1:] - solution[:-1] ** 2
    term2 = 1 - solution[:-1]
    result = 100 * term1**2 + term2**2
    return result.sum()


# @jit(nopython=True)
def _ackley(solution):
    term1 = (solution**2).sum()
    term1 = -0.2 * np.sqrt(term1 / solution.size)
    term2 = (np.cos(2 * np.pi * solution)).sum() / solution.size
    return np.exp(1) - 20 * np.exp(term1) - np.exp(term2) + 20


# @jit(nopython=False)
def _weierstrass(solution, iter=20):
    return np.sum(np.array([0.5**k * np.cos(2 * np.pi * 3**k * (solution + 0.5)) for k in range(iter)]))


# @jit(nopython=True)
def _griewank(solution):
    term1 = (solution**2).sum()
    term2 = np.prod(np.cos(solution / np.sqrt(np.arange(1, solution.size + 1))))
    return 1 + term1 / 4000 - term2


# @jit(nopython=True)
def _rastrigin(solution, A=10):
    return A * len(solution) + (solution**2 - A * np.cos(2 * np.pi * solution)).sum()


# @jit(nopython=True)
def _mod_schwefel(solution):
    fit = 0
    for i in range(solution.size):
        z = solution[i] + 4.209687462275036e2
        if z > 500:
            fit = fit - (500 - z % 500) * np.sin((500 - z % 500) ** 0.5)
            tmp = (z - 500) / 100
            fit = fit + tmp * tmp / solution.size
        elif z < -500:
            fit = fit - (-500 - abs(z) % 500) * np.sin((500 - abs(z) % 500) ** 0.5)
            tmp = (z + 500) / 100
            fit = fit + tmp * tmp / solution.size
        else:
            fit = fit - z * np.sin(abs(z) ** 0.5)
    return fit + 4.189828872724338e2 * solution.size


# @jit(nopython=True)
def _katsuura(solution):
    A = 10 / solution.size**2

    temp_list = [
        1
        + (i + 1)
        * np.sum(
            (
                np.abs(2 ** (np.arange(1, 32 + 1)) * solution[i] - np.round(2 ** (np.arange(1, 32 + 1)) * solution[i]))
                * 2 ** (-np.arange(1, 32 + 1, dtype=float))
            )
            ** (10 / solution.size**1.2)
        )
        for i in range(solution.size)
    ]
    prod_val = np.prod(temp_list)
    return A * prod_val - A


# @jit(nopython=True)
def _happy_cat(solution):
    z = solution + 4.189828872724338e2
    r2 = (z * solution).sum()
    s = solution.sum()
    return np.abs(r2 - solution.size) ** 0.25 + (0.5 * r2 + s) / solution.size + 0.5


# @jit(nopython=True)
def _hgbat(solution):
    z = solution + 4.189828872724338e2
    r2 = (z * solution).sum()
    s = solution.sum()
    return np.abs((r2**2 - s**2)) ** 0.5 + (0.5 * r2 + s) / solution.size + 0.5


# @jit(nopython=True)
def _exp_griewank_plus_rosenbrock(solution):
    z = solution[:-1] + 4.189828872724338e2
    tmp1 = solution[:-1] ** 2 - solution[1:]
    tmp2 = z - 1
    tmp = 100 * tmp1**2 + tmp2**2
    grw = (tmp**2 / 4000 - np.cos(tmp) + 1).sum()

    term1 = solution[1:] - solution[:-1] ** 2
    term2 = 1 - solution[:-1]
    ros = (100 * term1**2 + term2**2).sum()

    return grw + ros**2 / 4000 - np.cos(ros) + 1


# @jit(nopython=True)
def _exp_shafferF6(solution):
    term1 = np.sin(np.sqrt(np.sum(solution[:-1] ** 2 + solution[1:] ** 2))) ** 2 - 0.5
    term2 = 1 + 0.001 * (solution[:-1] ** 2 + solution[1:] ** 2).sum()
    temp = 0.5 + term1 / term2

    term1 = np.sin(np.sqrt(np.sum((solution.size - 1) ** 2 + solution[0] ** 2))) ** 2 - 0.5
    term2 = 1 + 0.001 * ((solution.size - 1) ** 2 + solution[0] ** 2)

    return temp + 0.5 + term1 / term2


# @jit(nopython=True)
def _sum_powell(solution):
    return (np.abs(solution) ** np.arange(2, solution.shape[0] + 2)).sum()


# @jit(nopython=True)
def _n4xinshe_yang(solution):
    sum_1 = np.exp(-(solution**2).sum())
    sum_2 = np.exp(-(np.sin(np.sqrt(np.abs(solution))) ** 2).sum())
    return (np.sin(solution) ** 2 - sum_1).sum() * sum_2
