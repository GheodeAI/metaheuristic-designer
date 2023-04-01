import sys
sys.path.append("../src")

import numpy as np
import random
from numba import jit
from pyevolcomp import ObjectiveFunc


class MaxOnes(ObjectiveFunc):
    def __init__(self, size, opt="max"):
        self.size = size
        super().__init__(self.size, opt, "Max ones")

    def objective(self, solution):
        return solution.sum()
    
    def random_solution(self):
        return (np.random.random(self.size) < 0.5).astype(np.int32)
    
    def repair_solution(self, solution):
        return (solution >= 0.5).astype(np.int32)

class DiophantineEq(ObjectiveFunc):
    def __init__(self, size, coeff, target, opt="min"):
        self.size = size
        self.coeff = coeff
        self.target = target
        super().__init__(self.size, opt, "Diophantine equation")
    
    def objective(self, solution):
        return abs((solution*self.coeff).sum() - self.target)
    
    def random_solution(self):
        return (np.random.randint(-100, 100, size=self.size)).astype(np.int32)
    
    def repair_solution(self, solution):
        return solution.astype(np.int32)

class MaxOnesReal(ObjectiveFunc):
    def __init__(self, size, opt="max"):
        self.size = size
        super().__init__(self.size, opt, "Max ones")

    def objective(self, solution):
        return solution.sum()
    
    def random_solution(self):
        return np.random.random(self.size)
    
    def repair_solution(self, solution):
        return np.clip(solution.copy(), 0, 1)

### Benchmark functions
class Sphere(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Sphere function")

    def objective(self, solution):
        return sphere(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class HighCondElliptic(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "High condition elliptic function")

    def objective(self, solution):
        return high_cond_elipt_f(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def repair_solution(self, solution):
        return np.clip(solution, -5.12, 5.12)

class BentCigar(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Bent Cigar function")

    def objective(self, solution):
        return bent_cigar(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class Discus(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Discus function")

    def objective(self, solution):
        return discus(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def repair_solution(self, solution):
        return np.clip(solution, -5.12, 5.12)

class Rosenbrock(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Rosenbrock function")

    def objective(self, solution):
        return rosenbrock(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class Ackley(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Ackley function")

    def objective(self, solution):
        return ackley(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def repair_solution(self, solution):
        return np.clip(solution, -5.12, 5.12)

class Weierstrass(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Weierstrass function")

    def objective(self, solution):
        return weierstrass(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class Griewank(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Griewank function")

    def objective(self, solution):
        return griewank(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class Rastrigin(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Rastrigin function")

    def objective(self, solution):
        return rastrigin(solution)
    
    def random_solution(self):
        return 10.24*np.random.random(self.size)-5.12
    
    def repair_solution(self, solution):
        return np.clip(solution, -5.12, 5.12)

class ModSchwefel(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Modified Schweafel function")

    def objective(self, solution):
        return mod_schwefel(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class Katsuura(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Katsuura function")

    def objective(self, solution):
        return katsuura(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class HappyCat(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Happy Cat function")

    def objective(self, solution):
        return happy_cat(solution)
    
    def random_solution(self):
        return 4*np.random.random(self.size)-2
    
    def repair_solution(self, solution):
        return np.clip(solution, -2, 2)

class HGBat(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "HGBat function")

    def objective(self, solution):
        return hgbat(solution)
    
    def random_solution(self):
        return 4*np.random.random(self.size)-2
    
    def repair_solution(self, solution):
        return np.clip(solution, -2, 2)

class ExpandedGriewankPlusRosenbrock(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Expanded Griewank + Rosenbrock")

    def objective(self, solution):
        return exp_griewank_plus_rosenbrock(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)

class ExpandedShafferF6(ObjectiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt, "Expanded Shaffer F6 function")

    def objective(self, solution):
        return exp_shafferF6(solution)
    
    def random_solution(self):
        return 200*np.random.random(self.size)-100
    
    def repair_solution(self, solution):
        return np.clip(solution, -100, 100)


class SumPowell(ObjectiveFunc):
    """
    Sum of Powell function
    """
    def __init__(self, size, opt="min", lim_min=-1, lim_max=1):
        self.size = size
        self.lim_min = lim_min
        self.lim_max = lim_max
        super().__init__(self.size, opt, "Sum Powell")

    def objective(self, solution):
        return sum_powell(solution)
    
    def random_solution(self):
        return np.random.random(self.size) * (self.lim_max - self.lim_min) - self.lim_min

    def repair_solution(self, solution, parent=None):
        # bounce back method
        if parent:
            mask_inf = (solution < self.lim_min) 
            mask_sup = (solution > self.lim_max)
            solution[mask_inf] = parent[mask_inf] + np.random.random() * (self.lim_min - parent[mask_inf])
            solution[mask_sup] = parent[mask_sup] + np.random.random() * (parent[mask_sup] - self.lim_max)
        # random in range
        else:
            mask = (solution < self.lim_min) | (solution > self.lim_max)
            solution[mask] = np.random.random(len(mask[mask==True])) * (self.lim_max - self.lim_min) - self.lim_min
        return solution

class N4XinSheYang(ObjectiveFunc):
    """
    N4 Xin-She Yang function
    """
    def __init__(self, size, opt="min", lim_min=-10, lim_max=10):
        self.size = size
        self.lim_min = lim_min
        self.lim_max = lim_max
        super().__init__(self.size, opt, "N4 Xin-She Yang")

    def objective(self, solution):
        return n4xinshe_yang(solution)
    
    def random_solution(self, lim_min=None, lim_max=None):
        return np.random.random(self.size) * (self.lim_max - self.lim_min) - self.lim_min
    
    def repair_solution(self, solution, parent=None):
        # bounce back method
        if parent:
            mask_inf = (solution < self.lim_min) 
            mask_sup = (solution > self.lim_max)
            solution[mask_inf] = parent[mask_inf] + np.random.random() * (self.lim_min - parent[mask_inf])
            solution[mask_sup] = parent[mask_sup] + np.random.random() * (parent[mask_sup] - self.lim_max)
        # random in range
        else:
            mask = (solution < self.lim_min) | (solution > self.lim_max)
            solution[mask] = np.random.random(len(mask[mask==True])) * (self.lim_max - self.lim_min) - self.lim_min
        return solution

@jit(nopython=True)
def sphere(solution):
    return (solution**2).sum()

@jit(nopython=True)
def high_cond_elipt_f(vect):
    c = 1.0e6**((np.arange(vect.shape[0])/(vect.shape[0]-1)))
    return np.sum(c*vect*vect)

@jit(nopython=True)
def bent_cigar(solution):
    return solution[0]**2 + 1e6*(solution[1:]**2).sum()

@jit(nopython=True)
def discus(solution):
    return 1e6*solution[0]**2 + (solution[1:]**2).sum()

@jit(nopython=True)
def rosenbrock(solution):
    term1 = solution[1:] - solution[:-1]**2
    term2 = 1 - solution[:-1]
    result = 100*term1**2 + term2**2
    return result.sum()

@jit(nopython=True)
def ackley(solution):
    term1 = (solution**2).sum()
    term1 = -0.2 * np.sqrt(term1/solution.size)
    term2 = (np.cos(2*np.pi*solution)).sum()/solution.size
    return np.exp(1) - 20 * np.exp(term1) - np.exp(term2) + 20

#@jit(nopython=False)
def weierstrass(solution, iter=20):
    return np.sum(np.array([0.5**k * np.cos(2*np.pi*3**k*(solution+0.5)) for k in range(iter)]))

#@jit(nopython=True)
def griewank(solution):
    term1 = (solution**2).sum()
    term2 = np.prod(np.cos(solution/np.sqrt(np.arange(1, solution.size+1))))
    return 1 + term1/4000 - term2

@jit(nopython=True)
def rastrigin(solution, A=10):
    return (A * len(solution) + (solution**2 - A*np.cos(2*np.pi*solution)).sum())

@jit(nopython=True)
def mod_schwefel(solution):
    fit = 0
    for i in range(solution.size):
        z = solution[i] + 4.209687462275036e2
        if z > 500:
            fit = fit - (500 - z % 500) * np.sin((500 - z%500)** 0.5)
            tmp = (z - 500) / 100
            fit = fit + tmp * tmp / solution.size
        elif z < -500:
            fit = fit - (-500 - abs(z) % 500) * np.sin((500 - abs(z)%500)** 0.5)
            tmp = (z + 500) / 100
            fit = fit + tmp * tmp / solution.size
        else:
            fit = fit - z * np.sin(abs( z )**0.5)
    return fit + 4.189828872724338e2 * solution.size

#@jit(nopython=True)
def katsuura(solution):
    A = 10/solution.size**2

    temp_list = [1 + (i+1)*np.sum((np.abs(2**(np.arange(1, 32+1))*solution[i]-np.round(2**(np.arange(1, 32+1))*solution[i]))*2**(-np.arange(1, 32+1, dtype=float)))**(10/solution.size**1.2)) for i in range(solution.size)]
    prod_val = np.prod(temp_list) 
    return  A * prod_val - A

@jit(nopython=True)
def happy_cat(solution):
    z = solution+4.189828872724338e2
    r2 = (z * solution).sum()
    s = solution.sum()
    return np.abs(r2-solution.size)**0.25 + (0.5*r2+s)/solution.size + 0.5

@jit(nopython=True)
def hgbat(solution):
    z = solution+4.189828872724338e2
    r2 = (z * solution).sum()
    s = solution.sum()
    return np.abs((r2**2-s**2))**0.5 + (0.5*r2 + s)/solution.size + 0.5

@jit(nopython=True)
def exp_griewank_plus_rosenbrock(solution):
    z = solution[:-1]+4.189828872724338e2
    tmp1 = solution[:-1]**2-solution[1:]
    tmp2 = z - 1
    tmp = 100*tmp1**2 + tmp2**2
    grw = (tmp**2/4000 - np.cos(tmp) + 1).sum()

    term1 = solution[1:] - solution[:-1]**2
    term2 = 1 - solution[:-1]
    ros = (100*term1**2 + term2**2).sum()
    
    return grw + ros**2/4000 - np.cos(ros) + 1

#@jit(nopython=True)
def exp_shafferF6(solution):
    term1 = np.sin(np.sqrt(np.sum(solution[:-1]**2 + solution[1:]**2)))**2 - 0.5
    term2 = 1 + 0.001*(solution[:-1]**2 + solution[1:]**2).sum()
    temp = 0.5 + term1/term2

    term1 = np.sin(np.sqrt(np.sum((solution.size - 1)**2 + solution[0]**2)))**2 - 0.5
    term2 = 1 + 0.001*((solution.size - 1)**2 + solution[0]**2)

    return temp + 0.5 + term1/term2

@jit(nopython=True)
def sum_powell(solution):
    return (np.abs(solution)**np.arange(2,solution.shape[0]+2)).sum()

@jit(nopython=True)
def n4xinshe_yang(solution):
    sum_1 = np.exp(-(solution**2).sum())
    sum_2 = np.exp(-(np.sin(np.sqrt(np.abs(solution)))**2).sum())
    return (np.sin(solution)**2 - sum_1).sum() * sum_2