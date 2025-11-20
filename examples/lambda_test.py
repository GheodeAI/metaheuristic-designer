import numpy as np

from metaheuristic_designer import ConstraintHandlerFromLambda, ObjectiveFromLambda, InitializerFromLambda, OperatorFromLambda
from metaheuristic_designer.algorithms import GeneralAlgorithm
from metaheuristic_designer.strategies import *


def objective_fn(vector):
    return np.sum(np.abs(vector))

def constraint_repair_fn(vector):
    return np.clip(vector, -100, 100)

def initializer_fn():
    return np.random.uniform(-100, 100, 10)

def mutate_fn(population_matrix, **_):
    mask = np.random.uniform(0,1,size=population_matrix.shape) > 0.3
    population_matrix[mask] += np.random.normal(0, 1e-2, np.count_nonzero(mask))
    return population_matrix

def run_algorithm():
    params = {
        "stop_cond": "time_limit",
        "time_limit": 10.0,
        "cpu_time_limit": 10.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1e-30,
        "verbose": True,
        "v_timer": 0.5,
    }

    constriant_handler = ConstraintHandlerFromLambda(repair_solution_fn=constraint_repair_fn)
    objfunc = ObjectiveFromLambda(objective_fn, constraint_handler=constriant_handler, mode="min", name="Sphere (lambda)")
    pop_init = InitializerFromLambda(initializer_fn, pop_size=100)
    mutation_op = OperatorFromLambda(mutate_fn)

    search_strat = HillClimb(pop_init, mutation_op)

    alg = GeneralAlgorithm(objfunc, search_strat, params=params)

    population = alg.optimize()
    print(population.best_solution())
    alg.display_report()


if __name__ == "__main__":
    run_algorithm()
