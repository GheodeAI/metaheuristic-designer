import numpy as np

from metaheuristic_designer import ConstraintHandlerFromLambda, ObjectiveFromLambda, InitializerFromLambda, OperatorFromLambda
from metaheuristic_designer.algorithms import StandardAlgorithm
from metaheuristic_designer.strategies import *
from metaheuristic_designer.operators import OperatorVectorDef


def objective_fn(vector):
    return np.sum(vector**2)

def constraint_repair_fn(vector):
    return np.clip(vector, -100, 100)

def initializer_fn(random_state=None):
    return random_state.uniform(-100, 100, 10)

def mutate_fn(population_matrix, _fitness_array=None, random_state=None, **_):
    mask = random_state.uniform(0,1,size=population_matrix.shape) > 0.3
    population_matrix[mask] += random_state.normal(0, 1e-2, np.count_nonzero(mask))
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

    constraint_handler = ConstraintHandlerFromLambda(repair_solution_fn=constraint_repair_fn)
    objfunc = ObjectiveFromLambda(objective_fn, constraint_handler=constraint_handler, mode="min", name="Sphere (lambda)")
    pop_init = InitializerFromLambda(initializer_fn, pop_size=100)
    mutation_op = OperatorFromLambda(OperatorVectorDef(mutate_fn))
    search_strategy = HillClimb(pop_init, mutation_op)
    alg = StandardAlgorithm(objfunc, search_strategy, **params)

    population = alg.optimize()
    print(population.best_solution())
    alg.display_report()


if __name__ == "__main__":
    run_algorithm()
