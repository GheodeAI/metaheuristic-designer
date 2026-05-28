import numpy as np

from metaheuristic_designer import (
    ConstraintHandlerFromLambda,
    ObjectiveFromLambda,
    InitializerFromLambda,
    OperatorFromLambda,
)
from metaheuristic_designer.algorithms import Algorithm
from metaheuristic_designer.strategies import HillClimb
from metaheuristic_designer.operators import OperatorFnDef


# ----------------------------------------------------------------------
# 1. Define the problem: minimize sum of squares (Sphere function)
# ----------------------------------------------------------------------
def objective_fn(vector):
    """Return the raw objective value (sum of squares)."""
    return np.sum(vector**2)


# ----------------------------------------------------------------------
# 2. Constraint handler: keep every variable inside [-100, 100]
# ----------------------------------------------------------------------
def constraint_repair_fn(vector):
    """Clip each component to the feasible region."""
    return np.clip(vector, -100, 100)


# ----------------------------------------------------------------------
# 3. Initializer: generate a random 10‑dimensional vector in [-100, 100]
# ----------------------------------------------------------------------
def initializer_fn(random_state=None):
    """Return a single random candidate solution."""
    return random_state.uniform(-100, 100, 10)


# ----------------------------------------------------------------------
# 4. Mutation operator: add Gaussian noise to ~70% of the components
# ----------------------------------------------------------------------
@OperatorFnDef
def mutate_fn(population_matrix, fitness_array=None, random_state=None, **_):
    """
    Perturb each row (solution) by adding a small Gaussian noise
    to a random subset of its components.
    """
    # Boolean mask: True where we do NOT modify (the component stays unchanged)
    mask = random_state.uniform(0, 1, size=population_matrix.shape) > 0.3
    # Number of components that will be changed
    num_changed = np.count_nonzero(~mask)
    # Add noise to those components
    population_matrix[~mask] += random_state.normal(0, 1e-2, num_changed)
    return population_matrix


# ----------------------------------------------------------------------
# 5. Assemble the algorithm and run
# ----------------------------------------------------------------------
def run_algorithm():
    print("Building components from lambdas...")

    # Wrap the repair function into a constraint handler
    constraint_handler = ConstraintHandlerFromLambda(repair_solution_fn=constraint_repair_fn)

    # Wrap the objective function (minimisation mode)
    objfunc = ObjectiveFromLambda(
        objective_fn,
        constraint_handler=constraint_handler,
        mode="min",
        name="Sphere (lambda)",
    )

    # Wrap the initializer
    pop_init = InitializerFromLambda(initializer_fn, vecsize=10, pop_size=100)

    # Wrap the mutation operator (note: OperatorVectorDef can be used as decorator too)
    mutation_op = OperatorFromLambda(mutate_fn)

    # Build a hill‑climbing strategy with the components
    search_strategy = HillClimb(pop_init, mutation_op)

    # Configure runtime parameters
    params = {
        "stop_cond": "max_iterations",
        "max_iterations": 5e4,  # run for 10 seconds
        "reporter": "tqdm",  # nice progress bar
    }

    alg = Algorithm(objfunc, search_strategy, **params)

    print("Starting optimisation...")
    population = alg.optimize()

    # Retrieve the best result (decoded solution + objective)
    best_solution, best_objective = population.best_solution()
    print()
    print("Optimization finished.")
    print(f"Best objective (Sphere) = {best_objective:.6f}")
    print("Best solution vector:")
    print(np.array2string(np.asarray(best_solution), precision=4, suppress_small=True))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_algorithm()
