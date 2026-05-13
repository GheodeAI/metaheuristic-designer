from typing import Callable
import numpy as np

def simple_hillclimber(objective: Callable, dimension:int, lower_bound: np.ndarray, upper_bound: np.ndarray, max_evaluations:int, sigma: float, seed: int):
    # We seed a PCG64 random number generator
    random_state = np.random.Generator(np.random.PCG64(seed))

    # We generate and evaluate an uniformly random vector 
    curr_solution = random_state.uniform(lower_bound, upper_bound, size=dimension)
    prev_objective = objective(curr_solution)

    # Create a trace
    trace = [curr_solution]

    # We loop with the specified budget (we spent 1 function evaluation)
    for _ in range(max_evaluations-1):
        # Apply additive gaussian noise to our solution
        noise = random_state.normal(0, sigma, size=curr_solution.shape)
        new_solution = curr_solution + noise

        # Apply simple bound checking
        new_solution = np.clip(curr_solution, lower_bound, upper_bound)

        # Evaluate the solution
        new_objective = objective(new_solution)

        # If we got a better solution, we keep it. Discard it otherwise
        if new_objective >= prev_objective:
            curr_solution = new_solution
            prev_objective = new_objective

        # Add our solution to the trace
        trace.append(curr_solution)
        
    
    return curr_solution, trace