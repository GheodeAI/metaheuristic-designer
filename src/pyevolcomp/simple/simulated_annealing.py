from ..Initializers import UniformVectorInitializer
from ..Operators import OperatorInt, OperatorReal, OperatorBinary
from ..Algorithms import SA
from ..SearchMethods import GeneralSearch

def simulated_annealing(objfunc, params):
    encoding_str = params["encoding"] if "encoding" in params else "bin"

    if encoding_str.lower() == "bin":
        alg = simulated_annealing_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = simulated_annealing_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = simulated_annealing_real_vec(objfunc, params)
    else:
        raise ValueError(f"The encoding \"{encoding_str}\" does not exist, try \"real\", \"int\" or \"bin\"")
    
    return alg

def simulated_annealing_bin_vec(objfunc, params):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """
    n_iter = params["iter"] if "iter" in params else 100
    temp_init = params["temp_init"] if "temp_init" in params else 100
    alpha = params["alpha"] if "alpha" in params else 0.99
    mutstr = params["mutstr"] if "mutstr" in params else 1

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=bool)

    mutation_op = OperatorBinary("Flip", {"N":mutstr})

    search_strat = SA(pop_initializer, mutation_op, {"iter":n_iter, "temp_init":temp_init, "alpha":alpha})

    return GeneralSearch(objfunc, search_strat, params=params)


def simulated_annealing_int_vec(objfunc, params):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    n_iter = params["iter"] if "iter" in params else 100
    temp_init = params["temp_init"] if "temp_init" in params else 100
    alpha = params["alpha"] if "alpha" in params else 0.99
    mutstr = params["mutstr"] if "mutstr" in params else 1

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=int)

    mutation_op = OperatorInt("MutRand", {"method":"Uniform", "Low":objfunc.low_lim, "Up":objfunc.up_lim, "N":mutstr})

    search_strat = SA(pop_initializer, mutation_op, {"iter":n_iter, "temp_init":temp_init, "alpha":alpha})

    return GeneralSearch(objfunc, search_strat, params=params)


def simulated_annealing_real_vec(objfunc, params):
    """
    Instantiates a hill climbing algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    n_iter = params["iter"] if "iter" in params else 100
    temp_init = params["temp_init"] if "temp_init" in params else 100
    alpha = params["alpha"] if "alpha" in params else 0.99
    mutstr = params["mutstr"] if "mutstr" in params else 1e-5 

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, dtype=float)

    mutation_op = OperatorReal("RandNoise", {"method":"Gauss", "F":mutstr})
    
    search_strat = SA(pop_initializer, mutation_op, {"iter":n_iter, "temp_init":temp_init, "alpha":alpha})

    return GeneralSearch(objfunc, search_strat, params=params)