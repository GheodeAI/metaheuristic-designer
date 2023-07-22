from ..Initializers import UniformVectorInitializer
from ..Operators import OperatorInt, OperatorReal, OperatorBinary
from ..ParentSelection import ParentSelection
from ..SurvivorSelection import SurvivorSelection
from ..Encodings import TypeCastEncoding
from ..Algorithms import GA
from ..SearchMethods import GeneralSearch

def genetic_algorithm(objfunc, params):
    encoding_str = params["encoding"] if "encoding" in params else "bin"

    if encoding_str.lower() == "bin":
        alg = genetic_algorithm_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = genetic_algorithm_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = genetic_algorithm_real_vec(objfunc, params)
    else:
        raise ValueError(f"The encoding \"{encoding_str}\" does not exist, try \"real\", \"int\" or \"bin\"")
    
    return alg


def genetic_algorithm_bin_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    n_parents = params["n_parents"] if "n_parents" in params else 20
    cross_method = params["cross"] if "cross" in params else "Multipoint"
    pcross = params["pcross"] if "pcross" in params else 0.8
    pmut = params["pmut"] if "pmut" in params else 0.1
    mutstr = params["mutstr"] if "mutstr" in params else 1

    encoding = TypeCastEncoding(int, bool)

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, 0, 1, pop_size=pop_size, dtype=int, encoding=encoding)

    cross_op = OperatorBinary(cross_method)
    mutation_op = OperatorBinary("Flip", {"N":mutstr})

    parent_sel_op = ParentSelection("Best", {"amount": n_parents})
    selection_op = SurvivorSelection("KeepBest")
    
    search_strat = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross":pcross, "pmut":pmut})

    return GeneralSearch(objfunc, search_strat, params=params)


def genetic_algorithm_int_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    n_parents = params["n_parents"] if "n_parents" in params else 20
    cross_method = params["cross"] if "cross" in params else "Multipoint"
    pcross = params["pcross"] if "pcross" in params else 0.8
    pmut = params["pmut"] if "pmut" in params else 0.1
    mutstr = params["mutstr"] if "mutstr" in params else 1

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=int)

    cross_op = OperatorInt(cross_method)
    mutation_op = OperatorInt("MutRand", {"method":"Uniform", "Low":objfunc.low_lim, "Up":objfunc.up_lim, "N":mutstr})

    parent_sel_op = ParentSelection("Best", {"amount": n_parents})
    selection_op = SurvivorSelection("KeepBest")

    
    search_strat = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross":pcross, "pmut":pmut})

    return GeneralSearch(objfunc, search_strat, params=params)


def genetic_algorithm_real_vec(objfunc, params):
    """
    Instantiates a genetic algorithm to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    n_parents = params["n_parents"] if "n_parents" in params else 20
    cross_method = params["cross"] if "cross" in params else "Multipoint"
    pcross = params["pcross"] if "pcross" in params else 0.8
    pmut = params["pmut"] if "pmut" in params else 0.1
    mutstr = params["mutstr"] if "mutstr" in params else 1e-5 


    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float)

    cross_op = OperatorReal(cross_method)
    mutation_op = OperatorReal("RandNoise", {"method":"Gauss", "F":mutstr, "N":1})

    parent_sel_op = ParentSelection("Best", {"amount": n_parents})
    selection_op = SurvivorSelection("KeepBest")
    
    search_strat = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross":pcross, "pmut":pmut})

    return GeneralSearch(objfunc, search_strat, params=params)
