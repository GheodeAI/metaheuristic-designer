from __future__ import annotations
from ..Initializers import UniformVectorInitializer
from ..Operators import OperatorInt, OperatorReal, OperatorBinary
from ..ParentSelection import ParentSelection
from ..SurvivorSelection import SurvivorSelection
from ..Algorithms import ES
from ..SearchMethods import GeneralSearch

def evolution_strategy(objfunc: ObjectiveVectorFunc, params: dict) -> Search:
    """
    Instantiates a evolution strategy to optimize the given objective function.
    """

    encoding_str = params["encoding"] if "encoding" in params else "bin"

    if encoding_str.lower() == "bin":
        alg = _evolution_strategy_bin_vec(objfunc, params)
    elif encoding_str.lower() == "int":
        alg = _evolution_strategy_int_vec(objfunc, params)
    elif encoding_str.lower() == "real":
        alg = _evolution_strategy_real_vec(objfunc, params)
    else:
        raise ValueError(f"The encoding \"{encoding_str}\" does not exist, try \"real\", \"int\" or \"bin\"")
    
    return alg


def _evolution_strategy_bin_vec(objfunc, params):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept binary coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    offspring_size = params["offspring_size"] if "offspring_size" in params else 150
    n_parents = params["n_parents"] if "n_parents" in params else 100
    mutstr = params["mutstr"] if "mutstr" in params else 1

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=bool)

    cross_op = OperatorBinary("Nothing")
    mutation_op = OperatorBinary("Flip", {"N":mutstr})

    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")

    search_strat = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize":offspring_size})

    return GeneralSearch(objfunc, search_strat, params=params)


def _evolution_strategy_int_vec(objfunc, params):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept integer coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    offspring_size = params["offspring_size"] if "offspring_size" in params else 150
    n_parents = params["n_parents"] if "n_parents" in params else 100
    mutstr = params["mutstr"] if "mutstr" in params else 1

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=int)

    cross_op = OperatorInt("Nothing")
    mutation_op = OperatorInt("MutRand", {"method":"Uniform", "Low":objfunc.low_lim, "Up":objfunc.up_lim, "N":mutstr})

    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")

    search_strat = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize":offspring_size})

    return GeneralSearch(objfunc, search_strat, params=params)


def _evolution_strategy_real_vec(objfunc, params):
    """
    Instantiates a evolution strategy to optimize the given objective function.
    This objective function should accept real coded vectors.
    """

    pop_size = params["pop_size"] if "pop_size" in params else 100
    offspring_size = params["offspring_size"] if "offspring_size" in params else 150
    n_parents = params["n_parents"] if "n_parents" in params else 100
    mutstr = params["mutstr"] if "mutstr" in params else 1e-5

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, dtype=float)

    cross_op = OperatorReal("Nothing")
    mutation_op = OperatorReal("RandNoise", {"method":"Gauss", "F":mutstr})
    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")
    
    search_strat = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize":offspring_size})

    return GeneralSearch(objfunc, search_strat, params=params)
