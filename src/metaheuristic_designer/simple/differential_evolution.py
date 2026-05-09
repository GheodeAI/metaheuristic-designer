from __future__ import annotations
from ..algorithm import Algorithm
from ..initializers import UniformInitializer
from ..encodings import TypeCastEncoding, SigmoidEncoding
from ..strategies import DE
from ..utils import check_random_state


def differential_evolution_binary(
    objfunc,
    population_size: int = 100,
    F: float = 0.8,
    Cr: float = 0.9,
    de_operator_name: str = "de/rand/1",
    encoding=None,
    random_state=None,
    **kwargs,
):
    """
    Differential Evolution for binary-coded vectors.
    """
    random_state = check_random_state(random_state)
    if encoding is None:
        encoding = SigmoidEncoding(as_probability=False, threshold=0.5)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = DE(
        initializer=pop_initializer,
        de_operator_name=de_operator_name,
        F=F,
        Cr=Cr,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def differential_evolution_discrete(
    objfunc,
    population_size: int = 100,
    F: float = 0.8,
    Cr: float = 0.9,
    de_operator_name: str = "de/rand/1",
    encoding=None,
    random_state=None,
    **kwargs,
):
    """
    Differential Evolution for integer-coded vectors.
    """
    random_state = check_random_state(random_state)
    if encoding is None:
        encoding = TypeCastEncoding(float, int)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = DE(
        initializer=pop_initializer,
        de_operator_name=de_operator_name,
        F=F,
        Cr=Cr,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def differential_evolution_real(
    objfunc,
    population_size: int = 100,
    F: float = 0.8,
    Cr: float = 0.9,
    de_operator_name: str = "de/rand/1",
    encoding=None,
    random_state=None,
    **kwargs,
):
    """
    Differential Evolution for real-coded vectors.
    """
    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.dimension,
        objfunc.lower_bound,
        objfunc.upper_bound,
        population_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = DE(
        initializer=pop_initializer,
        de_operator_name=de_operator_name,
        F=F,
        Cr=Cr,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)
