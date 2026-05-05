from __future__ import annotations
from ..algorithm import Algorithm
from ..initializers import UniformInitializer
from ..encodings import TypeCastEncoding, SigmoidEncoding
from ..strategies import PSO
from ..utils import check_random_state


def particle_swarm_binary(
    objfunc,
    population_size: int = 100,
    w=0.7,
    c1=1.5,
    c2=1.5,
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
        objfunc.vecsize,
        objfunc.lower_bound,
        objfunc.upper_bound,
        pop_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = PSO(
        initializer=pop_initializer,
        w=w,
        c1=c1,
        c2=c2,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def particle_swarm_discrete(
    objfunc,
    population_size: int = 100,
    w=0.7,
    c1=1.5,
    c2=1.5,
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
        objfunc.vecsize,
        objfunc.lower_bound,
        objfunc.upper_bound,
        pop_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = PSO(
        initializer=pop_initializer,
        w=w,
        c1=c1,
        c2=c2,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)


def particle_swarm_real(
    objfunc,
    population_size: int = 100,
    w=0.7,
    c1=1.5,
    c2=1.5,
    encoding=None,
    random_state=None,
    **kwargs,
):
    """
    Differential Evolution for real-coded vectors.
    """
    random_state = check_random_state(random_state)
    pop_initializer = UniformInitializer(
        objfunc.vecsize,
        objfunc.lower_bound,
        objfunc.upper_bound,
        pop_size=population_size,
        dtype=float,
        encoding=encoding,
        random_state=random_state,
    )
    search_strat = PSO(
        initializer=pop_initializer,
        w=w,
        c1=c1,
        c2=c2,
        random_state=random_state,
    )
    return Algorithm(objfunc, search_strat, **kwargs)
