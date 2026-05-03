# test_checkpoint.py
import pytest
import numpy as np
import os

from conftest import (
    rng,
    dummy_objfunc,
    simple_encoding,
)

from metaheuristic_designer.algorithm import Algorithm
from metaheuristic_designer.search_strategy import SearchStrategy
from metaheuristic_designer.checkpointer import Checkpointer
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.operators.factories.mutation import create_mutation_operator
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.benchmarks.benchmark_funcs import SleepTest
from metaheuristic_designer.reporters.silent_reporter import SilentReporter


# ---------------------------------------------------------------
# Checkpoint split vs. continuous run – both must finish identically
# ---------------------------------------------------------------
def test_checkpoint_split_vs_continuous(tmp_path, rng, dummy_objfunc):
    checkpoint = tmp_path / "split.pkl"

    # ---- continuous run (10 generations) ----
    rng_a = np.random.default_rng(42)
    init_a = UniformInitializer(2, -10, 10, pop_size=10, random_state=rng_a)
    mut_a = create_mutation_operator("gaussian_mutation", random_state=rng_a, N=1, F=0.1)
    surv_a = create_survivor_selection("generational", random_state=rng_a)
    strat_a = SearchStrategy(init_a, operator=mut_a, survivor_sel=surv_a, name="cont")
    algo_a = Algorithm(dummy_objfunc, strat_a, stop_cond="ngen", ngen=10, reporter="silent")
    pop_a = algo_a.optimize()
    best_a = pop_a.best_solution(problem_space=True)[1]

    # ---- split run: 5 generations, checkpoint, then 5 more ----
    rng_b = np.random.default_rng(42)
    init_b = UniformInitializer(2, -10, 10, pop_size=10, random_state=rng_b)
    mut_b = create_mutation_operator("gaussian_mutation", random_state=rng_b, N=1, F=0.1)
    surv_b = create_survivor_selection("generational", random_state=rng_b)
    strat_b = SearchStrategy(init_b, operator=mut_b, survivor_sel=surv_b, name="split")
    algo_b = Algorithm(
        dummy_objfunc,
        strat_b,
        checkpoint_file=str(checkpoint),
        checkpoint_iteration_frequency=5,   # save at gen 5
        stop_cond="ngen",
        ngen=5,
        reporter="silent",
    )
    algo_b.optimize()
    assert checkpoint.exists()

    # resume
    ckp = Checkpointer(str(checkpoint))
    algo_c = ckp.load(str(checkpoint), reporter="silent")
    algo_c.stopping_condition.max_iterations = 10
    pop_c = algo_c.optimize()
    best_c = pop_c.best_solution(problem_space=True)[1]

    # results must match
    assert algo_c.stopping_condition.iterations == 10
    assert best_c == pytest.approx(best_a)


# ---------------------------------------------------------------
# Iteration‑based checkpoint frequency
# ---------------------------------------------------------------
def test_checkpoint_iteration_frequency(tmp_path, rng, dummy_objfunc):
    checkpoint = tmp_path / "freq.pkl"

    rng_a = np.random.default_rng(42)
    init = UniformInitializer(2, -10, 10, pop_size=10, random_state=rng_a)
    mut = create_mutation_operator("gaussian_mutation", random_state=rng_a, N=1, F=0.1)
    surv = create_survivor_selection("generational", random_state=rng_a)
    strat = SearchStrategy(init, operator=mut, survivor_sel=surv, name="freq")
    algo = Algorithm(
        dummy_objfunc,
        strat,
        checkpoint_file=str(checkpoint),
        checkpoint_iteration_frequency=2,   # save every 2 iterations
        stop_cond="ngen",
        ngen=5,
        reporter="silent",
    )
    algo.optimize()

    # When the algorithm stops at gen=5, the last checkpoint was at gen=4.
    loaded = Checkpointer(str(checkpoint)).load(str(checkpoint), reporter="silent")
    assert loaded.stopping_condition.iterations == 4


# ---------------------------------------------------------------
# Time‑based checkpoint (uses SleepTest benchmark to force delay)
# ---------------------------------------------------------------
def test_checkpoint_time_frequency(tmp_path, rng):
    checkpoint = tmp_path / "time.pkl"

    # SleepTest with 0.1s per evaluation guarantees at least two saves
    sleepy = SleepTest(vecsize=2, sleep_time=0.1, mode="min")
    rng_a = np.random.default_rng(42)
    init = UniformInitializer(2, -10, 10, pop_size=3, random_state=rng_a)
    mut = create_mutation_operator("gaussian_mutation", random_state=rng_a, N=1, F=0.1)
    surv = create_survivor_selection("generational", random_state=rng_a)
    strat = SearchStrategy(init, operator=mut, survivor_sel=surv, name="time")
    algo = Algorithm(
        sleepy,
        strat,
        checkpoint_file=str(checkpoint),
        checkpoint_time_frequency=0.05,   # 50 ms
        stop_cond="ngen",
        ngen=3,
        reporter="silent",
    )
    algo.optimize()

    assert checkpoint.exists()
    loaded = Checkpointer(str(checkpoint)).load(str(checkpoint), reporter="silent")
    # The exact iteration depends on timing, but must be ≤ 3
    assert loaded.stopping_condition.iterations <= 3


# ---------------------------------------------------------------
# Checkpointer disabled when no file is provided
# ---------------------------------------------------------------
def test_checkpoint_disabled(rng, dummy_objfunc):
    rng_a = np.random.default_rng(42)
    init = UniformInitializer(2, -10, 10, pop_size=10, random_state=rng_a)
    mut = create_mutation_operator("gaussian_mutation", random_state=rng_a, N=1, F=0.1)
    surv = create_survivor_selection("generational", random_state=rng_a)
    strat = SearchStrategy(init, operator=mut, survivor_sel=surv, name="none")
    algo = Algorithm(
        dummy_objfunc,
        strat,
        stop_cond="ngen",
        ngen=3,
        reporter="silent",
    )
    assert algo.checkpointer is None


# ---------------------------------------------------------------
# Interruption saves a valid checkpoint
# ---------------------------------------------------------------
def test_checkpoint_interruption_saves(tmp_path, rng, dummy_objfunc):
    checkpoint = tmp_path / "interrupt.pkl"

    class InterruptOnStep(Algorithm):
        def step(self, population=None):
            raise KeyboardInterrupt

    rng_a = np.random.default_rng(42)
    init = UniformInitializer(2, -10, 10, pop_size=10, random_state=rng_a)
    mut = create_mutation_operator("gaussian_mutation", random_state=rng_a, N=1, F=0.1)
    surv = create_survivor_selection("generational", random_state=rng_a)
    strat = SearchStrategy(init, operator=mut, survivor_sel=surv, name="interrupt")
    algo = InterruptOnStep(
        dummy_objfunc,
        strat,
        checkpoint_file=str(checkpoint),
        checkpoint_iteration_frequency=1,
        stop_cond="ngen",
        ngen=5,
        reporter="silent",
    )

    with pytest.raises(KeyboardInterrupt):
        algo.optimize()

    assert checkpoint.exists()

    loaded = Checkpointer(str(checkpoint)).load(str(checkpoint), reporter="silent")
    # The interrupt happened before the first generation finished
    assert loaded.stopping_condition.iterations == 0
    # Population must be initialised
    assert loaded.population is not None
    assert loaded.population.genotype_matrix.shape == (10, 2)


# ---------------------------------------------------------------
# Load with a custom reporter
# ---------------------------------------------------------------
def test_checkpoint_load_with_reporter(tmp_path, rng, dummy_objfunc):
    checkpoint = tmp_path / "reporter.pkl"

    rng_a = np.random.default_rng(42)
    init = UniformInitializer(2, -10, 10, pop_size=10, random_state=rng_a)
    mut = create_mutation_operator("gaussian_mutation", random_state=rng_a, N=1, F=0.1)
    surv = create_survivor_selection("generational", random_state=rng_a)
    strat = SearchStrategy(init, operator=mut, survivor_sel=surv, name="reporter")
    algo = Algorithm(
        dummy_objfunc,
        strat,
        checkpoint_file=str(checkpoint),
        checkpoint_iteration_frequency=1,
        stop_cond="ngen",
        ngen=2,
        reporter="silent",
    )
    algo.optimize()

    # Load with verbose reporter
    loaded = Checkpointer(str(checkpoint)).load(str(checkpoint), reporter="verbose")
    # We can't compare types across packages easily, but at least ensure it's not silent
    assert not isinstance(loaded.reporter, SilentReporter)


# ---------------------------------------------------------------
# Load non‑existent file raises FileNotFoundError
# ---------------------------------------------------------------
def test_checkpoint_load_missing_file(tmp_path):
    missing = tmp_path / "ghost.pkl"
    ckp = Checkpointer(str(missing))
    with pytest.raises(FileNotFoundError):
        ckp.load(str(missing))