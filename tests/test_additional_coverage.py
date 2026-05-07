"""
Additional tests targeting low-coverage modules:
  - utils.py (NumpyEncoder, check_random_state, per_individual, per_individual_list)
  - parameter_schedules/exponential_decay_schedule.py
  - operators/operator_functions/mutation.py  (additional distributions)
  - reporters (VerboseReporter, TQDMReporter via output capture)
  - operators/composite_operator.py, branch_operator.py, masked_operator.py
  - search_strategy.py (SearchStrategyFromLambda)
  - strategies/classic/CMA_ES.py
"""

import json
import logging
import numpy as np
import pytest
from enum import Enum
from io import StringIO
from unittest.mock import MagicMock, patch

# ===================================================================
#  utils.py
# ===================================================================
from metaheuristic_designer.utils import (
    NumpyEncoder,
    check_random_state,
    per_individual,
    per_individual_list,
)


class _TestEnum(Enum):
    FOO = 1
    BAR = 2


class TestNumpyEncoder:
    def test_numpy_integer(self):
        val = np.int32(42)
        result = json.dumps(val, cls=NumpyEncoder)
        assert result == "42"

    def test_numpy_float(self):
        val = np.float64(3.14)
        result = json.dumps(val, cls=NumpyEncoder)
        assert float(result) == pytest.approx(3.14)

    def test_numpy_ndarray(self):
        arr = np.array([1, 2, 3])
        result = json.loads(json.dumps(arr, cls=NumpyEncoder))
        assert result == [1, 2, 3]

    def test_enum_value(self):
        result = json.dumps(_TestEnum.FOO, cls=NumpyEncoder)
        assert "FOO" in result

    def test_fallback_raises_for_unknown_type(self):
        with pytest.raises(TypeError):
            json.dumps(object(), cls=NumpyEncoder)


class TestCheckRandomState:
    def test_none_returns_generator(self):
        rng = check_random_state(None)
        assert isinstance(rng, np.random.Generator)

    def test_numpy_random_module_returns_generator(self):
        rng = check_random_state(np.random)
        assert isinstance(rng, np.random.Generator)

    def test_int_seed_returns_deterministic_generator(self):
        rng1 = check_random_state(0)
        rng2 = check_random_state(0)
        assert isinstance(rng1, np.random.Generator)
        r1 = rng1.random()
        r2 = rng2.random()
        assert r1 == r2

    def test_generator_passthrough(self):
        gen = np.random.default_rng(7)
        result = check_random_state(gen)
        assert result is gen

    def test_invalid_seed_raises_value_error(self):
        with pytest.raises(ValueError):
            check_random_state("not_a_valid_seed")


class TestPerIndividual:
    def test_applies_function_row_wise(self):
        @per_individual
        def row_sum(row, **kwargs):
            return np.sum(row)

        mat = np.array([[1, 2], [3, 4], [5, 6]])
        result = row_sum(mat)
        np.testing.assert_array_equal(result, [3, 7, 11])

    def test_passes_kwargs(self):
        @per_individual
        def scale_row(row, factor=1):
            return row * factor

        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = scale_row(mat, factor=2)
        np.testing.assert_array_equal(result, [[2.0, 4.0], [6.0, 8.0]])


class TestPerIndividualList:
    def test_applies_function_element_wise(self):
        @per_individual_list
        def double(x, **kwargs):
            return x * 2

        result = double([1, 2, 3])
        assert result == [2, 4, 6]

    def test_passes_kwargs(self):
        @per_individual_list
        def add(x, offset=0):
            return x + offset

        result = add([10, 20], offset=5)
        assert result == [15, 25]


# ===================================================================
#  ExponentialDecaySchedule
# ===================================================================
from metaheuristic_designer.parameter_schedules.exponential_decay_schedule import (
    ExponentialDecaySchedule,
)


class TestExponentialDecaySchedule:
    def test_init_stores_values(self):
        sched = ExponentialDecaySchedule(init_value=10.0, final_value=0.0, alpha=0.9)
        assert sched.init_value == 10.0
        assert sched.final_value == 0.0
        assert sched.curr_value == 10.0
        assert sched.alpha == 0.9
        assert sched.iterative is True

    def test_iterative_decay_decreases_value(self):
        sched = ExponentialDecaySchedule(init_value=10.0, final_value=0.0, alpha=0.9)
        v1 = sched.evaluate(0.0)
        v2 = sched.evaluate(0.0)
        assert v1 == pytest.approx(9.0)
        assert v2 == pytest.approx(8.1)

    def test_iterative_converges_to_final(self):
        sched = ExponentialDecaySchedule(init_value=1.0, final_value=0.0, alpha=0.5)
        for _ in range(50):
            val = sched.evaluate(0.0)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_progress_based_decay(self):
        sched = ExponentialDecaySchedule(
            init_value=10.0, final_value=0.0, alpha=1.0, iterative=False
        )
        # At progress=0, value should be init_value
        v0 = sched.evaluate(0.0)
        assert v0 == pytest.approx(10.0)
        # At progress=1, value should approach final_value
        v1 = sched.evaluate(1.0)
        # 0 + (10 - 0) * exp(-1.0 * 1) ≈ 3.678
        assert v1 == pytest.approx(10.0 * np.exp(-1.0), rel=1e-4)

    def test_non_iterative_with_nonzero_final(self):
        sched = ExponentialDecaySchedule(
            init_value=10.0, final_value=2.0, alpha=1.0, iterative=False
        )
        # At progress=0: 2 + (10 - 2) * exp(0) = 10
        assert sched.evaluate(0.0) == pytest.approx(10.0)
        # At large progress, approaches final_value
        assert sched.evaluate(100.0) == pytest.approx(2.0, abs=1e-5)

    def test_alpha_outside_0_1_triggers_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            ExponentialDecaySchedule(init_value=1.0, alpha=1.5, iterative=True)
        assert "HIGHLY recommended" in caplog.text


# ===================================================================
#  mutation.py – additional distribution coverage
# ===================================================================
from metaheuristic_designer.operators.operator_functions.mutation import (
    ProbDist,
    sample_distribution,
    mutate_sample,
    mutate_noise,
    rand_sample,
    rand_noise,
    multivariate_categorical,
    xor_mask,
)


@pytest.fixture
def rng_fixed():
    return np.random.default_rng(42)


class TestProbDistFromStr:
    def test_valid_string_gauss(self):
        assert ProbDist.from_str("gauss") == ProbDist.GAUSS

    def test_valid_string_gaussian(self):
        assert ProbDist.from_str("gaussian") == ProbDist.GAUSS

    def test_valid_string_normal(self):
        assert ProbDist.from_str("normal") == ProbDist.GAUSS

    def test_valid_string_uniform(self):
        assert ProbDist.from_str("uniform") == ProbDist.UNIFORM

    def test_valid_string_categorical(self):
        assert ProbDist.from_str("categorical") == ProbDist.CATEGORICAL

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="not defined"):
            ProbDist.from_str("nonexistent_distribution")


class TestMultivariateCategorical:
    def test_rvs_default_size_uses_cumsum_rows(self):
        cats = np.array([0, 1, 2])
        weights = np.ones((3, 3)) / 3
        dist = multivariate_categorical(cats, weights)
        # Default size = cumsum_matrix.shape[0] = 3 (number of rows)
        # The vectorize signature "(n),()->()" broadcasts over the batch dim,
        # returning a 1D array of shape (3,) — one selected category per row.
        result = dist.rvs(random_state=np.random.default_rng(1))
        assert result.shape[0] == 3
        assert np.all(result >= 0) and np.all(result < 3)

    def test_rvs_scalar_size(self):
        cats = np.array([0, 1, 2])
        weights = np.ones((3, 3)) / 3
        dist = multivariate_categorical(cats, weights)
        result = dist.rvs(size=5, random_state=np.random.default_rng(1))
        assert result is not None

    def test_rvs_tuple_size(self):
        cats = np.array([0, 1])
        weights = np.array([[0.7, 0.3], [0.4, 0.6]])
        dist = multivariate_categorical(cats, weights)
        result = dist.rvs(size=(4,), random_state=np.random.default_rng(1))
        assert result.shape[1] == 2


class TestSampleDistributionAdditionalDistribs:
    def test_cauchy(self, rng_fixed):
        result = sample_distribution((3, 2), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.CAUCHY)
        assert result.shape == (3, 2)

    def test_laplace(self, rng_fixed):
        result = sample_distribution((3, 2), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.LAPLACE)
        assert result.shape == (3, 2)

    def test_gamma(self, rng_fixed):
        result = sample_distribution((3, 2), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.GAMMA, a=2)
        assert result.shape == (3, 2)
        assert np.all(result >= 0)

    def test_expon(self, rng_fixed):
        result = sample_distribution((3, 2), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.EXPON)
        assert result.shape == (3, 2)

    def test_poisson(self, rng_fixed):
        result = sample_distribution((3, 2), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.POISSON, mu=3)
        assert result.shape == (3, 2)
        assert np.all(result >= 0)

    def test_bernoulli(self, rng_fixed):
        result = sample_distribution((4, 3), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.BERNOULLI, p=0.5)
        assert result.shape == (4, 3)
        assert set(np.unique(result)).issubset({0, 1})

    def test_binomial(self, rng_fixed):
        result = sample_distribution((3, 2), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.BINOMIAL, n=10, p=0.3)
        assert result.shape == (3, 2)
        assert np.all(result >= 0) and np.all(result <= 10)

    def test_categorical(self, rng_fixed):
        p = np.array([0.2, 0.5, 0.3])
        result = sample_distribution((5, 3), random_state=rng_fixed, distrib=ProbDist.CATEGORICAL, p=p)
        assert result.shape == (5, 3)
        assert np.all(result >= 0) and np.all(result < 3)

    @pytest.mark.xfail(reason="Bug in multivariate_categorical.rvs: shape mismatch when size != cumsum_matrix rows (see ERRORES.md)")
    def test_multicategorical(self, rng_fixed):
        p = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
        # MULTICATEGORICAL sets shape = shape[0] for rvs
        result = sample_distribution((2, 3), random_state=rng_fixed, distrib=ProbDist.MULTICATEGORICAL, p=p)
        assert result is not None

    def test_custom_distribution(self, rng_fixed):
        import scipy.stats as stats
        custom_dist = stats.norm(loc=5, scale=2)
        result = sample_distribution((3, 2), random_state=rng_fixed, distrib=ProbDist.CUSTOM, distrib_class=custom_dist)
        assert result.shape == (3, 2)

    def test_custom_without_distrib_class_raises(self, rng_fixed):
        with pytest.raises(ValueError, match="distrib_class"):
            sample_distribution((2, 2), random_state=rng_fixed, distrib=ProbDist.CUSTOM)

    def test_string_distrib_name(self, rng_fixed):
        result = sample_distribution((2, 2), loc=0, scale=1, random_state=rng_fixed, distrib="gauss")
        assert result.shape == (2, 2)

    def test_multigauss_single(self, rng_fixed):
        shape = (3, 2)
        result = sample_distribution(shape, loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.MULTIGAUSS)
        assert result.shape == shape

    def test_levystable(self, rng_fixed):
        result = sample_distribution((3, 2), loc=0, scale=1, random_state=rng_fixed, distrib=ProbDist.LEVYSTABLE, a=1.5, b=0)
        assert result.shape == (3, 2)


class TestXorMaskModes:
    def test_xor_mask_bin_mode(self):
        rng = np.random.default_rng(0)
        pop = np.array([[0b1010, 0b0101], [0b1100, 0b0011]], dtype=np.uint8)
        result = xor_mask(pop, None, random_state=rng, N=2, BinRep="bin")
        assert result.shape == pop.shape

    def test_xor_mask_int_mode(self):
        rng = np.random.default_rng(0)
        pop = np.array([[100, 200], [300, 400]], dtype=np.int32)
        result = xor_mask(pop, None, random_state=rng, N=2, BinRep="int")
        assert result.shape == pop.shape

    def test_xor_mask_unknown_mode_returns_unchanged(self):
        rng = np.random.default_rng(0)
        pop = np.array([[5, 10], [15, 20]], dtype=np.int32)
        result = xor_mask(pop, None, random_state=rng, N=2, BinRep="unknown_mode")
        np.testing.assert_array_equal(result, pop)


class TestMutateSampleWithCalculatedLocScale:
    def test_calculated_loc_and_scale(self):
        rng = np.random.default_rng(42)
        pop = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
        result = mutate_sample(pop.copy(), None, random_state=rng, N=2, distrib=ProbDist.GAUSS, loc="calculated", scale="calculated")
        assert result.shape == (2, 3)

    def test_uniform_with_min_max(self):
        rng = np.random.default_rng(42)
        pop = np.ones((3, 4), dtype=float)
        result = mutate_sample(pop.copy(), None, random_state=rng, N=2, distrib=ProbDist.UNIFORM, min=0.0, max=1.0)
        assert result.shape == (3, 4)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


class TestMutateNoiseUniform:
    def test_uniform_with_min_max(self):
        rng = np.random.default_rng(42)
        pop = np.zeros((3, 4), dtype=float)
        result = mutate_noise(pop.copy(), None, random_state=rng, N=2, distrib=ProbDist.UNIFORM, min=0.0, max=1.0)
        assert result.shape == (3, 4)

    def test_with_strength_array(self):
        rng = np.random.default_rng(42)
        pop = np.ones((3, 4), dtype=float)
        F = np.array([0.5, 1.0, 0.3])
        result = mutate_noise(pop.copy(), None, random_state=rng, N=2, distrib=ProbDist.GAUSS, loc=0, scale=1, F=F)
        assert result.shape == (3, 4)


class TestRandSampleUniform:
    def test_uniform_with_min_max(self):
        rng = np.random.default_rng(42)
        pop = np.ones((3, 4), dtype=float)
        result = rand_sample(pop.copy(), None, random_state=rng, distrib=ProbDist.UNIFORM, min=0.0, max=5.0)
        assert result.shape == (3, 4)
        assert np.all(result >= 0.0) and np.all(result <= 5.0)

    def test_calculated_loc_scale(self):
        rng = np.random.default_rng(42)
        pop = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = rand_sample(pop.copy(), None, random_state=rng, distrib=ProbDist.GAUSS, loc="calculated", scale="calculated")
        assert result.shape == (2, 2)


class TestRandNoiseUniform:
    def test_uniform_with_min_max(self):
        rng = np.random.default_rng(42)
        pop = np.ones((3, 4), dtype=float)
        result = rand_noise(pop.copy(), None, random_state=rng, distrib=ProbDist.UNIFORM, min=0.0, max=2.0)
        assert result.shape == (3, 4)

    def test_with_strength_array(self):
        rng = np.random.default_rng(42)
        pop = np.ones((3, 4), dtype=float)
        F = np.array([0.5, 1.0, 0.3])
        result = rand_noise(pop.copy(), None, random_state=rng, distrib=ProbDist.GAUSS, loc=0, scale=1, F=F)
        assert result.shape == (3, 4)


# ===================================================================
#  CompositeOperator – uncovered branch (with_weights=True via step)
# ===================================================================
from metaheuristic_designer.operators.composite_operator import CompositeOperator
from metaheuristic_designer.operator import NullOperator
from metaheuristic_designer.benchmarks.benchmark_funcs import MaxOnes
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.population import Population


def _make_population(n=5, dim=4, seed=0):
    objfunc = MaxOnes(dim)
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 2, size=(n, dim)).astype(float)
    pop = Population(objfunc, geno)
    pop.calculate_fitness()
    return pop, objfunc


class TestCompositeOperatorCoverage:
    def test_evolve_chained_operators(self):
        pop, _ = _make_population()
        op1 = NullOperator()
        op2 = NullOperator()
        comp = CompositeOperator([op1, op2])
        result = comp.evolve(pop)
        assert result.pop_size == pop.pop_size

    def test_auto_name_generation(self):
        op1 = NullOperator(name="A")
        op2 = NullOperator(name="B")
        comp = CompositeOperator([op1, op2])
        assert "A" in comp.name and "B" in comp.name

    def test_get_state_includes_op_list(self):
        ops = [NullOperator(name="op1"), NullOperator(name="op2")]
        comp = CompositeOperator(ops)
        state = comp.get_state()
        # The key is 'op_list' in the actual implementation
        assert "op_list" in state

    def test_step_propagates_to_suboperators(self):
        ops = [NullOperator(), NullOperator()]
        comp = CompositeOperator(ops)
        comp.step(0.5)


# ===================================================================
#  BranchOperator – uncovered branches
# ===================================================================
from metaheuristic_designer.operators.branch_operator import BranchOperator


class TestBranchOperatorCoverage:
    def test_random_mode_selects_one_operator(self):
        pop, _ = _make_population()
        op1 = NullOperator()
        op2 = NullOperator()
        branch = BranchOperator([op1, op2], mode="RANDOM", random_state=42)
        result = branch.evolve(pop)
        assert result.pop_size == pop.pop_size

    def test_pick_mode_cycles_operators(self):
        pop, _ = _make_population()
        op1 = NullOperator()
        op2 = NullOperator()
        branch = BranchOperator([op1, op2], mode="PICK", random_state=42)
        result1 = branch.evolve(pop)
        result2 = branch.evolve(pop)
        assert result1.pop_size == pop.pop_size
        assert result2.pop_size == pop.pop_size

    def test_get_state(self):
        op1 = NullOperator()
        op2 = NullOperator()
        branch = BranchOperator([op1, op2], random_state=0)
        state = branch.get_state()
        assert "op_list" in state

    def test_step_propagates(self):
        branch = BranchOperator([NullOperator(), NullOperator()], random_state=0)
        branch.step(0.3)

    def test_with_weights(self):
        pop, _ = _make_population()
        op1 = NullOperator()
        op2 = NullOperator()
        branch = BranchOperator([op1, op2], mode="RANDOM", weights=[0.9, 0.1], random_state=42)
        result = branch.evolve(pop)
        assert result.pop_size == pop.pop_size


# ===================================================================
#  MaskedOperator – uncovered branch
# ===================================================================
from metaheuristic_designer.operators.masked_operator import MaskedOperator


class TestMaskedOperatorCoverage:
    def test_evolve_applies_operator_to_mask(self):
        pop, _ = _make_population(n=4, dim=4)
        op1 = NullOperator()
        op2 = NullOperator()
        mask = np.array([True, True, False, False])
        # MaskedOperator takes a list of operators, not a single operator
        masked = MaskedOperator([op1, op2], mask)
        result = masked.evolve(pop)
        assert result.pop_size == pop.pop_size

    def test_get_state(self):
        op1 = NullOperator()
        op2 = NullOperator()
        mask = np.array([True, False])
        masked = MaskedOperator([op1, op2], mask)
        state = masked.get_state()
        assert state is not None

    def test_step_propagates(self):
        op1 = NullOperator()
        op2 = NullOperator()
        mask = np.array([True, False])
        masked = MaskedOperator([op1, op2], mask)
        masked.step(0.5)


# ===================================================================
#  CMA-ES strategy – basic initialization and perturbation test
# ===================================================================
from metaheuristic_designer.strategies.classic.CMA_ES import CMA_ES
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.algorithm import Algorithm
from metaheuristic_designer.survivor_selection import create_survivor_selection


class TestCMAES:
    @pytest.mark.xfail(reason="CMA_ES.initialize uses self.offspring_size but VariablePopulation stores it in self.params.offspring_size (see ERRORES.md)")
    def test_initialize_with_bounded_objfunc(self):
        dim = 3
        initializer = UniformInitializer(dim, np.full(dim, -5.0), np.full(dim, 5.0), pop_size=10)
        objfunc = Sphere(dim, mode="min")
        strategy = CMA_ES(initializer, survivor_sel=create_survivor_selection("keep_best", amount=10))
        pop = strategy.initialize(objfunc)
        assert pop.pop_size == 10
        assert pop.genotype_matrix.shape[1] == dim

    @pytest.mark.xfail(reason="CMA_ES.initialize uses self.offspring_size but VariablePopulation stores it in self.params.offspring_size (see ERRORES.md)")
    def test_initialize_without_bounds(self):
        from metaheuristic_designer.initializers import UniformInitializer as UI2
        dim = 2
        initializer = UI2(dim, np.full(dim, -1.0), np.full(dim, 1.0), pop_size=5)
        objfunc = Sphere(dim, mode="min")
        strategy = CMA_ES(initializer, survivor_sel=create_survivor_selection("keep_best", amount=5))
        pop = strategy.initialize(objfunc)
        assert pop.pop_size == 5

    @pytest.mark.xfail(reason="CMA_ES.initialize uses self.offspring_size but VariablePopulation stores it in self.params.offspring_size (see ERRORES.md)")
    def test_perturb_runs(self):
        dim = 3
        initializer = UniformInitializer(dim, np.full(dim, -5.0), np.full(dim, 5.0), pop_size=10)
        objfunc = Sphere(dim, mode="min")
        strategy = CMA_ES(initializer, survivor_sel=create_survivor_selection("keep_best", amount=10))
        pop = strategy.initialize(objfunc)
        pop.calculate_fitness()
        offspring = strategy.perturb(pop)
        assert offspring is not None

    @pytest.mark.xfail(reason="CMA_ES.initialize uses self.offspring_size but VariablePopulation stores it in self.params.offspring_size (see ERRORES.md)")
    def test_run_short_optimization(self):
        dim = 2
        initializer = UniformInitializer(dim, np.full(dim, -3.0), np.full(dim, 3.0), pop_size=5)
        objfunc = Sphere(dim, mode="min")
        strategy = CMA_ES(initializer, survivor_sel=create_survivor_selection("keep_best", amount=5))
        algo = Algorithm(objfunc, strategy, stop_cond="max_iterations", max_iterations=3)
        pop = algo.optimize()
        assert pop is not None
        _, fit = algo.best_solution()
        assert isinstance(fit, float)


# ===================================================================
#  VerboseReporter and TQDMReporter – output capture
# ===================================================================
from metaheuristic_designer.reporters.verbose_reporter import VerboseReporter
from metaheuristic_designer.reporters.tqdm_reporter import TQDMReporter
from metaheuristic_designer.reporters.silent_reporter import SilentReporter
from metaheuristic_designer.reporters.create_reporter import create_reporter


def _make_mock_algorithm():
    algo = MagicMock()
    algo.name = "TestAlgo"
    algo.iterations = 10
    algo.evaluations = 100
    algo.progress = 0.5
    algo.best_solution.return_value = (np.array([1.0, 2.0]), 0.5)
    algo.patience_left = 50
    algo.search_strategy = MagicMock()
    algo.search_strategy.pop_size = 10
    algo.search_strategy.extra_report = MagicMock()
    algo.objfunc = MagicMock()
    algo.objfunc.name = "MockObjFunc"
    # Attributes used by VerboseReporter.log_end
    algo.history_tracker = MagicMock()
    algo.history_tracker.recorded_iterations = 10
    algo.stopping_condition = MagicMock()
    algo.stopping_condition.real_time_spent = 1.23
    algo.stopping_condition.cpu_time_spent = 1.00
    algo.stopping_condition.evaluations = 100
    algo.stopping_condition.patience_left = 50
    # log_step uses these attributes
    algo.stopping_condition.real_time_limit = 60.0
    algo.stopping_condition.progress = 0.5
    return algo


class TestVerboseReporter:
    def test_log_init_does_not_crash(self, capsys):
        reporter = VerboseReporter()
        algo = _make_mock_algorithm()
        reporter.log_init(algo)
        out = capsys.readouterr().out
        assert isinstance(out, str)

    def test_log_step_does_not_crash(self, capsys):
        reporter = VerboseReporter()
        algo = _make_mock_algorithm()
        reporter.log_init(algo)
        reporter.log_step(algo)
        capsys.readouterr()

    def test_log_end_does_not_crash(self, capsys):
        reporter = VerboseReporter()
        algo = _make_mock_algorithm()
        reporter.log_init(algo)
        reporter.log_end(algo)
        capsys.readouterr()


class TestTQDMReporter:
    def test_log_init_creates_progress_bar(self):
        reporter = TQDMReporter()
        algo = _make_mock_algorithm()
        reporter.log_init(algo)
        reporter.log_end(algo)

    def test_log_step_updates_bar(self):
        reporter = TQDMReporter()
        algo = _make_mock_algorithm()
        reporter.log_init(algo)
        reporter.log_step(algo)
        reporter.log_end(algo)


class TestCreateReporter:
    def test_create_silent_reporter(self):
        r = create_reporter("silent")
        assert isinstance(r, SilentReporter)

    def test_create_verbose_reporter(self):
        r = create_reporter("verbose")
        assert isinstance(r, VerboseReporter)

    def test_create_tqdm_reporter(self):
        r = create_reporter("tqdm")
        assert isinstance(r, TQDMReporter)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_reporter("nonexistent_reporter_type")


# ===================================================================
#  SearchStrategy – SearchStrategyFromLambda and gather_parameters
# ===================================================================
from metaheuristic_designer.search_strategy import SearchStrategyFromLambda
from metaheuristic_designer.initializers import UniformInitializer as UI


class TestSearchStrategyFromLambda:
    def test_creates_and_runs(self):
        dim = 2
        init = UI(dim, np.full(dim, -1.0), np.full(dim, 1.0), pop_size=4)
        objfunc = Sphere(dim, mode="min")

        def my_perturb(pop, **kwargs):
            return pop

        strategy = SearchStrategyFromLambda(
            initializer=init,
            perturb_func=my_perturb,
        )
        pop = strategy.initialize(objfunc)
        assert pop.pop_size == 4
        result = strategy.perturb(pop)
        assert result.pop_size == 4

    def test_gather_parameters_returns_dict(self):
        dim = 2
        init = UI(dim, np.full(dim, -1.0), np.full(dim, 1.0), pop_size=3)
        strategy = SearchStrategyFromLambda(
            initializer=init,
            perturb_func=lambda pop, **kwargs: pop,
        )
        params = strategy.gather_parameters()
        assert isinstance(params, dict)
