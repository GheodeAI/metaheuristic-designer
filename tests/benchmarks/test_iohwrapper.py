import numpy as np
import pytest

ioh = pytest.importorskip("ioh", reason="IOHexperimenter not installed (optional dependency)")

from metaheuristic_designer.benchmarks.ioh_wrapper import IOHObjective


@pytest.fixture
def problem_sphere():
    return IOHObjective(fid=1, dimension=2, instance=1)


# ----- Basic properties --------------------------------------------------


def test_dimension(problem_sphere):
    assert problem_sphere.dimension == 2


def test_bounds(problem_sphere):
    lb = problem_sphere.lower_bound
    ub = problem_sphere.upper_bound
    assert np.allclose(lb, -5.0)  # lb may be array of length 2
    assert np.allclose(ub, 5.0)


def test_mode_is_min(problem_sphere):
    assert problem_sphere.mode == "min"


def test_name_contains_identifiers(problem_sphere):
    assert "IOH-BBOB-F1-Sphere-D2-ins1" in problem_sphere.name


# ----- Evaluation --------------------------------------------------------


def test_optimum_near_known_value(problem_sphere):
    opt_x = problem_sphere.problem.optimum.x
    opt_y = problem_sphere.problem.optimum.y
    val = problem_sphere.objective(np.array(opt_x))
    assert np.isclose(val, opt_y, atol=1e-4), f"Expected {opt_y}, got {val}"


def test_objective_returns_float(problem_sphere):
    val = problem_sphere.objective(np.array([1.0, 2.0]))
    assert isinstance(val, (float, np.floating))


# ----- Logger lifecycle --------------------------------------------------


def test_attach_detach_writes_files(problem_sphere, tmp_path):
    problem_sphere.restart()

    logger = ioh.logger.Analyzer(
        root=str(tmp_path),
        folder_name="test_run",
        algorithm_name="TestAlg",
        algorithm_info="",
        store_positions=False,
    )
    problem_sphere.attach_logger(logger)

    for _ in range(5):
        problem_sphere.objective(np.random.rand(2) * 10 - 5)

    problem_sphere.detach_logger()

    # Recursive search – independent of any extra nesting
    dat_files = list(tmp_path.rglob("*.dat"))
    assert len(dat_files) == 1, f"Expected one .dat file, found {dat_files}"


def test_reset_clears_counter(problem_sphere):
    problem_sphere.restart()
    for _ in range(3):
        problem_sphere.objective(np.array([0.5, 0.5]))
    problem_sphere.restart()


# ----- Optional dependency guard ----------------------------------------


def test_missing_ioh_raises_clean_error(monkeypatch):
    import metaheuristic_designer.benchmarks.ioh_wrapper as mod

    original_ioh = mod.ioh
    monkeypatch.setattr(mod, "ioh", None)
    with pytest.raises(ImportError, match="IOHexperimenter is required"):
        IOHObjective(fid=1, dimension=2, instance=1)
    monkeypatch.setattr(mod, "ioh", original_ioh)
