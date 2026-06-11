from conftest import dummy_objfunc, dummy_initializer, dummy_strategy
from metaheuristic_designer.strategies import NoSearch
from metaheuristic_designer.algorithms import Algorithm
from metaheuristic_designer.reporters import VerboseReporter, TQDMReporter


def test_verbose_reporter(dummy_objfunc, dummy_strategy):
    reporter = VerboseReporter(verbose_timer=0)
    alg = Algorithm(objfunc=dummy_objfunc, search_strategy=dummy_strategy, stop_condition_str="max_iterations", max_iterations=10, reporter=reporter)
    alg.optimize()


def test_verbose_reporter_skip(dummy_objfunc, dummy_strategy):
    reporter = VerboseReporter(verbose_timer=10)
    alg = Algorithm(objfunc=dummy_objfunc, search_strategy=dummy_strategy, stop_condition_str="max_iterations", max_iterations=10, reporter=reporter)
    alg.optimize()


def test_tqdm_reporter(dummy_objfunc, dummy_strategy):
    alg = Algorithm(
        objfunc=dummy_objfunc, search_strategy=dummy_strategy, stop_condition_str="max_iterations", max_iterations=2, reporter=TQDMReporter()
    )
    alg.optimize()


def test_tqdm_reporter_round_resolution(dummy_objfunc, dummy_strategy):
    reporter = TQDMReporter(resolution=100.2)
    assert isinstance(reporter.resolution, int)


def test_tqdm_reporter_fill_bar(dummy_objfunc, dummy_strategy):
    reporter = TQDMReporter()
    alg = Algorithm(objfunc=dummy_objfunc, search_strategy=dummy_strategy, stop_condition_str="max_iterations", max_iterations=2, reporter=reporter)
    alg.initialize()
    reporter.log_init(alg)
    reporter.log_end(alg)
