import numpy as np

try:
    import ioh
    from ioh import ProblemClass
except ImportError:
    ioh = None
    ProblemClass = None

from metaheuristic_designer.objective_function import ObjectiveFunc


class IOHObjective(ObjectiveFunc):
    """
    Adapts an IOH benchmark problem to the ObjectiveFunc interface.

    Parameters
    ----------
    fid : int or str
        BBOB function ID (1-24) or name (e.g. ``"Sphere"``).
    dimension : int
        Problem dimensionality.
    instance : int, optional
        Problem instance (default 1).
    problem_class : ProblemClass, optional
        IOH problem type (default ``ProblemClass.BBOB``).
    ioh_options : dict, optional
        Extra keyword arguments passed to ``ioh.get_problem``.
    name : str, optional
        Custom display name; auto-generated if None.
    """

    def __init__(self, fid, dimension, instance=1, problem_class=None,
                 ioh_options=None, name=None):
        if ioh is None:
            raise ImportError(
                "IOHexperimenter is required for IOHObjective. "
                "Install it with `pip install ioh`."
            )

        if problem_class is None:
            problem_class = ProblemClass.BBOB

        self._fid = fid
        self._instance = instance

        self._ioh_problem = ioh.get_problem(
            fid,
            instance=instance,
            dimension=dimension,
            problem_class=problem_class,
            **(ioh_options or {})
        )

        bounds = self._ioh_problem.bounds

        # Generate name before super().__init__ so it's ready
        if name is None:
            base = fid if isinstance(fid, str) else f"F{fid}"
            name = f"IOH-bbob-{base}-D{dimension}-ins{instance}"

        super().__init__(
            dimension=dimension,
            lower_bound=bounds.lb,
            upper_bound=bounds.ub,
            mode="min",
            name=name,
            constraint_handler=None
        )

    def objective(self, x):
        return self._ioh_problem(np.asarray(x, dtype=float))

    def attach_logger(self, logger):
        self._ioh_problem.attach_logger(logger)

    def detach_logger(self):
        self._ioh_problem.detach_logger()

    def restart(self):
        super().restart()
        self._ioh_problem.reset()