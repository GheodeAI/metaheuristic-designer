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

    def __init__(self, fid, dimension, instance=1, problem_class=None, ioh_options=None, name=None):
        if ioh is None:
            raise ImportError("IOHexperimenter is required for IOHObjective. " "Install it with `pip install ioh`.")

        if problem_class is None:
            problem_class = ProblemClass.BBOB

        if ioh_options is None:
            ioh_options = {}

        self._fid = fid
        self._instance = instance

        self.problem = ioh.get_problem(fid, instance=instance, dimension=dimension, problem_class=problem_class, **ioh_options)

        # Generate name if not specified
        if name is None:
            base = fid if isinstance(fid, str) else f"F{fid}"
            name = f"IOH-{problem_class}-{base}-D{dimension}-ins{instance}"

        super().__init__(
            dimension=dimension,
            lower_bound=self.problem.bounds.lb,
            upper_bound=self.problem.bounds.ub,
            mode="max",
            name=name,
            constraint_handler=None,
        )

    def objective(self, x):
        return self.problem(np.asarray(x, dtype=float))

    def attach_logger(self, logger):
        self.problem.attach_logger(logger)

    def detach_logger(self):
        self.problem.detach_logger()

    def restart(self):
        super().restart()
        self.problem.reset()


class BBOBObjective(IOHObjective):
    def __init__(self, fid, dimension, instance, ioh_options=None, name=None):
        if name is None:
            base = fid if isinstance(fid, str) else f"F{fid}"
            name = f"BBOB-{base}-D{dimension}-ins{instance}"

        super().__init__(fid=fid, dimension=dimension, instance=instance, problem_class=ProblemClass.BBOB, ioh_options=ioh_options, name=name)
