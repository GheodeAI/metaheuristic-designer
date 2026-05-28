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
    compact_name : str, optional
        Use a shortened name for the benchmark when compact_name is True.
    """

    def __init__(self, fid: int | str, dimension: int, instance: int = 1, problem_class: ProblemClass = None, compact_name: bool = False):
        if ioh is None:
            raise ImportError("IOHexperimenter is required for IOHObjective. " "Install it with `pip install ioh`.")

        if problem_class is None:
            problem_class = ProblemClass.BBOB

        self.fid = fid
        self.instance = instance

        self.problem = ioh.get_problem(fid, instance=instance, dimension=dimension, problem_class=problem_class)

        # Generate name if not specified
        fname = self.problem.meta_data.name
        if compact_name:
            name = f"IOH-{problem_class.name}-{fname}-D{dimension}"
        else:
            base = fid if isinstance(fid, str) else f"F{fid}"
            name = f"IOH-{problem_class.name}-{base}-{fname}-D{dimension}-ins{instance}"

        super().__init__(
            dimension=dimension,
            lower_bound=self.problem.bounds.lb.squeeze(),
            upper_bound=self.problem.bounds.ub.squeeze(),
            mode="min",
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
    def __init__(self, fid, dimension, instance, compact_name=None):
        super().__init__(fid=fid, dimension=dimension, instance=instance, problem_class=ProblemClass.BBOB)

        if compact_name:
            self.name = f"BBOB-{self.problem.meta_data.name}-d{dimension}"
        else:
            base = fid if isinstance(fid, str) else f"F{fid}"
            self.name = f"BBOB-{base}-{self.problem.meta_data.name}-d{dimension}-ins{instance}"
