"""
Base class for the Objective Function module.

This module implements objective functions that measure the quality of solutions
in metaheuristic optimization problems. It provides abstractions for defining
fitness evaluation, constraint handling, and solution evaluation strategies.

The module includes several classes to support different optimization scenarios:

- :class:`ObjectiveFunc`: Abstract base class for all objective functions
- :class:`VectorObjectiveFunc`: Specialized class for vector-based objective functions
- :class:`NullObjectiveFunc`: No-op objective function for testing
- :class:`ObjectiveFromLambda`: Wrapper for callable-based objective functions
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np

from metaheuristic_designer.constraint_handlers.bounce_bound_constraint import BounceBoundConstraint
from metaheuristic_designer.constraint_handlers.extended_constraint import ExtendedConstraintHandler
from metaheuristic_designer.encodings.parameter_extending_encoding import ParameterExtendingEncoding
from .constraint_handler import ConstraintHandler, NullConstraint
from .constraint_handlers import ClipBoundConstraint, CompositeConstraint
from .parametrizable_mixin import ParametrizableMixin
from .utils import MatrixLike, check_random_state, RNGLike, VectorLike, ScalarLike

if TYPE_CHECKING:
    from metaheuristic_designer.population import Population

logger = logging.getLogger(__name__)


class ObjectiveFunc(ParametrizableMixin, ABC):
    """
    Abstract base class for objective functions in metaheuristic optimization.

    This class defines the interface and common functionality for evaluating the quality
    of candidate solutions in optimization problems. Subclasses must implement the
    :meth:`objective` method to define problem-specific fitness evaluation.

    The class handles several key aspects of optimization:
    - **Fitness calculation**: Evaluating solutions with optional constraint penalties
    - **Constraint management**: Applying penalties or repairs for constraint violations
    - **Mode support**: Supports both minimization and maximization problems
    - **Vectorized evaluation**: Can evaluate populations all at once or individually
    - **Caching**: Optionally caches fitness values to avoid redundant calculations

    Parameters
    ----------
    constraint_handler : ConstraintHandler, optional
        Handler for enforcing problem constraints. If not provided, uses :class:`NullConstraint`
        which applies no constraints. Default is None.
    mode : str, optional
        Optimization mode: 'max' for maximization or 'min' for minimization.
        Default is 'max'.
    name : str, optional
        Display name for this objective function. Default is 'some function'.
    vectorized : bool, optional
        If True, the :meth:`objective` method accepts and returns arrays for entire
        populations. If False, it processes one solution at a time. Default is False.
    recalculate : bool, optional
        If True, recalculates fitness for all individuals even if previously computed.
        If False, skips calculation for individuals with cached fitness. Default is False.
    **kwargs
        Additional keyword arguments stored and passed to the objective function
        via :meth:`ParametrizableMixin.store_kwargs`.

    Attributes
    ----------
    constraint_handler : ConstraintHandler
        The constraint handler managing restrictions for this problem.
    name : str
        Display name of the objective function.
    counter : int
        Number of fitness evaluations performed.
    factor : int
        Scaling factor: 1 for maximization, -1 for minimization.
    vectorized : bool
        Whether the objective function is vectorized.
    recalculate : bool
        Whether to recalculate fitness even if cached.
    mode : str
        The optimization mode ('max' or 'min').

    Raises
    ------
    ValueError
        If mode is not 'max' or 'min'.

    See Also
    --------
    VectorObjectiveFunc : Specialized class for vector optimization problems
    ObjectiveFromLambda : Wrapper for callable-based objective functions
    ConstraintHandler : Base class for constraint handling strategies
    """

    def __init__(
        self,
        constraint_handler: Optional[ConstraintHandler] = None,
        mode: str = "max",
        name: str = "some function",
        vectorized: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Initialize an ObjectiveFunc instance.

        Parameters
        ----------
        constraint_handler : ConstraintHandler, optional
            Constraint handler for the problem. Default is None (uses NullConstraint).
        mode : str, optional
            Optimization direction. Must be 'max' or 'min'. Default is 'max'.
        name : str, optional
            Display name for the objective function. Default is 'some function'.
        vectorized : bool, optional
            Whether the objective method is vectorized. Default is False.
        recalculate : bool, optional
            Whether to recalculate fitness regardless of cache. Default is False.
        **kwargs
            Additional arguments stored via the ParametrizableMixin.

        Raises
        ------
        ValueError
            If mode is neither 'max' nor 'min'.
        """
        super().__init__()

        if constraint_handler is None:
            constraint_handler = NullConstraint()
        self.constraint_handler = constraint_handler
        self.name = name
        self.counter = 0
        self.factor = 1
        self.vectorized = vectorized
        self.recalculate = recalculate

        if mode not in ["max", "min"]:
            raise ValueError('Optimization objective (mode) must be "min" or "max".')
        self.mode = mode

        if mode == "min":
            self.factor = -1

        self.store_kwargs(**kwargs)

    def __call__(self, population: Population, adjusted: bool = True, parallel: bool = False, threads: int = 8) -> VectorLike:
        """
        Evaluate the objective function for a population.

        This is a convenience method that delegates to :meth:`fitness`.

        Parameters
        ----------
        population : Population
            The population of solutions to evaluate.
        adjusted : bool, optional
            Whether to apply mode adjustment and constraint penalties. Default is True.
        parallel : bool, optional
            Whether to use parallel evaluation (currently not implemented). Default is False.
        threads : int, optional
            Number of threads for parallel evaluation (currently not implemented). Default is 8.

        Returns
        -------
        fitness : ndarray
            Adjusted fitness values for all individuals in the population.

        Notes
        -----
        Currently, the `adjusted` parameter is ignored; fitness is always adjusted.
        """

        return self.fitness(population, adjusted)

    def fitness(self, population: Population, parallel: bool = False, threads: int = 8) -> VectorLike:
        """
        Calculate adjusted fitness values for a population with constraint penalties.

        This method is the main entry point for fitness evaluation. It:

        1. Determines which individuals need fitness calculation (based on caching)
        2. Computes constraint penalties for violating solutions
        3. Evaluates objective values (vectorized or per-individual)
        4. Combines objective values with penalties and mode adjustment
        5. Updates the population's fitness and objective caches

        The fitness value is computed as:

        .. math::

            f_i = \\text{factor} \\times (obj_i - penalty_i)

        where :math:`\\text{factor} = 1` for maximization and :math:`-1` for minimization,
        and :math:`obj_i` is the raw objective value for solution :math:`i`.

        Parameters
        ----------
        population : Population
            The population of candidate solutions to evaluate.
        parallel : bool, optional
            Enable parallel evaluation (not currently implemented). Default is False.
        threads : int, optional
            Number of threads for parallel computation (ignored if not implemented). Default is 8.

        Returns
        -------
        fitness : ndarray
            Array of adjusted fitness values, shape (population.pop_size,).
            Values are properly adjusted for minimization/maximization and penalized
            for constraint violations.

        Notes
        -----
        - Fitness values are cached in the population and only recalculated if needed
        - If `self.recalculate` is True, all individuals are recalculated
        - If `self.recalculate` is False, only individuals with `fitness_calculated == 0` are evaluated
        - Duplicate solutions are detected and cached to avoid redundant calculations
        - The population is modified in-place to store fitness and objective values

        Warnings
        --------
        - Parallel evaluation is not yet available (parameter is ignored with a warning)
        - All population fitness/objective attributes are modified in-place

        See Also
        --------
        objective : The problem-specific objective function to implement
        constraint_handler : Handles constraint penalties via :meth:`ConstraintHandler.penalty`
        """

        if parallel:
            logger.warning("Parallel fitness computing not available at the moment. Ignoring parallel option.")

        logger.debug("Calculating fitness of the population...")
        fitness = population.fitness
        objective = population.objective
        solutions = population.decode()
        genotypes = population.genotype_matrix

        if not self.recalculate and np.all(population.fitness_calculated == 1):
            logger.debug("Fitness was not calculated. Every individual is duplicated.")
            return population.fitness

        if self.recalculate:
            fitness_mask = np.ones(population.pop_size, dtype=bool)
        else:
            fitness_mask = population.fitness_calculated == 0

        # Penalty is always vectorized. We use the genotype instead of the decoded solutions
        penalty_vector = self.constraint_handler.penalty(genotypes[fitness_mask])

        if self.vectorized:
            if isinstance(solutions, np.ndarray):
                solutions = solutions[fitness_mask]
            else:
                solutions = [solutions[i] for i, include_value in enumerate(fitness_mask) if include_value]

            objective_values = self.objective(solutions)
            fitness_values = self.factor * (objective_values - penalty_vector)
            fitness[fitness_mask] = fitness_values
            objective[fitness_mask] = objective_values
        else:
            # Expand the penalty to have the size `pop_size`
            penalty_vector_aux = np.zeros(population.pop_size)
            penalty_vector_aux[fitness_mask] = penalty_vector
            penalty_vector = penalty_vector_aux

            for idx, (solution, do_calculation) in enumerate(zip(solutions, fitness_mask)):
                if self.recalculate or do_calculation:
                    objective_value = self.objective(solution)
                    fitness_value = self.factor * (objective_value - penalty_vector[idx])

                    fitness[idx] = fitness_value
                    objective[idx] = objective_value

        self.counter += np.count_nonzero(fitness_mask)
        population.fitness_calculated = np.ones_like(fitness_mask)

        # Write the fitness and objective values in-place
        population.fitness = fitness
        population.objective = objective

        logger.debug("Done calculating the fitness.")
        return fitness

    @abstractmethod
    def objective(self, solution: Any) -> VectorLike | ScalarLike:
        """
        Evaluate the objective function for one or more solutions.

        This abstract method must be implemented by subclasses to define
        the problem-specific fitness evaluation logic.

        Parameters
        ----------
        solution : Any
            A single solution or population of solutions. The exact type and format
            depend on the problem definition and the vectorization setting:

            - If `self.vectorized` is False: a single decoded solution
            - If `self.vectorized` is True: a collection of solutions (typically ndarray or list)

        Returns
        -------
        objective_value : float or ndarray
            The objective value(s):

            - If `self.vectorized` is False: returns a scalar float
            - If `self.vectorized` is True: returns an ndarray of shape (n_solutions,)

        Notes
        -----
        This is an abstract method and must be implemented by subclasses.
        The method should NOT apply constraint penalties or mode adjustments;
        that is handled by :meth:`fitness`.

        Examples
        --------
        For a minimization problem with scalar solutions:

        >>> class SphereFunction(ObjectiveFunc):
        ...     def objective(self, solution):
        ...         return np.sum(solution ** 2)

        For a vectorized version:

        >>> class SphereVectorized(ObjectiveFunc):
        ...     def objective(self, solutions):  # solutions is 2D array
        ...         return np.sum(solutions ** 2, axis=1)
        """

    def repair_solution(self, solution: MatrixLike) -> MatrixLike:
        """
        Apply constraint repair to make a solution feasible.

        This method delegates to the constraint handler's repair mechanism.
        It transforms invalid solutions into ones satisfying problem restrictions.

        Parameters
        ----------
        solution : MatrixLike
            A solution vector or matrix that may violate problem constraints.

        Returns
        -------
        repaired_solution : MatrixLike
            A modified version of the input solution that satisfies all constraints.
            May be the same as input if already feasible or if repair is not applicable.

        See Also
        --------
        constraint_handler : The underlying :class:`ConstraintHandler` performing the repair
        """

        return self.constraint_handler.repair_solution(solution)

    def restart(self):
        """
        Reset the evaluation counter to zero.

        This is useful when starting a new optimization run to get accurate
        counts of fitness evaluations for the new run.

        Notes
        -----
        This only resets the `counter` attribute. The constraint handler
        and other settings are not modified.
        """
        self.counter = 0

    def get_state(self) -> dict:
        """
        Serialize the objective function state to a dictionary.

        Returns
        -------
        state : dict
            Dictionary containing:

            - 'class_name': str
                The class name of this objective function
            - 'name': str
                The display name of the objective function
            - 'constraint': dict
                Serialized state of the constraint handler
            - Additional items from :meth:`ParametrizableMixin.get_params`

        Returns
        -------
        state : dict
            Complete state representation suitable for serialization or reconstruction.

        See Also
        --------
        ParametrizableMixin.get_params : Returns parameter configuration
        ConstraintHandler.get_state : Returns constraint handler state
        """
        data = {"class_name": self.__class__.__name__, "name": self.name, "constraint": self.constraint_handler.get_state(), **self.get_params()}

        return data


class VectorObjectiveFunc(ObjectiveFunc):
    """
    Objective function specialized for vector-based optimization problems.

    This class extends :class:`ObjectiveFunc` for problems where solutions are
    continuous vectors with specified bounds. It automatically manages bound
    constraints and can integrate additional problem-specific constraints.

    The class combines the base objective functionality with automatic bound
    enforcement. Bounds are enforced via :class:`ClipBoundConstraint` which is
    composed with any additional constraint handler provided by the user.

    Parameters
    ----------
    dimension : int
        The dimensionality of solution vectors. All solutions must have
        exactly this many components.
    lower_bound : float
        Lower bound constraint for all vector components.
    upper_bound : float
        Upper bound constraint for all vector components.
    constraint_handler : ConstraintHandler, optional
        Additional constraint handler for problem-specific restrictions.
        Will be composed with bound constraints. Default is None.
    mode : str, optional
        Optimization direction: 'max' or 'min'. Default is 'max'.
    name : str, optional
        Display name for the objective function. Default is 'Some function'.
    vectorized : bool, optional
        Whether the objective method processes entire populations at once. Default is False.
    recalculate : bool, optional
        Whether to recalculate fitness for all individuals. Default is False.
    **kwargs
        Additional keyword arguments passed to parent class.

    Attributes
    ----------
    dimension : int
        Dimensionality of solution vectors.
    lower_bound : float
        Lower bound for all components.
    upper_bound : float
        Upper bound for all components.

    Notes
    -----
    Bound constraints are always applied, even if no additional constraint
    handler is specified. The actual constraint handler used will be either:

    - ``ClipBoundConstraint(dimension, lower_bound, upper_bound)`` if no additional handler
    - ``CompositeConstraint([user_handler, bound_handler])`` if a handler is provided

    Examples
    --------
    Define a simple sphere function optimization problem:

    >>> def sphere_obj(x):
    ...     return np.sum(x ** 2)
    >>> 
    >>> obj_func = VectorObjectiveFunc(
    ...     dimension=10,
    ...     lower_bound=-5.0,
    ...     upper_bound=5.0,
    ...     mode='min',
    ...     name='Sphere Function'
    ... )
    >>> obj_func.objective = sphere_obj

    See Also
    --------
    ObjectiveFunc : Base class with general objective function interface
    ClipBoundConstraint : Enforces box constraints on vectors
    CompositeConstraint : Combines multiple constraint handlers
    """

    def __init__(
        self,
        dimension: int,
        lower_bound: float,
        upper_bound: float,
        constraint_handler: Optional[ConstraintHandler] = None,
        mode: str = "max",
        name: str = "Some function",
        vectorized: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Initialize a VectorObjectiveFunc instance.

        Parameters
        ----------
        dimension : int
            Number of dimensions in solution vectors.
        lower_bound : float
            Lower bound for all vector components.
        upper_bound : float
            Upper bound for all vector components.
        constraint_handler : ConstraintHandler, optional
            Additional constraints beyond bounds. Default is None.
        mode : str, optional
            Optimization mode ('max' or 'min'). Default is 'max'.
        name : str, optional
            Display name for the objective. Default is 'Some function'.
        vectorized : bool, optional
            Whether objective method is vectorized. Default is False.
        recalculate : bool, optional
            Whether to recalculate all fitness values. Default is False.
        **kwargs
            Additional arguments for the parent class.
        """

        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        bound_constraint_handler = ClipBoundConstraint(dimension, lower_bound, upper_bound)
        if constraint_handler is None:
            constraint_handler = bound_constraint_handler
        else:
            constraint_handler = CompositeConstraint([constraint_handler, bound_constraint_handler])

        super().__init__(constraint_handler=constraint_handler, mode=mode, name=name, vectorized=vectorized, recalculate=recalculate, **kwargs)

    def add_parameter_constraints(self, parameter_extending_encoding: ParameterExtendingEncoding, param_handlers: dict[str, ConstraintHandler]):
        """
        Add constraints for problem parameters via parameter-extending encoding.

        This method integrates parameter-dependent constraints by wrapping the
        existing constraint handler with an :class:`ExtendedConstraintHandler`.
        Used when optimization parameters have associated constraints that vary
        with problem parameters.

        Parameters
        ----------
        parameter_extending_encoding : ParameterExtendingEncoding
            The encoding scheme that maps between solutions and extended parameters.
        param_handlers : dict[str, ConstraintHandler]
            Mapping of parameter names to their associated constraint handlers.

        Raises
        ------
        AssertionError
            If the existing constraint handler is already an ExtendedConstraintHandler
            and its parameter handler keys do not match the provided keys.

        Notes
        -----
        This method modifies the constraint handler in-place by wrapping it
        with an ExtendedConstraintHandler.

        See Also
        --------
        ParameterExtendingEncoding : Handles encoding with problem parameters
        ExtendedConstraintHandler : Manages parameter-dependent constraints
        """
        if isinstance(self.constraint_handler, ExtendedConstraintHandler):
            assert self.constraint_handler.param_handler_dict.keys() == param_handlers.keys()

        base_constraint_handler = self.constraint_handler

        self.constraint_handler = ExtendedConstraintHandler(
            solution_handler=base_constraint_handler, param_handler_dict=param_handlers, encoding=parameter_extending_encoding
        )


class NullObjectiveFunc(ObjectiveFunc):
    """
    A no-operation objective function for testing and initialization.

    This class implements a trivial objective function that always returns 0.
    Useful for testing components that depend on objective functions without
    introducing meaningful optimization logic.

    This class should not be used for actual optimization; it exists solely
    to satisfy interface requirements in testing or placeholder scenarios.

    Attributes
    ----------
    name : str
        Always set to 'Null objective'.

    Notes
    -----
    The objective method returns 0 for any input, regardless of dimension
    or type of solution provided.
    """

    def __init__(self, **kwargs):
        """
        Initialize a NullObjectiveFunc instance.

        Parameters
        ----------
        **kwargs
            Arguments passed to parent :class:`ObjectiveFunc`, name is overridden
            to 'Null objective'.
        """
        super().__init__(name="Null objective", **kwargs)

    def objective(self, _) -> VectorLike | ScalarLike:
        """
        Return zero for any input.

        Parameters
        ----------
        _ : Any
            Any input is ignored.

        Returns
        -------
        objective_value : int
            Always returns 0.
        """
        return 0


class ObjectiveFromLambda(ObjectiveFunc):
    """
    Objective function wrapper for callable-based problem definitions.

    This class allows users to define objective functions using plain Python
    callables (functions or lambdas) without creating a full subclass.
    The callable is invoked with the solution and any stored keyword arguments.

    Parameters
    ----------
    obj_func : Callable
        The objective function to wrap. Must accept a solution as its first argument.
        Additional arguments can be passed via keyword arguments stored in this class.
    constraint_handler : ConstraintHandler, optional
        Constraint enforcement strategy. Default is None (no constraints).
    mode : str, optional
        Optimization mode: 'max' for maximization, 'min' for minimization. Default is 'max'.
    name : str, optional
        Display name for the objective function. If not provided, uses the
        function name from `obj_func.__name__`. Default is None.
    vectorized : bool, optional
        Whether `obj_func` accepts vectorized inputs (populations). Default is False.
    recalculate : bool, optional
        Whether to recalculate all fitness values on each evaluation. Default is False.
    **kwargs
        Additional keyword arguments to pass to `obj_func` at evaluation time.

    Attributes
    ----------
    obj_func : Callable
        The wrapped objective function.

    Examples
    --------
    Simple scalar optimization:

    >>> def sphere(x):
    ...     return np.sum(x ** 2)
    >>> obj = ObjectiveFromLambda(sphere, mode='min', name='Sphere')

    With custom parameters:

    >>> def rosenbrock(x, a=1, b=100):
    ...     return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    >>> obj = ObjectiveFromLambda(rosenbrock, mode='min', a=1, b=100)

    Vectorized evaluation:

    >>> def sphere_vec(solutions):  # solutions is 2D array
    ...     return np.sum(solutions ** 2, axis=1)
    >>> obj = ObjectiveFromLambda(sphere_vec, mode='min', vectorized=True)

    See Also
    --------
    ObjectiveFunc : Base class for custom objective implementations
    """

    def __init__(
        self,
        obj_func: Callable,
        constraint_handler: Optional[ConstraintHandler] = None,
        mode: str = "max",
        name: Optional[str] = None,
        vectorized: bool = False,
        recalculate: bool = False,
        **kwargs,
    ):
        """
        Initialize an ObjectiveFromLambda instance.

        Parameters
        ----------
        obj_func : Callable
            The objective function to wrap. Called with (solution, **kwargs).
        constraint_handler : ConstraintHandler, optional
            Constraint handler for the problem. Default is None.
        mode : str, optional
            Optimization mode ('max' or 'min'). Default is 'max'.
        name : str, optional
            Display name. Uses `obj_func.__name__` if not provided. Default is None.
        vectorized : bool, optional
            Whether obj_func is vectorized. Default is False.
        recalculate : bool, optional
            Whether to recalculate fitness always. Default is False.
        **kwargs
            Additional arguments passed to obj_func at evaluation time.
        """

        if name is None:
            name = obj_func.__name__

        self.obj_func = obj_func

        super().__init__(constraint_handler=constraint_handler, mode=mode, name=name, vectorized=vectorized, recalculate=recalculate, **kwargs)

    def objective(self, solution: Any) -> VectorLike | ScalarLike:
        """
        Evaluate the wrapped objective function.

        Delegates to the stored callable, passing any stored keyword arguments.

        Parameters
        ----------
        solution : Any
            The solution to evaluate. Format depends on vectorization setting.

        Returns
        -------
        objective_value : float or ndarray
            Result from calling `self.obj_func(solution, **self.current_kwargs)`.

        Notes
        -----
        Uses `self.current_kwargs` (from ParametrizableMixin) to pass stored
        keyword arguments to the wrapped function at evaluation time.
        """
        return self.obj_func(solution, **self.current_kwargs)
