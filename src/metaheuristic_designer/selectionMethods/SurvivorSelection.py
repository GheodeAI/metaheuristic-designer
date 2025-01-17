from __future__ import annotations
import enum
from enum import Enum
from ..Population import Population
from ..ParamScheduler import ParamScheduler
from ..SelectionMethod import SelectionMethod
from .survivor_selection_functions import *


class SurvSelMethod(Enum):
    ELITISM = enum.auto()
    COND_ELITISM = enum.auto()
    GENERATIONAL = enum.auto()
    ONE_TO_ONE = enum.auto()
    PROB_ONE_TO_ONE = enum.auto()
    MANY_TO_ONE = enum.auto()
    PROB_MANY_TO_ONE = enum.auto()
    MU_PLUS_LAMBDA = enum.auto()
    MU_COMMA_LAMBDA = enum.auto()
    CRO = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in surv_method_map:
            raise ValueError(f'Parent selection method "{str_input}" not defined')

        return surv_method_map[str_input]


surv_method_map = {
    "elitism": SurvSelMethod.ELITISM,
    "condelitism": SurvSelMethod.COND_ELITISM,
    "generational": SurvSelMethod.GENERATIONAL,
    "nothing": SurvSelMethod.GENERATIONAL,
    "one-to-one": SurvSelMethod.ONE_TO_ONE,
    "hillclimb": SurvSelMethod.ONE_TO_ONE,
    "prob-one-to-one": SurvSelMethod.PROB_ONE_TO_ONE,
    "probhillclimb": SurvSelMethod.PROB_ONE_TO_ONE,
    "many-to-one": SurvSelMethod.MANY_TO_ONE,
    "localsearch": SurvSelMethod.MANY_TO_ONE,
    "prob-many-to-one": SurvSelMethod.PROB_MANY_TO_ONE,
    "problocalsearch": SurvSelMethod.PROB_MANY_TO_ONE,
    "(m+n)": SurvSelMethod.MU_PLUS_LAMBDA,
    "keepbest": SurvSelMethod.MU_PLUS_LAMBDA,
    "(m,n)": SurvSelMethod.MU_COMMA_LAMBDA,
    "keepoffsping": SurvSelMethod.MU_COMMA_LAMBDA,
    "cro": SurvSelMethod.CRO,
}


class SurvivorSelection(SelectionMethod):
    """
    Survivor selection methods.

    Selects the individuals that will remain for the next generation of our algorithm.

    Parameters
    ----------
    method: str
        Strategy used in the selection process.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the behaviour of the selection method.
    padding: bool, optional
        Whether to fill the entire list of selected individuals to match the size of the original one.
    name: str, optional
        The name that will be assigned to this selection method.
    """

    def __init__(
        self,
        method: str,
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor for the SurvivorSelection class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = SurvSelMethod.from_str(method)

    def select(self, population: Population, offspring: Population) -> Population:
        new_population = None
        full_idx = None
        population_fitness = copy(population.fitness)
        offspring_fitness = copy(offspring.fitness)

        if self.method == SurvSelMethod.ELITISM:
            full_idx = elitism(population_fitness, offspring_fitness, self.params["amount"])

        elif self.method == SurvSelMethod.COND_ELITISM:
            full_idx = cond_elitism(population_fitness, offspring_fitness, self.params["amount"])

        elif self.method == SurvSelMethod.GENERATIONAL:
            self.last_selection_idx = range(len(population), len(offspring))
            new_population = offspring

        elif self.method == SurvSelMethod.ONE_TO_ONE:
            if population.pop_size == offspring.pop_size == 1:
                choose_new_population = population_fitness < offspring_fitness
                full_idx = np.array([choose_new_population.squeeze()], dtype=int)
            else:
                full_idx = one_to_one(population_fitness, offspring_fitness)

        elif self.method == SurvSelMethod.PROB_ONE_TO_ONE:
            if population.pop_size == offspring.pop_size == 1:
                choose_new_population = population_fitness < offspring_fitness or RAND_GEN.random() < self.params["p"]
                full_idx = np.array([choose_new_population.squeeze()], dtype=int)
            else:
                full_idx = prob_one_to_one(population_fitness, offspring_fitness, self.params["p"])

        elif self.method == SurvSelMethod.MANY_TO_ONE:
            full_idx = many_to_one(population_fitness, offspring_fitness)

        elif self.method == SurvSelMethod.PROB_MANY_TO_ONE:
            full_idx = prob_many_to_one(population_fitness, offspring_fitness, self.params["p"])

        elif self.method == SurvSelMethod.MU_PLUS_LAMBDA:
            full_idx = lamb_plus_mu(population_fitness, offspring_fitness)

        elif self.method == SurvSelMethod.MU_COMMA_LAMBDA:
            full_idx = lamb_comma_mu(population_fitness, offspring_fitness)

        elif self.method == SurvSelMethod.CRO:
            full_idx = cro_selection(
                population_fitness,
                offspring_fitness,
                self.params["Fd"],
                self.params["Pd"],
                self.params["attempts"],
                self.params["maxPopSize"],
            )

        if new_population is None:
            self.last_selection_idx = full_idx
            new_population = population.join(offspring).take_selection(full_idx)

        return new_population
