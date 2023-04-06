from __future__ import annotations
from enum import Enum
from .ParamScheduler import ParamScheduler


class SurvSelMethod(Enum):
    ELITISM = 1
    COND_ELITISM = 2
    GENERATIONAL = 3
    ONE_TO_ONE = 4
    MU_PLUS_LAMBDA = 5
    MU_COMMA_lAMBDA = 6

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in surv_method_map:
            raise ValueError(f"Parent selection method \"{str_input}\" not defined")

        return surv_method_map[str_input]


surv_method_map = {
    "elitism": SurvSelMethod.ELITISM,
    "condelitism": SurvSelMethod.COND_ELITISM,
    "generational": SurvSelMethod.GENERATIONAL,
    "nothing": SurvSelMethod.GENERATIONAL,
    "one-to-one": SurvSelMethod.ONE_TO_ONE,
    "(m+n)": SurvSelMethod.MU_PLUS_LAMBDA,
    "(m,n)": SurvSelMethod.MU_COMMA_lAMBDA
}


class SurvivorSelection:
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None, padding: bool = False):
        """
        Constructor for the SurvivorSelection class
        """

        if name is None:
            self.name = method

        self.method = SurvSelMethod.from_str(method)

        self.param_scheduler = None
        if params is None:
            self.params = {"amount": 10, "p": 0.1}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params

    def __call__(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:
        """
        Shorthand for calling the 'select' method
        """

        return self.select(popul, offspring)

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

            if "amount" in self.params:
                self.params["amount"] = round(self.params["amount"])

    def select(self, popul: List[Individual], offspring: List[Individual]) -> List[Individual]:
        """
        Takes a population with its offspring and returns the individuals that survive
        to produce the next generation.
        """

        result = []
        if self.method == SurvSelMethod.ELITISM:
            result = elitism(popul, offspring, self.params["amount"])

        elif self.method == SurvSelMethod.COND_ELITISM:
            result = cond_elitism(popul, offspring, self.params["amount"])

        elif self.method == SurvSelMethod.GENERATIONAL:
            result = offspring

        elif self.method == SurvSelMethod.ONE_TO_ONE:
            result = one_to_one(popul, offspring)

        elif self.method == SurvSelMethod.MU_PLUS_LAMBDA:
            result = lamb_plus_mu(popul, offspring)

        elif self.method == SurvSelMethod.MU_COMMA_lAMBDA:
            result = lamb_comma_mu(popul, offspring)

        return result


def one_to_one(popul, offspring):
    new_population = []
    for parent, child in zip(popul, offspring):
        if child.fitness > parent.fitness:
            new_population.append(child)
        else:
            new_population.append(parent)

    if len(offspring) < len(popul):
        n_leftover = len(offspring) - len(popul)
        new_population += popul[n_leftover:]

    return new_population


def elitism(popul, offspring, amount):
    selected_offspring = sorted(offspring, reverse=True, key=lambda x: x.fitness)[:len(popul) - amount]
    best_parents = sorted(popul, reverse=True, key=lambda x: x.fitness)[:amount]

    return best_parents + selected_offspring


def cond_elitism(popul, offspring, amount):
    best_parents = sorted(popul, reverse=True, key=lambda x: x.fitness)[:amount]
    new_offspring = sorted(offspring, reverse=True, key=lambda x: x.fitness)[:len(popul)]
    best_offspring = new_offspring[:amount]

    for idx, val in enumerate(best_parents):
        if val.fitness > best_offspring[0].fitness:
            new_offspring.pop()
            new_offspring = [val] + new_offspring

    return new_offspring


def lamb_plus_mu(popul, offspring):
    population = popul + offspring
    return sorted(population, reverse=True, key=lambda x: x.fitness)[:len(popul)]


def lamb_comma_mu(popul, offspring):
    return sorted(offspring, reverse=True, key=lambda x: x.fitness)[:len(popul)]
