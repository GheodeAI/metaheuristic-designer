from __future__ import annotations
from typing import Tuple, Any
from copy import copy
import numpy as np
from numpy import ndarray
from .utils import RAND_GEN
from .ObjectiveFunc import ObjectiveFunc
from .Encoding import Encoding, DefaultEncoding, ExtendedEncoding


class Population:
    """
    Individual that holds a tentative solution with its fitness.

    Parameters
    ----------
    objfunc: ObjectiveFunc
        The objective function to be maximized or minimized.
    genotype_matrix: ndarray
        The solutions that will form part of the population.
    encoding: Encoding, optional
        The encoding to be used when calculating the objective function.
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        genotype_matrix: ndarray,
        encoding: Encoding = None,
    ):
        """
        Constructor of the Individual class.
        """

        # Objective function
        self.objfunc = objfunc

        # Population of solutions
        self.genotype_matrix = genotype_matrix

        # Size of the population
        self.pop_size = genotype_matrix.shape[0]
        self.vec_size = genotype_matrix.shape[1]

        # Fitness of each individual in the population
        self.fitness = np.full(self.pop_size, -np.inf)
        self.fitness_calculated = np.zeros(self.pop_size)

        # Best solution found so far
        self.best = None
        self.best_fitness = None

        # Best inidividual in each spot of the population
        self.historical_best_matrix = genotype_matrix
        self.historical_best_fitness = np.full(self.pop_size, -np.inf)

        # Encoding to use
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

        self.index = -1

    def __len__(self):
        return self.genotype_matrix.shape[0]

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index <= len(self):
            return self.genotype_matrix[self.index - 1]
        raise StopIteration

    def __repr__(self):
        return (
            "Population{"
            f"\n\tobjfunc = {self.objfunc.name}"
            f"\n\tgenotype_matrix = {self.genotype_matrix}"
            f"\n\tpop_size = {self.pop_size}"
            f"\n\tvec_size = {self.vec_size}"
            f"\n\tfitness = {self.fitness}"
            f"\n\tfitness_calculated = {self.fitness_calculated}"
            f"\n\thistorical_best_matrix = {self.historical_best_matrix}"
            f"\n\thistorical_best_fitness = {self.historical_best_fitness}"
            f"\n\tbest = {self.best}"
            f"\n\tbest_fitness = {self.best_fitness}"
            "\n}"
        )

    def __copy__(self) -> Population:

        copied_pop = Population(self.objfunc, copy(self.genotype_matrix), encoding=self.encoding)
        copied_pop.fitness = copy(self.fitness)
        copied_pop.fitness_calculated = copy(self.fitness_calculated)
        copied_pop.historical_best_matrix = copy(self.historical_best_matrix)
        copied_pop.historical_best_fitness = copy(self.historical_best_fitness)
        copied_pop.best = copy(self.best)
        copied_pop.best_fitness = copy(self.best_fitness)

        return copied_pop

    def best_solution(self, decoded=False) -> Tuple[ndarray, float]:
        """
        Returns the best solution.

        Parameters
        ----------
        decoded: bool, optional
            Whether to return the raw vector of the best solution or it's decoded version.

        Returns
        -------
        best_solution : Tuple[ndarray, float]
            A pair of the best individual with its fitness.
        """

        best_fitness = self.best_fitness
        if self.objfunc.mode == "min":
            best_fitness *= -1

        best_solution = self.best
        if decoded:
            best_solution = self.encoding.decode(self.best[None, :])[0]

        return best_solution, best_fitness

    def update_genotype_matrix(self, genotype_matrix: ndarray) -> Population:
        """
        Replaces the solutions in the population with the ones inputted.

        Parameters
        ----------
        genotype_matrix: ndarray
            The set of solutions that will replace the ones that were in the population.

        Returns
        -------
        self: Population
        """

        if genotype_matrix.shape[1] != self.vec_size:
            raise ValueError("Individual vector size should not change when updating the population.")

        if len(genotype_matrix) != len(self.genotype_matrix):
            self.fitness = np.full(len(genotype_matrix), -np.inf)
            self.fitness_calculated = np.zeros(len(genotype_matrix))
            self.historical_best_fitness = np.full(len(genotype_matrix), -np.inf)
            self.historical_best_matrix = copy(genotype_matrix)
        else:
            self.fitness_calculated = np.all(self.genotype_matrix == genotype_matrix, axis=1)
        self.genotype_matrix = genotype_matrix
        self.pop_size = genotype_matrix.shape[0]

        return self

    def take_selection(self, selection_idx: ndarray) -> Population:
        """
        Takes a subset of the population given a mask.

        Parameters
        ----------
        selection_idx: ndarray
            An array of indices or a mask that indicate which individuals to take from the population.

        Returns
        -------
        selected_population: Population
            A copy of the population containing only the chosen individuals.
        """

        selected_genotype_matrix = copy(self.genotype_matrix[selection_idx, :])

        selected_pop = Population(self.objfunc, selected_genotype_matrix, encoding=self.encoding)
        selected_pop.fitness = copy(self.fitness[selection_idx])
        selected_pop.fitness_calculated = copy(self.fitness_calculated[selection_idx])
        selected_pop.historical_best_matrix = copy(self.historical_best_matrix[selection_idx, :])
        selected_pop.historical_best_fitness = copy(self.historical_best_fitness[selection_idx])
        selected_pop.best = copy(self.best)
        selected_pop.best_fitness = copy(self.best_fitness)

        return selected_pop

    def apply_selection(self, selected_pop: Population, selection_idx: ndarray) -> Population:
        """
        Replaces the chosen individuals from the input population to the current population.

        Parameters
        ----------
        selected_pop: Population
            Population where to take the individuals that will replace the ones in the population.
        selection_idx: ndarray
            An array of indices or a mask that indicate which individuals to take from the population.

        Returns
        -------
        self: Population
        """

        # population_copy = copy(self)
        self.genotype_matrix[selection_idx, :] = selected_pop.genotype_matrix
        self.fitness[selection_idx] = selected_pop.fitness
        self.fitness_calculated[selection_idx] = selected_pop.fitness_calculated
        self.historical_best_matrix[selection_idx, :] = selected_pop.historical_best_matrix
        self.historical_best_fitness[selection_idx] = selected_pop.historical_best_fitness

        # if selected_pop.best_fitness is None or (self.best_fitness is not None and self.best_fitness > selected_pop.best_fitness):
        if self.best is None or (selected_pop.best is not None and self.best_fitness < selected_pop.best_fitness):
            self.best = selected_pop.best
            self.best_fitness = selected_pop.best_fitness

        return self

    def take_slice(self, mask: ndarray) -> Population:
        """
        Takes a subset of the components in the population vectors.

        Parameters
        ----------
        mask: ndarray
            An array of indices or a mask that indicate which components to take from each vector in the population.

        Returns
        -------
        sliced_population: Population
            A copy of the population containing the masked individuals.
        """

        sliced_genotype_matrix = copy(self.genotype_matrix[:, mask])

        sliced_pop = Population(self.objfunc, sliced_genotype_matrix, encoding=self.encoding)
        sliced_pop.historical_best_matrix = copy(self.historical_best_matrix[:, mask])
        sliced_pop.historical_best_fitness = copy(self.historical_best_fitness)
        sliced_pop.fitness_calculated = copy(self.fitness_calculated)
        sliced_pop.fitness = copy(self.fitness)
        sliced_pop.best = copy(self.best)
        sliced_pop.best_fitness = copy(self.best_fitness)

        return sliced_pop

    def apply_slice(self, sliced_pop: Population, mask: ndarray) -> Population:
        """
        Apply the values of the population to a subset of the components of the population vectors.

        Parameters
        ----------
        sliced_pop: Population
            Population where to take the individuals from which we will take the components that will replace the ones in the
            current population.
        mask: ndarray
            An array of indices or a mask that indicate which components to take from each vector in the population.

        Returns
        -------
        self: Population
        """

        self.genotype_matrix[:, mask] = sliced_pop.genotype_matrix

        if self.best is None or (sliced_pop.best is not None and self.best_fitness < sliced_pop.best_fitness):
            self.best = sliced_pop.best
            self.best_fitness = sliced_pop.best_fitness

        return self

    @staticmethod
    def join_populations(population1, population2):
        """
        Concatenates the individuals in both populations into a new one.
        """

        joined_genotype_matrix = np.concatenate((population1.genotype_matrix, population2.genotype_matrix), axis=0)

        joined_pop = Population(population1.objfunc, joined_genotype_matrix, encoding=population1.encoding)
        joined_pop.historical_best_matrix = np.concatenate((population1.historical_best_matrix, population2.historical_best_matrix), axis=0)
        joined_pop.historical_best_fitness = np.concatenate((population1.historical_best_fitness, population2.historical_best_fitness))
        joined_pop.fitness_calculated = np.concatenate((population1.fitness_calculated, population2.fitness_calculated))
        joined_pop.fitness = np.concatenate((population1.fitness, population2.fitness))

        if population1.best is None or (population2.best is not None and population1.best_fitness < population2.best_fitness):
            joined_pop.best = population1.best
            joined_pop.best_fitness = population1.best_fitness
        else:
            joined_pop.best = population2.best
            joined_pop.best_fitness = population2.best_fitness

        return joined_pop

    def join(self, other_population: Population) -> Population:
        """
        Adds to the current population the individuals of the input population.

        Parameters
        ----------
        other_population: Population
            Population that will be concatenated with the current one.


        Returns
        -------
        joined_populations: Population
            A population containing both the individuals from the current population and the ones from the input population.
        """

        self.genotype_matrix = np.concatenate((self.genotype_matrix, other_population.genotype_matrix), axis=0)
        self.pop_size += other_population.genotype_matrix.shape[0]
        self.historical_best_matrix = np.concatenate((self.historical_best_matrix, other_population.historical_best_matrix), axis=0)
        self.historical_best_fitness = np.concatenate((self.historical_best_fitness, other_population.historical_best_fitness))
        self.fitness_calculated = np.concatenate((self.fitness_calculated, other_population.fitness_calculated), axis=0)
        self.fitness = np.concatenate((self.fitness, other_population.fitness))

        if self.best is None or (other_population.best is not None and self.best_fitness < other_population.best_fitness):
            self.best = other_population.best
            self.best_fitness = other_population.best_fitness

        return self

    def sort_population(self) -> Population:
        """
        Sorts the individuals by fitness.

        Returns
        -------
        self: Population
        """

        fitness_order = np.argsort(self.fitness)

        self.genotype_matrix = self.genotype_matrix[fitness_order, :]
        self.historical_best_matrix = self.historical_best_matrix[fitness_order, :]
        self.historical_best_fitness = self.historical_best_fitness[fitness_order]
        self.fitness_calculated = self.fitness_calculated[fitness_order]
        self.fitness = self.fitness[fitness_order]

        return self

    def update_best_from_parents(self, parents: Population) -> Population:
        """
        Updates the best fitness and best individual from an input population.

        Parameters
        ----------
        parents: Population

        Returns
        -------
        self: Population
        """

        if self.best is None or (parents.best is not None and self.best_fitness < parents.best_fitness):
            self.best = parents.best
            self.best_fitness = parents.best_fitness
        return self

    def update(self) -> Population:
        """
        Updates the best solution in the population.

        Returns
        -------
        self: Population
        """
        if self.best is None or np.any(self.best_fitness < self.fitness):
            best_idx = np.argmax(self.fitness)
            self.best = self.genotype_matrix[best_idx, :]
            self.best_fitness = self.fitness[best_idx]
        
        self.genotype_matrix = self.encoding.update(self)

        return self

    def repeat(self, amount: int = 2) -> Population:
        """
        Duplicates the individuals of the population.

        Parameters
        ----------
        amount: int, optional
            The amount of times to repeat the individuals in the population.

        Returns
        -------
        repeated_population: Population
        """

        genotype_matrix = np.tile(self.genotype_matrix, (amount, 1))
        return Population(self.objfunc, genotype_matrix, encoding=self.encoding)

    def calculate_fitness(self, parallel: bool = False, threads: int = 8) -> ndarray:
        """
        Calculates the fitness of the individual if it has not been calculated before

        Parameters
        ----------
        parallel: bool, optional
            Wheather to evaluate the individuals in the population in parallel.
        threads: int, optional
            Number of processes to use at once if calculating the fitness in parallel.

        Returns
        -------
        fitness: ndarray
        """

        prev_fitness = copy(self.fitness)
        self.fitness = self.objfunc.fitness(self, parallel=parallel, threads=threads)

        if len(prev_fitness) != len(self.fitness):
            self.historical_best_fitness = self.fitness
        else:
            improved_mask = prev_fitness < self.fitness
            self.historical_best_fitness[improved_mask] = self.fitness[improved_mask]
            self.historical_best_matrix[improved_mask, :] = self.genotype_matrix[improved_mask, :]

        if self.best is None or np.any(self.fitness > self.best_fitness):
            best_idx = np.argmax(self.fitness)
            self.best = self.genotype_matrix[best_idx]
            self.best_fitness = self.fitness[best_idx]

        return self

    def repair_solutions(self) -> Population:
        """
        Repairs the solutions in the population.

        Returns
        -------
        self: Population
        """

        for idx, indiv in enumerate(self.genotype_matrix):
            self.genotype_matrix[idx] = self.objfunc.repair_solution(indiv)

        return self

    def decode(self) -> Any:
        """
        Return the population passed through the decoding funciton defined in the encoding.

        Returns
        -------
        decoded_population: Any
        """

        return self.encoding.decode(self.genotype_matrix)

    def decode_params(self) -> Any:
        """
        Return the population passed through the decoding funciton defined in the encoding.

        Returns
        -------
        decoded_population: Any
        """

        if isinstance(self.encoding, ExtendedEncoding):
            return self.encoding.decode_params(self.genotype_matrix)
        else:
            return None

    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Parameters
        ----------
        show_best: bool, optional
            Save the best parent of this individual.

        Returns
        -------
        state: dict
            The current state of this individual.
        """

        data = {
            "genotype_matrix": self.genotype_matrix,
            "fitness": self.fitness,
            "historical_best_matrix": self.genotype_matrix,
            "historical_best_fitness": self.historical_best_fitness,
            "best": self.best,
            "best_fitness": self.best_fitness,
            "encoding": type(self.encoding).__name__,
        }

        return data
