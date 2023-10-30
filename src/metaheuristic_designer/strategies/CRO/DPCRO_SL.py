from __future__ import annotations
import random
from typing import Union, List
import numpy as np
from ...ParamScheduler import ParamScheduler
from .CRO_SL import CRO_SL


class DPCRO_SL(CRO_SL):
    """
    Dynamic Probabilistic Coral Reef Optimization with Substrate Layers.

    Published in:
    - Pérez-Aracil, Jorge, et al. "New Probabilistic, Dynamic Multi-Method Ensembles for Optimization Based on the CRO-SL." Mathematics 11.7 (2023): 1666.

    Original implementation in https://github.com/jperezaracil/PyCROSL/
    """

    def __init__(
        self,
        pop_init: Initializer,
        operator_list: List[Operator],
        params: Union[ParamScheduler, dict] = {},
        name: str = "DPCRO-SL",
    ):
        super().__init__(pop_init, operator_list, params=params, name=name)

        self.group_subs = params["group_subs"]
        self.dyn_method = params["dyn_method"]
        self.dyn_metric = params["dyn_metric"]
        self.dyn_steps = params["dyn_steps"]
        self.prob_amp = params["prob_amp"]

        self.operator_idx = random.choices(
            range(len(self.operator_list)), k=self.maxpopsize
        )
        self.operator_weight = [1 / len(operator_list)] * len(operator_list)

        if self.dyn_method == "success":
            self.operator_data = [[0] for i in operator_list]
        else:
            self.operator_data = [[] for i in operator_list]

        self.operator_metric_prev = [0 for i in operator_list]

        self.larva_count = [0 for i in operator_list]

        self.operator_w_history = []
        self.op_steps = 0
        self.operator_metric = [0] * len(operator_list)
        self.operator_history = []

    def perturb(self, parent_list, objfunc, **kwargs):
        offspring = []

        divided_population = [[] for i in self.operator_list]
        for idx, indiv in enumerate(parent_list):
            op_idx = self.operator_idx[idx]
            divided_population[op_idx].append(indiv)

        for idx, indiv in enumerate(parent_list):
            # Select operator
            op_idx = self.operator_idx[idx]

            op = self.operator_list[op_idx]

            new_parent_list = (
                divided_population[op_idx] if self.group_subs else parent_list
            )

            # Apply operator
            new_indiv = op(indiv, new_parent_list, objfunc, self.best, self.pop_init)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Collect data about each operator
            if self.dyn_method == "fitness" or self.dyn_method == "diff":
                self.operator_data[op_idx].append(new_indiv.fitness)

            # Add to offspring list
            offspring.append(new_indiv)

        # Update best solution
        current_best = max(offspring, key=lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best

        return offspring

    def select_individuals(self, population, offspring, **kwargs):
        offspring_ids = [i.id for i in offspring]
        new_population = self.selection_op(population, offspring)
        new_ids = [i.id for i in new_population]

        for idx, off_id in enumerate(offspring_ids):
            op_idx = self.operator_idx[idx]

            self.larva_count[op_idx] += 1

            if off_id in new_ids:
                self.operator_data[op_idx][0] += 1

        return new_population

    def update_params(self, **kwargs):
        progress = kwargs["progress"]

        self._generate_substrates(progress)
        super().update_params(progress=progress)

    def extra_step_info(self):
        print("\n\tSubstrate probability:")
        op_names = [i.name for i in self.operator_list]
        weights = [round(i, 6) for i in self.operator_weight]
        adjust = max([len(i) for i in op_names])
        for idx, val in enumerate(op_names):
            print(f"\t\t{val}:".ljust(adjust + 3, " ") + f"{weights[idx]}")

    def _operator_metric(self, data):
        result = 0

        # Choose what information to extract from the data gathered
        if len(data) > 0:
            if self.dyn_metric == "best":
                result = max(data)
            elif self.dyn_metric == "avg":
                result = sum(data) / len(data)
            elif self.dyn_metric == "med":
                data = sorted(data)
                if len(data) % 2 == 0:
                    result = (data[len(data) // 2 - 1] + data[len(data) // 2]) / 2
                else:
                    result = data[len(data) // 2]
            elif self.dyn_metric == "worse":
                result = min(data)

        return result

    def _operator_probability(self, values):
        # Normalization to avoid passing big values to softmax
        weight = np.array(values)
        weight_sum = np.abs(weight).sum()
        if weight_sum != 0:
            weight = weight / weight_sum
        else:
            weight = weight / (weight_sum + 1e-5)

        # softmax to convert to a probability distribution
        exp_vec = np.exp(weight)
        amplified_vec = exp_vec ** (1 / self.prob_amp)

        # if there are numerical error default repeat with a default value
        if (amplified_vec == 0).any() or not np.isfinite(amplified_vec).all():
            if not self.prob_amp_warned:
                print(
                    "Warning: the probability amplification parameter is too small, defaulting to prob_amp = 1"
                )
                self.prob_amp_warned = True
            prob = exp_vec / exp_vec.sum()
        else:
            prob = amplified_vec / amplified_vec.sum()

        # If probabilities get too low, equalize them
        if (prob <= 0.02 / len(values)).any():
            prob += 0.02 / len(values)
            prob = prob / prob.sum()

        return prob

    def _evaluate_operators(self):
        metric = 0

        # take reference data for the calculation of the difference of the next evaluation
        if self.dyn_method == "diff":
            full_data = [
                data_point for op_data in self.operator_data for data_point in op_data
            ]
            metric = self._operator_metric(full_data)

        # calculate the value of each operator with the data gathered
        for idx, s_data in enumerate(self.operator_data):
            if self.dyn_method == "success":
                # obtain the rate of success of the larvae
                if self.larva_count[idx] > 0:
                    self.operator_metric[idx] = s_data[0] / self.larva_count[idx]
                else:
                    self.operator_metric[idx] = 0

                # Reset data for nex iteration
                self.operator_data[idx] = [0]
                self.larva_count[idx] = 0

            elif self.dyn_method == "fitness" or self.dyn_method == "diff":
                # obtain the value used in the evaluation of the operator
                self.operator_metric[idx] = self._operator_metric(s_data)

                # Calculate the difference of the fitness in this generation to the previous one and
                # store the current value for the next evaluation
                if self.dyn_method == "diff":
                    self.operator_metric[idx] = (
                        self.operator_metric[idx] - self.operator_metric_prev[idx]
                    )
                    self.operator_metric_prev[idx] = metric

                # Reset data for next iteration
                self.operator_data[idx] = []

    def _generate_substrates(self, progress=0):
        n_operators = len(self.operator_list)

        if progress > self.op_steps / self.dyn_steps:
            self.op_steps += 1
            self._evaluate_operators()

        # Assign the probability of each operator
        self.operator_weight = self._operator_probability(self.operator_metric)
        self.operator_w_history.append(self.operator_weight)

        # Choose each operator with the weights chosen
        self.operator_idx = random.choices(
            range(n_operators), weights=self.operator_weight, k=self.maxpopsize
        )

        # save the evaluation of each operator
        self.operator_history.append(np.array(self.operator_metric))
