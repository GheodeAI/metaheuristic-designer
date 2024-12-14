import pytest
import numpy as np
from metaheuristic_designer.selectionMethods.survivor_selection_functions import * 
import metaheuristic_designer as mhd

mhd.reset_seed(0)

example_fitness = np.array([-10, -2, -1, 0, 0, 1, 2, 10])
