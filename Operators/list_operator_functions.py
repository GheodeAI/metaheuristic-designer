import math
import random
import numpy as np
import scipy as sp
import scipy.stats

def expand(input_list, n):
    new_values = [random.random() for i in n]
    result_list = input_list + new_values
    return result_list


def shrink_rand(input_list, n):
    idxs = random.choices(range(len(input_list)), k=n)
    result_list = [input_list[i] for i in idxs]

    return result_list

def shrink_right(input_list, n):
    return input_list[:-n]

def shrink_left(input_list, n):
    return input_list[n:]

