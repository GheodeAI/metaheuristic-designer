import math
import random
import numpy as np
import scipy as sp
import scipy.stats


def expand(input_list, n, method):
    result = input_list
    if method == "right":
        result = expand_right(input_list, n)
    elif method == "left":
        result = expand_left(input_list, n)
    
    return result

def expand_right(input_list, n):
    new_values = [random.random() for i in n]
    return input_list + new_values

def expand_left(input_list, n):
    new_values = [random.random() for i in n]
    return new_values + input_list


def shrink(input_list, n, method):
    result = input_list

    if method == "rand":
        result = shrink_rand(input_list, n)
    elif method == "right":
        result = shrink_right(input_list, n)
    elif method == "left":
        result = shrink_left(input_list, n)
    
    return result

def shrink_rand(input_list, n):
    idxs = random.choices(range(len(input_list)), k=n)
    result_list = [input_list[i] for i in idxs]

    return result_list

def shrink_right(input_list, n):
    return input_list[:-n]

def shrink_left(input_list, n):
    return input_list[n:]

