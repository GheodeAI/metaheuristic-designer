import random


def expand(input_list, n, method, maxlen, val_generator=None):
    result = input_list

    if val_generator is None:
        val_generator = random.random

    if len(input_list) < maxlen:
        if len(input_list) + n >= maxlen:
            n = maxlen - len(input_list)

        if method == "rand":
            result = expand_rand(input_list, n, val_generator)
        elif method == "right":
            result = expand_right(input_list, n, val_generator)
        elif method == "left":
            result = expand_left(input_list, n, val_generator)

    return result


def expand_rand(input_list, n, val_generator):
    for i in range(n):
        inject_idx = random.randrange(len(input_list))
        input_list = (
            input_list[:inject_idx] + [val_generator()] + input_list[inject_idx:]
        )

    return input_list + new_values


def expand_right(input_list, n, val_generator):
    new_values = [val_generator() for i in n]
    return input_list + new_values


def expand_left(input_list, n, val_generator):
    new_values = [val_generator() for i in n]
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
    n_idx_chosen = max(len(input_list) - n, 0)

    idxs = sorted(random.choices(range(len(input_list)), k=n_idx_chosen))
    result_list = [input_list[i] for i in idxs]

    return result_list


def shrink_right(input_list, n):
    return input_list[:-n]


def shrink_left(input_list, n):
    return input_list[n:]
