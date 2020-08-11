from itertools import product, starmap
from functools import wraps
import random
import string


def random_string(n):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(n))


def capture_variables(**kwargs):
    return "  " + "\n  ".join(f"{name} := {value}" for name, value in kwargs.items())


def random_int_tuple(a, b, n):
    return tuple(random.randint(a, b) for _ in range(n))


def product_range(starts, ends=None):
    if ends is None:
        ends = starts
        starts = (0,) * len(ends)
    yield from product(*tuple(starmap(range, zip(starts, ends))))


def repeat(n):
    def wrapper(func):
        @wraps(func)
        def repeater(*args):
            for i in range(n):
                func(*args)

        return repeater

    return wrapper
