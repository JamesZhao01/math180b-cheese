from math import floor, log
from collections import defaultdict
import random
import tqdm
import numpy as np
from numpy.random import poisson, binomial, uniform, exponential, normal


def dice():
    return random.randint(1, 6)


ct = 1000000


def q2():
    p = 0.5
    lamb = 12
    pois = poisson(lam=lamb, size=ct)
    x = np.array([binomial(t, p) for t in pois])
    print(pois, x)
    cova = np.cov(lamb, x)
