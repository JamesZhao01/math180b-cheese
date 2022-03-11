from math import floor, log
from collections import defaultdict
import random
import tqdm
import numpy as np


def unif(a, b):
    return random.random() * (b - a) + a


def dice():
    return random.randint(1, 6)


def geom(p):
    return floor(log(random.random()) / log(1 - p)) + 1


def ctheads(x):
    return sum([random.randint(0, 1) for i in range(x)])


ct = 1000000


def q1():
    print("=== q1 ===")
    print("q1ii)")
    success = 0
    for i in tqdm.tqdm(range(ct)):
        heads = ctheads(dice())
        if heads == 5:
            success += 1
    print(f"\tP(X = 5): {success / ct}")
    sums = 0
    for i in tqdm.tqdm(range(ct)):
        sums += ctheads(dice())
    print(f"\tE[X]: {sums / ct}")


def geom_test():
    print("=== q2 ===")
    outs = [0] * 10
    p = 0.5
    for i in range(ct):
        v = geom(p=0.5)
        if v < len(outs):
            outs[v] += 1
    print("Empirical: ", [t/ct for t in outs])
    print("Expected: ", [(1 - p) ** (i - 1) * p for i in range(len(outs))])


def q2():
    p = 0.1
    n = 14
    observe = [0] * n
    for i in tqdm.tqdm(range(ct)):
        a = geom(p)
        b = geom(p)
        z = a + b
        if z == n:
            observe[a] += 1
    print("Empirical", [round(t / sum(observe), 3) for t in observe])
    print("Expected", [round(1 / (n - 1), 3) if idx >= 1 and idx <
          n else 0 for idx, t in enumerate(observe)])


def q3():
    lamb = 444.0
    samples = [np.random.poisson(lam=lamb) for i in range(ct)]
    ct_dict = defaultdict(int)
    for v in tqdm.tqdm(samples):
        ct_dict[v] += 1
    odd_keys, odd_vals = zip(
        *[(k, v) for k, v in ct_dict.items() if k % 2 == 1])
    ct_odd = sum(odd_vals)
    sum_odd_vals = sum([k * v for k, v in zip(odd_keys, odd_vals)])

    print(f"Empirical E[X | X is odd]: {sum_odd_vals / ct_odd}")
    print(
        f"Analytical E[X | X is odd]: {lamb * (np.exp(lamb) + np.exp(-lamb)) / (np.exp(lamb) - np.exp(-lamb))}")


def q4():
    success = 0
    for i in tqdm.tqdm(range(ct)):
        d1 = dice()
        if d1 >= 4:
            continue
        d2 = dice()
        while d1 + d2 != 4 and d1 + d2 != 7:
            d2 = dice()
        if d1 + d2 == 4:
            success += 1
    print(f"Empirical P(Ends at 4) = {success / ct}")


def q5():
    mean = 2.24
    sigma = 3.35
    sigma_square = sigma ** 2
    lamb = 4

    samples = np.random.poisson(lamb, ct)
    sums = [np.sum(np.random.normal(mean, sigma, sample))
            for sample in tqdm.tqdm(samples)]
    print(len(sums), sums[0])
    print(f"Empirical E(X): {np.mean(sums)}, Empirical Var(X): {np.var(sums)}")
    print(
        f"Analytical E(X): {mean * lamb}, Empirical Var(X): {lamb * sigma_square + lamb * mean**2}")


def q6():
    lamb = 19
    n = 24
    x = np.random.exponential(lamb, ct)
    y = np.random.exponential(lamb, ct)
    xs = []
    samples = 0
    for sx, sy in zip(x, y):
        if abs(sx + sy - n) > 0.1:
            continue
        samples += 1
        xs.append(sx)
    xs.sort()
    lo = min(xs)
    hi = max(xs)
    increment = (hi - lo) / 10
    print(f"lo: {lo}, hi: {hi}")
    for i in range(10):
        lobound = i * increment
        hibound = (i + 1) * increment
        print(
            f"{i}th percentile:| {len([t for t in xs if t >= lobound and t <= hibound])}")


def q7():
    print("=== q7 ===")
    success = 0
    for i in tqdm.tqdm(range(ct)):
        u = unif(0, 1)
        t = unif(0, u)
        if t > 0.5:
            success += 1
    print(f"{success / ct}")


q6()
