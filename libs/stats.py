import numpy as np

def expected_value(value_fn, cdf_fn, range):
    delta = range[1] - range[0]
    result = sum(value_fn(x) * (cdf_fn(x + delta) - cdf_fn(x)) for x in range)
    return result

def dist_product(value_fn, cdf_fn, range):
    delta = range[1] - range[0]
    return [value_fn(x) * (cdf_fn(x + delta) - cdf_fn(x)) for x in range]

def expected_value_sim(value_fn, samples):
    sum_ = 0
    cnt_ = 0
    for x in samples:
        sum_ += value_fn(x)
        cnt_ += 1
    return sum_/cnt_

def test_expected_value_with_normal_dist():
    from statistics import NormalDist
    dist = NormalDist(mu=0, sigma=1)
    expected_value(
                value_fn=lambda md: 1,
                cdf_fn=dist.cdf,
                range=np.arange(-10, +10., 0.0001),
            )

def test_expected_value_sim_with_normal_dist():
    from statistics import NormalDist
    dist = NormalDist(mu=0, sigma=1)
    expected_value_sim(
                value_fn=lambda md: 1,
                samples=dist.samples(1_000_000),
            )
