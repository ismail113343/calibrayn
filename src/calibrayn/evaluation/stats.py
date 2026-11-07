import random
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float


def bootstrap_interval(values, seed=0, resamples=2000, level=0.95):
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    average = fmean(values)
    if len(values) == 1:
        return Interval(average, average, average)
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    tail = (1.0 - level) / 2.0
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1.0 - tail) * resamples))]
    return Interval(average, low, high)
