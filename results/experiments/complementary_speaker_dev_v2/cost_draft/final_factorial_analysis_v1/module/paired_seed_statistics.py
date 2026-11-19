"""Shared paired-seed statistics; no data access or training."""
import math
import statistics
from scipy.stats import t

def paired_statistics(values):
    n = len(values)
    mean = statistics.mean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n) if n > 1 else None
    return dict(n=n, mean_pp=mean, sample_sd_pp=sd,
                ci95_pp=[mean - half, mean + half] if half is not None else None)
