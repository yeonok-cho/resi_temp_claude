"""
Configuration for the RMT-based per-channel profile anomaly detector.
"""
from dataclasses import dataclass


@dataclass
class MPConfig:
    """
    Parameters for the PCA + Marchenko-Pastur anomaly score detector.

    One instance controls both the temp and resist channel models;
    they are trained independently but share the same hyperparameters.
    """
    input_length: int = 25           # per-channel downsample length (p)
    baseline_wafers: int = 10        # wafers to buffer before fitting
    alert_alpha: float = 0.01        # false-alarm rate: threshold = chi2.ppf(1-alpha, k_noise)
    min_noise_rank: int = 2          # floor on noise-subspace dimension (prevents rank-0 edge case)
    retrain_interval_wafers: int = 0 # periodic retrain (0 = train once and freeze)
