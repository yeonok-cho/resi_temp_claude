"""
RMT-based per-channel (temp / resist) profile anomaly detector.

For each channel, fits a PCA model on baseline residual vectors and uses
the Marchenko-Pastur bulk edge to partition the eigenspace into a
low-rank "signal" subspace (common normal-variation modes) and a
high-dimensional "noise" subspace (random chip-to-chip noise).

The per-chip anomaly score is a chi-square statistic over the noise
subspace, with a theoretically calibrated threshold from chi2.ppf.
"""
from .config import MPConfig
from .base_profile import BaseProfile, extract_channel_residual
from .mp_detector import MPProfileDetector
