"""
Feature extraction for HBM CoW bonding anomaly detection.

Extracts roughness and spike features from chip time-series residuals.
Roughness captures "how much does the signal shake" independently of
profile shape or level, making it insensitive to reference drift.
"""
from __future__ import annotations

import numpy as np

from .config import FeatureConfig, DEFAULT_CONFIG
from .data_models import ChipData, ChipFeatures


def compute_roughness(residual: np.ndarray) -> float:
    """
    Compute roughness as std of first-order differences of a residual signal.

    This is insensitive to overall level shifts and slowly-varying trends
    in the residual; it captures only high-frequency oscillation.
    """
    if len(residual) < 2:
        return 0.0
    diff = np.diff(residual)
    return float(np.std(diff))


def compute_spike(residual: np.ndarray) -> float:
    """Max absolute value of residual — sensitive to single-point outliers."""
    if len(residual) == 0:
        return 0.0
    return float(np.max(np.abs(residual)))


def extract_chip_features(
    chip: ChipData,
    temp_reference: np.ndarray,
    resist_reference: np.ndarray,
    resist_cluster: int = 0,
    config: FeatureConfig | None = None,
) -> ChipFeatures:
    """
    Extract roughness and spike features for one chip given reference profiles.

    Parameters
    ----------
    chip:
        The chip whose features are to be extracted.
    temp_reference:
        Reference temperature profile of shape (T,) from reference_manager.
    resist_reference:
        Reference resistance profile of shape (T,) — already selected for
        the correct bimodal cluster.
    resist_cluster:
        Which bimodal cluster this chip belongs to (0 or 1).
    config:
        Feature extraction configuration.
    """
    if config is None:
        config = DEFAULT_CONFIG.feature

    n = chip.n_points
    if n < config.min_points:
        return ChipFeatures(
            chip=chip,
            temp_roughness=0.0,
            resist_roughness=0.0,
            temp_spike=0.0,
            resist_spike=0.0,
            resist_cluster=resist_cluster,
        )

    # Align lengths (reference may differ by one point due to rounding)
    ref_len = min(len(temp_reference), len(resist_reference), n)
    temp_res = chip.temp_series[:ref_len] - temp_reference[:ref_len]
    resist_res = chip.resist_series[:ref_len] - resist_reference[:ref_len]

    return ChipFeatures(
        chip=chip,
        temp_roughness=compute_roughness(temp_res),
        resist_roughness=compute_roughness(resist_res),
        temp_spike=compute_spike(temp_res),
        resist_spike=compute_spike(resist_res),
        resist_cluster=resist_cluster,
    )


def normalize_roughness_sequence(
    roughness: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Z-score normalize a roughness sequence against a baseline mean/std.
    Used before feeding into CUSUM/EWMA charts.
    """
    return (roughness - baseline_mean) / (baseline_std + eps)
