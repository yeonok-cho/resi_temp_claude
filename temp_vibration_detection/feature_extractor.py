"""
Feature extraction for temperature vibration-increase detection.

Extracts a "roughness" feature from each chip's temperature residual.
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


def extract_chip_features(
    chip: ChipData,
    temp_reference: np.ndarray,
    config: FeatureConfig | None = None,
) -> ChipFeatures:
    """
    Extract the temperature roughness feature for one chip given a reference profile.

    Parameters
    ----------
    chip:
        The chip whose features are to be extracted.
    temp_reference:
        Reference temperature profile of shape (T,) from reference_manager.
    config:
        Feature extraction configuration.
    """
    if config is None:
        config = DEFAULT_CONFIG.feature

    n = chip.n_points
    if n < config.min_points:
        return ChipFeatures(chip=chip, temp_roughness=0.0)

    ref_len = min(len(temp_reference), n)
    residual = chip.temp_series[:ref_len] - temp_reference[:ref_len]

    return ChipFeatures(chip=chip, temp_roughness=compute_roughness(residual))
