"""
Intra-wafer drift detection using CUSUM and EWMA control charts.

Detects when roughness increases progressively within a single wafer's
bonding sequence — a signal that vibration is worsening chip by chip.

The detector is self-calibrating: it estimates the "normal" roughness level
from the first portion of the wafer's own bond sequence, then runs CUSUM
on the remainder. This avoids dependency on a pre-built global baseline and
makes the detector work from the very first anomalous wafer.
"""
from __future__ import annotations

import numpy as np

from ..config import IntraWaferConfig, DEFAULT_CONFIG
from ..data_models import WaferFeatures, AnomalyEvent, GroupKey

# Minimum coefficient of variation used to floor the within-wafer std estimate.
# Prevents near-zero std (from an unlucky homogeneous warmup sample) from
# making the CUSUM hypersensitive to tiny natural roughness variations.
_MIN_CV = 0.15


def _cusum_chart(
    series: np.ndarray,
    k: float,
    h: float,
) -> tuple[np.ndarray, bool, int]:
    """
    One-sided (upward) CUSUM chart on a z-scored series.

    k : allowance (typical: 0.5 for detecting 1-sigma shift)
    h : decision interval (typical: 3-5 sigma units)
    """
    S = np.empty(len(series))
    s = 0.0
    alarm_idx = -1
    for i, x in enumerate(series):
        s = max(0.0, s + x - k)
        S[i] = s
        if alarm_idx < 0 and s > h:
            alarm_idx = i
    return S, alarm_idx >= 0, alarm_idx


def detect_intra_wafer_drift(
    wafer_features: WaferFeatures,
    group_key: GroupKey,
    baseline_mean: float = 0.0,    # unused; kept for API compatibility
    baseline_std: float = 1.0,     # unused; kept for API compatibility
    channel: str = "resist",
    config: IntraWaferConfig | None = None,
) -> list[AnomalyEvent]:
    """
    Run CUSUM on the intra-wafer roughness sequence to detect progressive
    vibration increase within a single wafer.

    Uses the first ``warmup_chips`` chips as the within-wafer reference.
    Detection runs on all subsequent chips.

    Parameters
    ----------
    wafer_features :
        Features extracted for all chips in the wafer.
    group_key :
        Identifier for this equipment/recipe/head combination.
    channel :
        "temp", "resist", or "both".
    config :
        Detection parameters.
    """
    if config is None:
        config = DEFAULT_CONFIG.intra_wafer

    events: list[AnomalyEvent] = []
    channels = ["temp", "resist"] if channel == "both" else [channel]

    for ch in channels:
        roughness = wafer_features.roughness_sequence(ch)
        n = len(roughness)
        warmup = config.warmup_chips

        if n < warmup * 2:
            continue

        # Estimate normal level from the first portion of the wafer
        warmup_mean = float(np.mean(roughness[:warmup]))
        warmup_sample_std = float(np.std(roughness[:warmup], ddof=1)) if warmup > 1 else 0.0

        # Floor the std so natural chip-to-chip variation doesn't create
        # near-zero denominators or overly sensitive z-scores.
        effective_std = max(warmup_sample_std, warmup_mean * _MIN_CV, 1e-9)

        # Z-score the whole sequence relative to warmup stats,
        # then run CUSUM on the post-warmup portion only.
        z_scores = (roughness - warmup_mean) / effective_std
        detection_z = z_scores[warmup:]

        S, alarm, alarm_offset = _cusum_chart(detection_z, config.cusum_k, config.cusum_h)
        peak = float(np.max(S)) if len(S) > 0 else 0.0
        alarm_idx = warmup + alarm_offset if alarm else -1

        if alarm:
            events.append(AnomalyEvent(
                group_key=group_key,
                wafer_id=wafer_features.wafer_id,
                anomaly_type="intra_drift",
                channel=ch,
                severity=peak,
                details={
                    "cusum_peak": peak,
                    "alarm_chip_index": alarm_idx,
                    "n_chips": n,
                    "warmup_mean": warmup_mean,
                    "effective_std": effective_std,
                },
            ))

    return events
