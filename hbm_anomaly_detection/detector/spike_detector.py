"""
Resistance spike detection for HBM CoW bonding anomaly detection.

Detects chips with sudden, large resistance excursions (spikes) on a
per-chip basis, and monitors wafer-level spike rates for trending faults.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import SpikeConfig, DEFAULT_CONFIG
from ..data_models import WaferFeatures, AnomalyEvent, GroupKey


def _compute_spike_threshold(
    resist_spike_values: np.ndarray,
    sigma: float,
) -> float:
    """
    Compute a per-wafer spike detection threshold using median + sigma * MAD.

    Median and MAD are robust to the spike chips themselves (stable as long
    as fewer than 50% of chips are spiked), so the threshold stays anchored
    to the normal chip distribution even when many chips are anomalous.
    The factor 1.4826 makes MAD consistent with the Gaussian sigma.
    """
    med = float(np.median(resist_spike_values))
    mad = float(np.median(np.abs(resist_spike_values - med)))
    robust_std = max(mad * 1.4826, 1e-9)
    return med + sigma * robust_std


@dataclass
class SpikeRateTracker:
    """
    Tracks per-group wafer-level spike rates and detects upward trends
    using a simple online CUSUM on the spike rate.
    """
    config: SpikeConfig = field(default_factory=SpikeConfig)
    _rates: dict[GroupKey, list[float]] = field(default_factory=dict, init=False)

    def _get_rates(self, key: GroupKey) -> list[float]:
        if key not in self._rates:
            self._rates[key] = []
        return self._rates[key]

    def update_and_detect(
        self,
        group_key: GroupKey,
        wafer_id: str,
        spike_rate: float,
    ) -> list[AnomalyEvent]:
        """
        Add this wafer's spike rate and check if the trend is rising.

        spike_rate : fraction of chips with a spike in this wafer.
        """
        rates = self._get_rates(group_key)
        rates.append(spike_rate)
        events: list[AnomalyEvent] = []

        if len(rates) < 5:
            return events

        arr = np.array(rates)
        mean = float(np.mean(arr[:-1]))
        std = max(float(np.std(arr[:-1])), 1e-9)
        z = (spike_rate - mean) / std

        if z > self.config.wafer_spike_rate_sigma:
            events.append(AnomalyEvent(
                group_key=group_key,
                wafer_id=wafer_id,
                anomaly_type="spike",
                channel="resist",
                severity=float(z),
                details={
                    "spike_rate": spike_rate,
                    "historical_mean_rate": mean,
                    "historical_std_rate": std,
                    "z_score": float(z),
                },
            ))

        return events


def detect_chip_spikes(
    wafer_features: WaferFeatures,
    group_key: GroupKey,
    config: SpikeConfig | None = None,
) -> tuple[list[AnomalyEvent], float]:
    """
    Detect individual chips with resistance spikes within a wafer.

    Uses the within-wafer distribution of spike magnitudes to set a
    data-adaptive threshold, making it robust to recipe-level differences.

    Returns
    -------
    events : list of AnomalyEvent for each chip that exceeded threshold
    spike_rate : fraction of chips flagged as spiked
    """
    if config is None:
        config = DEFAULT_CONFIG.spike

    feats = wafer_features.chip_features
    if not feats:
        return [], 0.0

    spike_vals = np.array([f.resist_spike for f in feats])
    threshold = _compute_spike_threshold(spike_vals, config.residual_sigma)

    events: list[AnomalyEvent] = []
    n_spike = 0

    for f in feats:
        if f.resist_spike > threshold:
            n_spike += 1
            events.append(AnomalyEvent(
                group_key=group_key,
                wafer_id=wafer_features.wafer_id,
                anomaly_type="spike",
                channel="resist",
                severity=float(f.resist_spike),
                details={
                    "chip_x": f.chip.x,
                    "chip_y": f.chip.y,
                    "bond_order": f.chip.bond_order,
                    "spike_value": float(f.resist_spike),
                    "threshold": float(threshold),
                },
            ))

    spike_rate = n_spike / max(len(feats), 1)
    return events, spike_rate
