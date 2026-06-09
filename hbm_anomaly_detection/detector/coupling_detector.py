"""
Bimodal resistance × temperature coupling breakdown detection.

In equipment where the resistance profile is bimodal, the temperature profile
should be identical across both clusters. This detector measures the L2
distance between the mean temperature profiles of the two resistance clusters
and flags it when this distance exceeds a historical EWMA baseline by 3σ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CouplingConfig, DEFAULT_CONFIG
from ..data_models import ChipData, WaferData, AnomalyEvent, GroupKey


def _mean_profile(chips: list[ChipData], n_pts: int) -> np.ndarray | None:
    """Compute mean temperature profile over a list of chips."""
    if not chips:
        return None
    mat = np.stack([c.temp_series[:n_pts] for c in chips], axis=0)
    return np.mean(mat, axis=0)


def _profile_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two profiles (normalized by length)."""
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def _kmeans2_resist(values: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """1-D k=2 k-means on scalar values (mean resistance levels)."""
    lo, hi = np.min(values), np.max(values)
    c0 = lo + (hi - lo) * 0.33
    c1 = lo + (hi - lo) * 0.66
    for _ in range(max_iter):
        labels = (np.abs(values - c1) < np.abs(values - c0)).astype(int)
        nc0 = values[labels == 0].mean() if (labels == 0).any() else c0
        nc1 = values[labels == 1].mean() if (labels == 1).any() else c1
        if np.abs(nc0 - c0) < 1e-9 and np.abs(nc1 - c1) < 1e-9:
            break
        c0, c1 = nc0, nc1
    return labels, np.array([c0, c1])


@dataclass
class CouplingState:
    """Running EWMA baseline for coupling distance."""
    ewma_distance: float = 0.0
    distance_mean: float = 0.0
    distance_std: float = 1.0
    n_samples: int = 0
    m2: float = 0.0
    initialized: bool = False
    baseline_frozen: bool = False


def _welford_update(mean: float, m2: float, n: int, x: float) -> tuple[float, float, int]:
    n += 1
    delta = x - mean
    mean += delta / n
    m2 += delta * (x - mean)
    return mean, m2, n


@dataclass
class CouplingDetector:
    """
    Detects temperature coupling breakdown in bimodal resistance equipment.

    One instance covers all groups; state is keyed by group_key.
    """
    config: CouplingConfig = field(default_factory=CouplingConfig)
    _states: dict[GroupKey, CouplingState] = field(default_factory=dict, init=False)

    def _get_state(self, key: GroupKey) -> CouplingState:
        if key not in self._states:
            self._states[key] = CouplingState()
        return self._states[key]

    def process_wafer(
        self,
        wafer: WaferData,
        group_key: GroupKey,
        bimodal_separation_threshold: float = 0.15,
    ) -> list[AnomalyEvent]:
        """
        Compute inter-cluster temperature distance for this wafer and detect
        coupling breakdown.

        If the resistance distribution is unimodal, returns no events.
        """
        chips = wafer.chips
        if len(chips) < 10:
            return []

        n_pts = chips[0].n_points
        mean_resist = np.array([np.mean(c.resist_series) for c in chips])

        # Check bimodality
        spread = np.std(mean_resist)
        if spread < 1e-9:
            return []

        labels, centers = _kmeans2_resist(mean_resist)
        separation = (centers[1] - centers[0]) / (spread + 1e-9)

        if separation <= bimodal_separation_threshold:
            return []  # unimodal — coupling check not applicable

        group0 = [c for c, lbl in zip(chips, labels) if lbl == 0]
        group1 = [c for c, lbl in zip(chips, labels) if lbl == 1]

        prof0 = _mean_profile(group0, n_pts)
        prof1 = _mean_profile(group1, n_pts)

        if prof0 is None or prof1 is None:
            return []

        distance = _profile_distance(prof0, prof1)
        return self._update_and_detect(group_key, wafer.wafer_id, distance)

    def _update_and_detect(
        self,
        group_key: GroupKey,
        wafer_id: str,
        distance: float,
    ) -> list[AnomalyEvent]:
        state = self._get_state(group_key)
        cfg = self.config
        events: list[AnomalyEvent] = []

        if not state.initialized:
            state.ewma_distance = distance
            state.initialized = True

        lam = cfg.ewma_lambda
        state.ewma_distance = (1 - lam) * state.ewma_distance + lam * distance

        # Accumulate baseline statistics only during the baseline period.
        # Freezing prevents a growing anomaly from adaptively raising the threshold.
        if not state.baseline_frozen:
            state.distance_mean, state.m2, state.n_samples = _welford_update(
                state.distance_mean, state.m2, state.n_samples, distance
            )
            n = state.n_samples
            if n >= 2:
                state.distance_std = max(float(np.sqrt(state.m2 / (n - 1))), 1e-9)
            if n >= cfg.baseline_wafers:
                state.baseline_frozen = True
        else:
            n = state.n_samples

        if state.baseline_frozen and state.distance_std > 0:
            z = (state.ewma_distance - state.distance_mean) / state.distance_std
            if z > cfg.coupling_sigma:
                events.append(AnomalyEvent(
                    group_key=group_key,
                    wafer_id=wafer_id,
                    anomaly_type="coupling",
                    channel="both",
                    severity=float(z),
                    details={
                        "coupling_distance": distance,
                        "ewma_distance": state.ewma_distance,
                        "baseline_mean": state.distance_mean,
                        "baseline_std": state.distance_std,
                        "z_score": float(z),
                    },
                ))

        return events
