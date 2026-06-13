"""
Inter-wafer temperature vibration drift detection using an EWMA control chart.

Monitors wafer-level temperature roughness summary statistics across
consecutive wafers to detect slow, multi-wafer upward trends that would be
invisible within any single wafer.

The baseline (mean and std of the control chart) is FROZEN after the first
``baseline_wafers`` observations so that a genuine drift cannot adaptively
raise the detection threshold and evade detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import InterWaferConfig, DEFAULT_CONFIG
from ..data_models import GroupKey, AnomalyEvent


@dataclass
class InterWaferState:
    """
    Stateful EWMA tracker for one group_key's temperature roughness.

    The Welford accumulator is only updated during the baseline period.
    After ``baseline_wafers`` samples, mean/std are frozen.
    """
    ewma_value: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 1.0
    n_samples: int = 0
    m2: float = 0.0                 # Welford accumulator (used only during baseline)
    initialized: bool = False
    baseline_frozen: bool = False


def _welford_update(mean: float, m2: float, n: int, x: float) -> tuple[float, float, int]:
    n += 1
    delta = x - mean
    mean += delta / n
    m2 += delta * (x - mean)
    return mean, m2, n


@dataclass
class InterWaferTracker:
    """
    Tracks per-group EWMA state for inter-wafer temperature drift detection.

    One tracker instance covers all groups; state is keyed by group_key.
    """
    config: InterWaferConfig = field(default_factory=InterWaferConfig)
    _states: dict[GroupKey, InterWaferState] = field(default_factory=dict, init=False)

    def _get_state(self, group_key: GroupKey) -> InterWaferState:
        if group_key not in self._states:
            self._states[group_key] = InterWaferState()
        return self._states[group_key]

    def update_and_detect(
        self,
        group_key: GroupKey,
        wafer_id: str,
        wafer_roughness_mean: float,
    ) -> list[AnomalyEvent]:
        """
        Feed one wafer's mean temperature roughness into the EWMA chart and
        check for alarm.

        During the baseline period (first ``baseline_wafers`` wafers), the
        running mean/std are accumulated and no alarm is raised.
        After that, the baseline is frozen and only the EWMA value is updated.
        """
        state = self._get_state(group_key)
        cfg = self.config
        events: list[AnomalyEvent] = []
        x = wafer_roughness_mean

        # Bootstrap EWMA from first observation
        if not state.initialized:
            state.ewma_value = x
            state.initialized = True

        # Update EWMA regardless of phase
        lam = cfg.ewma_lambda
        state.ewma_value = (1 - lam) * state.ewma_value + lam * x

        # Update baseline statistics only during the baseline period
        if not state.baseline_frozen:
            state.baseline_mean, state.m2, state.n_samples = _welford_update(
                state.baseline_mean, state.m2, state.n_samples, x
            )
            n = state.n_samples
            if n >= 2:
                state.baseline_std = max(float(np.sqrt(state.m2 / (n - 1))), 1e-9)

            if n >= cfg.baseline_wafers:
                state.baseline_frozen = True
        else:
            n = state.n_samples  # stays at baseline_wafers

        # Detect only after baseline is frozen (established)
        if state.baseline_frozen:
            control_limit = state.baseline_mean + cfg.alert_sigma * state.baseline_std
            if state.ewma_value > control_limit:
                severity = (state.ewma_value - state.baseline_mean) / state.baseline_std
                events.append(AnomalyEvent(
                    group_key=group_key,
                    wafer_id=wafer_id,
                    anomaly_type="inter_drift",
                    severity=severity,
                    details={
                        "ewma": state.ewma_value,
                        "baseline_mean": state.baseline_mean,
                        "baseline_std": state.baseline_std,
                        "control_limit": control_limit,
                        "n_baseline_wafers": n,
                    },
                ))

        return events
