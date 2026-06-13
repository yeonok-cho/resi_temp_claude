"""
Joint (temperature, resistance) inter-wafer drift detection using Hotelling's
T-squared statistic on a Multivariate EWMA (MEWMA) chart.

Monitors the joint vector ``[temp_roughness_mean, resist_roughness_mean]``
across consecutive wafers. Because T^2 is computed against the *covariance*
of the baseline (not just per-channel variance), it naturally responds to:

  - a shift in either channel alone, and
  - a shift in the *relationship* between the two channels (e.g. temp
    roughness rising while resist roughness stays flat changes the
    correlation structure), which the two independent univariate
    InterWaferTracker charts (one per channel) cannot see.

The baseline (mean vector and covariance matrix) is FROZEN after the first
``baseline_wafers`` observations, mirroring the univariate InterWaferTracker,
so that a genuine drift cannot adaptively raise the detection threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

from ..config import MultivariateInterWaferConfig, DEFAULT_CONFIG
from ..data_models import GroupKey, AnomalyEvent

_DIM = 2  # [temp_roughness_mean, resist_roughness_mean]
_COV_REG = 1e-6  # diagonal regularization to keep the covariance invertible


@dataclass
class MultivariateInterWaferState:
    """
    Stateful MEWMA tracker for one group_key's joint roughness vector.

    The Welford-style covariance accumulator is only updated during the
    baseline period. After ``baseline_wafers`` samples, mean/covariance
    are frozen.
    """
    ewma_vec: np.ndarray = field(default_factory=lambda: np.zeros(_DIM))
    baseline_mean: np.ndarray = field(default_factory=lambda: np.zeros(_DIM))
    baseline_cov: np.ndarray = field(default_factory=lambda: np.eye(_DIM))
    m2: np.ndarray = field(default_factory=lambda: np.zeros((_DIM, _DIM)))
    n_samples: int = 0
    initialized: bool = False
    baseline_frozen: bool = False


def _welford_update_mv(
    mean: np.ndarray, m2: np.ndarray, n: int, x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Online update of mean vector and covariance accumulator (M2 matrix)."""
    n += 1
    delta = x - mean
    mean = mean + delta / n
    delta2 = x - mean
    m2 = m2 + np.outer(delta, delta2)
    return mean, m2, n


@dataclass
class MultivariateInterWaferTracker:
    """
    Tracks per-group MEWMA/T^2 state for joint inter-wafer drift detection.

    One tracker instance covers all groups; state is keyed by group_key.
    """
    config: MultivariateInterWaferConfig = field(default_factory=MultivariateInterWaferConfig)
    _states: dict[GroupKey, MultivariateInterWaferState] = field(default_factory=dict, init=False)

    def _get_state(self, group_key: GroupKey) -> MultivariateInterWaferState:
        if group_key not in self._states:
            self._states[group_key] = MultivariateInterWaferState()
        return self._states[group_key]

    def update_and_detect(
        self,
        group_key: GroupKey,
        wafer_id: str,
        feature_vector: np.ndarray,
    ) -> list[AnomalyEvent]:
        """
        Feed one wafer's [temp_roughness_mean, resist_roughness_mean] vector
        into the MEWMA chart and check the Hotelling T^2 statistic.

        During the baseline period (first ``baseline_wafers`` wafers), the
        running mean/covariance are accumulated and no alarm is raised.
        After that, the baseline is frozen and only the MEWMA vector is
        updated.
        """
        state = self._get_state(group_key)
        cfg = self.config
        events: list[AnomalyEvent] = []
        x = np.asarray(feature_vector, dtype=float)

        # Bootstrap MEWMA from first observation
        if not state.initialized:
            state.ewma_vec = x.copy()
            state.initialized = True

        # Update MEWMA regardless of phase
        lam = cfg.ewma_lambda
        state.ewma_vec = (1 - lam) * state.ewma_vec + lam * x

        # Update baseline statistics only during the baseline period
        if not state.baseline_frozen:
            state.baseline_mean, state.m2, state.n_samples = _welford_update_mv(
                state.baseline_mean, state.m2, state.n_samples, x
            )
            n = state.n_samples
            if n >= 2:
                state.baseline_cov = state.m2 / (n - 1)

            if n >= cfg.baseline_wafers:
                state.baseline_frozen = True
        else:
            n = state.n_samples  # stays at baseline_wafers

        # Detect only after baseline is frozen (established)
        if state.baseline_frozen:
            cov = state.baseline_cov + np.eye(_DIM) * _COV_REG
            diff = state.ewma_vec - state.baseline_mean
            t2 = float(diff @ np.linalg.inv(cov) @ diff)
            control_limit = float(chi2.ppf(1.0 - cfg.alert_alpha, df=_DIM))

            if t2 > control_limit:
                events.append(AnomalyEvent(
                    group_key=group_key,
                    wafer_id=wafer_id,
                    anomaly_type="inter_drift_mv",
                    channel="both",
                    severity=t2,
                    details={
                        "t2": t2,
                        "control_limit": control_limit,
                        "ewma_vec": state.ewma_vec.tolist(),
                        "baseline_mean": state.baseline_mean.tolist(),
                        "baseline_cov": state.baseline_cov.tolist(),
                        "n_baseline_wafers": n,
                    },
                ))

        return events
