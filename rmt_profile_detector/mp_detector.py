"""
Per-group, per-channel (temp / resist) stateful MP-score anomaly detector.

Usage
-----
detector = MPProfileDetector(MPConfig(...))

for wafer in wafers:
    temp_residuals = [extract_channel_residual(chip.temp_series, temp_ref, cfg.input_length)
                      for chip in wafer.chips]
    resist_residuals = [extract_channel_residual(chip.resist_series, resist_ref, cfg.input_length)
                        for chip in wafer.chips]
    events = detector.process_wafer(
        wafer.group_key, wafer.wafer_id, temp_residuals, resist_residuals, wafer.chips
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MPConfig
from .base_profile import BaseProfile

# These are imported from hbm_anomaly_detection so this module can slot in later;
# if used standalone, replace with equivalent local types.
from hbm_anomaly_detection.data_models import ChipData, GroupKey, AnomalyEvent


@dataclass
class _ChannelState:
    """Per-group, per-channel fitting state."""
    base_profile: BaseProfile = field(default_factory=BaseProfile)
    training_data: list[np.ndarray] = field(default_factory=list)
    n_wafers_seen: int = 0
    n_wafers_since_fit: int = 0
    fitted: bool = False


@dataclass
class _GroupState:
    """Holds temp and resist channel states for one group_key."""
    temp: _ChannelState = field(default_factory=_ChannelState)
    resist: _ChannelState = field(default_factory=_ChannelState)


class MPProfileDetector:
    """
    Stateful per-group MP-score detector for temp and resist residual profiles.

    One instance covers all groups (keyed by group_key).  During the
    baseline period (first ``baseline_wafers`` wafers), residual vectors
    are buffered for each channel.  Once enough wafers are collected the
    ``BaseProfile`` is fitted (once), and subsequent wafers are scored.

    Each chip gets two independent scores (temp channel, resist channel).
    An event is emitted for each chip×channel pair whose MP score exceeds
    the theoretical chi2 threshold.
    """

    def __init__(self, config: MPConfig | None = None) -> None:
        self._cfg = config or MPConfig()
        self._states: dict[GroupKey, _GroupState] = {}

    def _get_state(self, key: GroupKey) -> _GroupState:
        if key not in self._states:
            self._states[key] = _GroupState()
        return self._states[key]

    def process_wafer(
        self,
        group_key: GroupKey,
        wafer_id: str,
        temp_residuals: list[np.ndarray],
        resist_residuals: list[np.ndarray],
        chips: list[ChipData],
    ) -> list[AnomalyEvent]:
        """
        Process one wafer and return MP-score anomaly events.

        Parameters
        ----------
        group_key :
            Equipment / recipe / head group identifier.
        wafer_id :
            Identifier of the wafer being processed.
        temp_residuals, resist_residuals :
            Per-chip downsampled residual vectors, aligned with ``chips``.
        chips :
            ChipData objects aligned with the residual lists.
        """
        state = self._get_state(group_key)
        cfg = self._cfg
        events: list[AnomalyEvent] = []

        channels = (
            ("temp",   state.temp,   temp_residuals,   "mp_anomaly_temp"),
            ("resist", state.resist, resist_residuals, "mp_anomaly_resist"),
        )

        for ch_name, ch_state, residuals, anomaly_type in channels:
            events.extend(self._process_channel(
                ch_state, cfg, group_key, wafer_id,
                residuals, chips, ch_name, anomaly_type,
            ))

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_channel(
        self,
        ch_state: _ChannelState,
        cfg: MPConfig,
        group_key: GroupKey,
        wafer_id: str,
        residuals: list[np.ndarray],
        chips: list[ChipData],
        channel: str,
        anomaly_type: str,
    ) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []

        if not ch_state.fitted:
            ch_state.training_data.extend(residuals)
            ch_state.n_wafers_seen += 1
            if ch_state.n_wafers_seen >= cfg.baseline_wafers:
                self._fit_channel(ch_state, cfg)
            return events

        thr = ch_state.base_profile.threshold(cfg.alert_alpha)

        for chip, res in zip(chips, residuals):
            s = ch_state.base_profile.score(res)
            if s > thr:
                events.append(AnomalyEvent(
                    group_key=group_key,
                    wafer_id=wafer_id,
                    anomaly_type=anomaly_type,
                    channel=channel,
                    severity=float(s / thr),
                    details={
                        "chip_x": chip.x,
                        "chip_y": chip.y,
                        "bond_order": chip.bond_order,
                        "mp_score": float(s),
                        "threshold": float(thr),
                        "k_signal": ch_state.base_profile.k_signal_,
                        "k_noise": ch_state.base_profile.k_noise_,
                    },
                ))

        if cfg.retrain_interval_wafers > 0:
            ch_state.training_data.extend(residuals)
            ch_state.n_wafers_since_fit += 1
            if ch_state.n_wafers_since_fit >= cfg.retrain_interval_wafers:
                self._fit_channel(ch_state, cfg)

        return events

    def _fit_channel(self, ch_state: _ChannelState, cfg: MPConfig) -> None:
        X = np.stack(ch_state.training_data)
        bp = BaseProfile()
        bp.fit(X, min_noise_rank=cfg.min_noise_rank)
        ch_state.base_profile = bp
        ch_state.fitted = True
        ch_state.training_data = []
        ch_state.n_wafers_since_fit = 0
