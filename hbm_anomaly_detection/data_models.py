"""
Data classes for HBM CoW bonding anomaly detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class GroupKey(NamedTuple):
    """Unique key for a (equipment, recipe, head) combination that shares one baseline."""
    equipment_id: str
    recipe: str
    head: str


@dataclass
class ChipData:
    """
    Time-series data for a single bonded chip.

    temp_series and resist_series must have the same length (number of sample points).
    """
    equipment_id: str
    recipe: str
    head: str
    wafer_id: str
    x: int                          # chip column coordinate on wafer
    y: int                          # chip row coordinate on wafer
    bond_order: int                 # bonding sequence index within the wafer (0-based)
    temp_series: np.ndarray         # shape (T,) — temperature time series
    resist_series: np.ndarray       # shape (T,) — resistance time series

    @property
    def group_key(self) -> GroupKey:
        return GroupKey(self.equipment_id, self.recipe, self.head)

    @property
    def n_points(self) -> int:
        return len(self.temp_series)


@dataclass
class WaferData:
    """Collection of chip data from a single wafer track-out event."""
    wafer_id: str
    chips: list[ChipData] = field(default_factory=list)

    def chips_sorted_by_bond_order(self) -> list[ChipData]:
        return sorted(self.chips, key=lambda c: c.bond_order)

    @property
    def group_key(self) -> GroupKey | None:
        if not self.chips:
            return None
        return self.chips[0].group_key


@dataclass
class ChipFeatures:
    """Extracted features for a single chip."""
    chip: ChipData
    temp_roughness: float           # std of first-order diff of temp residual
    resist_roughness: float         # std of first-order diff of resist residual
    temp_spike: float               # max absolute temp residual
    resist_spike: float             # max absolute resist residual
    resist_cluster: int = 0         # bimodal cluster label (0 or 1), 0 if unimodal


@dataclass
class WaferFeatures:
    """Aggregated features for a wafer."""
    wafer_id: str
    chip_features: list[ChipFeatures]

    def roughness_sequence(self, channel: str = "resist") -> np.ndarray:
        """Roughness values in bond order for CUSUM/EWMA."""
        feats = sorted(self.chip_features, key=lambda f: f.chip.bond_order)
        if channel == "resist":
            return np.array([f.resist_roughness for f in feats])
        return np.array([f.temp_roughness for f in feats])

    def spike_mask(self, resist_threshold: float) -> np.ndarray:
        """Boolean mask — True where resist spike exceeds threshold."""
        feats = sorted(self.chip_features, key=lambda f: f.chip.bond_order)
        return np.array([f.resist_spike > resist_threshold for f in feats])

    def summary_stats(self, channel: str = "resist") -> dict[str, float]:
        seq = self.roughness_sequence(channel)
        return {
            "mean": float(np.mean(seq)),
            "max": float(np.max(seq)),
            "p90": float(np.percentile(seq, 90)),
        }


@dataclass
class AnomalyEvent:
    """A detected anomaly."""
    group_key: GroupKey
    wafer_id: str
    anomaly_type: str               # "intra_drift", "inter_drift", "spike", "coupling"
    channel: str                    # "temp", "resist", or "both"
    severity: float                 # detection statistic (e.g., CUSUM value or sigma distance)
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.anomaly_type.upper()}] wafer={self.wafer_id} "
            f"group={self.group_key} channel={self.channel} "
            f"severity={self.severity:.3f} details={self.details}"
        )
