"""
Reference profile management for temperature vibration-increase detection.

Maintains per-(equipment_id, recipe, head) pointwise median temperature
reference profile, updated via EWMA on each wafer track-out.

No bimodal clustering is performed here — that is a resistance-specific
concern excluded from this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ReferenceConfig, DEFAULT_CONFIG
from .data_models import WaferData, GroupKey


@dataclass
class GroupReference:
    """Reference state for one (equipment_id, recipe, head) group."""
    group_key: GroupKey
    n_points: int = 0                  # time-series length (set on first wafer)
    temp_ref: np.ndarray = field(default_factory=lambda: np.array([]))
    n_wafers_seen: int = 0

    def is_initialized(self) -> bool:
        return self.n_wafers_seen > 0 and len(self.temp_ref) > 0


def _pointwise_median(series_list: list[np.ndarray]) -> np.ndarray:
    """Compute element-wise median across a list of equal-length arrays."""
    if not series_list:
        return np.array([])
    mat = np.stack(series_list, axis=0)  # (N, T)
    return np.median(mat, axis=0)


def _ewma_update(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average update: out = (1-alpha)*old + alpha*new."""
    return (1.0 - alpha) * old + alpha * new


class ReferenceManager:
    """
    Manages per-group temperature reference profiles.

    On each call to update(), the reference is updated with EWMA smoothing
    using the current wafer's chip temperature data.
    """

    def __init__(self, config: ReferenceConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG.reference
        self._groups: dict[GroupKey, GroupReference] = {}

    def get_or_create(self, key: GroupKey) -> GroupReference:
        if key not in self._groups:
            self._groups[key] = GroupReference(group_key=key)
        return self._groups[key]

    def get(self, key: GroupKey) -> GroupReference | None:
        return self._groups.get(key)

    def reset(self, key: GroupKey) -> None:
        """
        Discard the reference profile for a group.

        The next wafer for this group is treated as the first wafer again
        (reference initialized directly from its pointwise median, no EWMA
        blending with the pre-reset profile). Use this after a PM event that
        may have changed the equipment's temperature profile.
        """
        self._groups.pop(key, None)

    def update(self, wafer: WaferData) -> None:
        """
        Update the temperature reference profile using chips from a new wafer.

        First call initializes the reference from the wafer's chip data.
        Subsequent calls apply EWMA smoothing.
        """
        if not wafer.chips:
            return

        key = wafer.group_key
        if key is None:
            return

        ref = self.get_or_create(key)
        alpha = self._config.ewma_alpha

        if ref.n_points == 0:
            ref.n_points = wafer.chips[0].n_points
        n_pts = ref.n_points

        temps = [c.temp_series[:n_pts] for c in wafer.chips]
        new_ref = _pointwise_median(temps)

        if ref.n_wafers_seen == 0 or len(ref.temp_ref) == 0:
            ref.temp_ref = new_ref
        else:
            ref.temp_ref = _ewma_update(ref.temp_ref, new_ref, alpha)

        ref.n_wafers_seen += 1
