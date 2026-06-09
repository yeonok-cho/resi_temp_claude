"""
Reference profile management for HBM CoW bonding anomaly detection.

Maintains per-(equipment_id, recipe, head) pointwise median/EWMA reference
profiles and handles bimodal resistance clustering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ReferenceConfig, DEFAULT_CONFIG
from .data_models import ChipData, WaferData, GroupKey


@dataclass
class BimodalState:
    """Stores k=2 resistance cluster centers and their reference profiles."""
    is_bimodal: bool = False
    cluster_centers: np.ndarray = field(default_factory=lambda: np.zeros(2))  # mean resist level
    temp_refs: list[np.ndarray] = field(default_factory=list)    # [ref0, ref1]
    resist_refs: list[np.ndarray] = field(default_factory=list)  # [ref0, ref1]


@dataclass
class GroupReference:
    """Reference state for one (equipment_id, recipe, head) group."""
    group_key: GroupKey
    n_points: int = 0                  # time-series length (set on first wafer)
    temp_ref: np.ndarray = field(default_factory=lambda: np.array([]))
    resist_ref: np.ndarray = field(default_factory=lambda: np.array([]))
    bimodal: BimodalState = field(default_factory=BimodalState)
    n_wafers_seen: int = 0

    # Running statistics for roughness normalization (inter-wafer baseline)
    roughness_mean_temp: float = 0.0
    roughness_mean_resist: float = 0.0
    roughness_std_temp: float = 1.0
    roughness_std_resist: float = 1.0
    roughness_m2_temp: float = 0.0   # Welford M2 accumulator
    roughness_m2_resist: float = 0.0
    roughness_n: int = 0             # number of wafers contributing to roughness stats

    def is_initialized(self) -> bool:
        return self.n_wafers_seen > 0 and len(self.temp_ref) > 0


def _kmeans2(values: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple k=2 1-D k-means by iterative center update.
    Returns (labels, centers).
    """
    lo, hi = np.min(values), np.max(values)
    c0, c1 = lo + (hi - lo) * 0.33, lo + (hi - lo) * 0.66

    for _ in range(max_iter):
        labels = (np.abs(values - c1) < np.abs(values - c0)).astype(int)
        new_c0 = values[labels == 0].mean() if (labels == 0).any() else c0
        new_c1 = values[labels == 1].mean() if (labels == 1).any() else c1
        if np.abs(new_c0 - c0) < 1e-9 and np.abs(new_c1 - c1) < 1e-9:
            break
        c0, c1 = new_c0, new_c1

    return labels, np.array([c0, c1])


def _detect_bimodal(
    mean_resist_levels: np.ndarray,
    config: ReferenceConfig,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Decide if resist levels are bimodal using k=2 clustering.

    Returns (is_bimodal, labels, centers).
    Bimodal is declared when the normalized gap between cluster means
    exceeds bimodal_separation_threshold.
    """
    if len(mean_resist_levels) < config.bimodal_min_chips:
        return False, np.zeros(len(mean_resist_levels), dtype=int), np.zeros(2)

    labels, centers = _kmeans2(mean_resist_levels)
    spread = np.std(mean_resist_levels)
    separation = (centers[1] - centers[0]) / (spread + 1e-9)
    is_bimodal = bool(separation > config.bimodal_separation_threshold)
    return is_bimodal, labels, centers


def _pointwise_median(series_list: list[np.ndarray]) -> np.ndarray:
    """Compute element-wise median across a list of equal-length arrays."""
    if not series_list:
        return np.array([])
    mat = np.stack(series_list, axis=0)  # (N, T)
    return np.median(mat, axis=0)


def _ewma_update(
    old: np.ndarray,
    new: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exponential moving average update: out = (1-alpha)*old + alpha*new."""
    return (1.0 - alpha) * old + alpha * new


def _welford_update(
    mean: float,
    m2: float,
    n: int,
    new_value: float,
) -> tuple[float, float, float]:
    """
    Online Welford update for mean and variance.
    Returns (new_mean, new_m2, new_n).
    """
    n += 1
    delta = new_value - mean
    mean += delta / n
    delta2 = new_value - mean
    m2 += delta * delta2
    return mean, m2, n


class ReferenceManager:
    """
    Manages per-group reference profiles and roughness baselines.

    On each call to update(), the reference is updated with EWMA smoothing
    using the current wafer's chip data.
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

    def assign_cluster(self, chip: ChipData, ref: GroupReference) -> int:
        """
        Assign a chip to the nearest bimodal cluster (0 or 1).
        Returns 0 if not bimodal.
        """
        if not ref.bimodal.is_bimodal or len(ref.bimodal.cluster_centers) < 2:
            return 0
        mean_r = float(np.mean(chip.resist_series))
        c0, c1 = ref.bimodal.cluster_centers
        return 1 if abs(mean_r - c1) < abs(mean_r - c0) else 0

    def get_references(
        self,
        chip: ChipData,
        ref: GroupReference,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Return (temp_ref, resist_ref, cluster_label) for a chip.
        Selects the appropriate bimodal reference if applicable.
        """
        cluster = self.assign_cluster(chip, ref)
        if ref.bimodal.is_bimodal and ref.bimodal.temp_refs:
            idx = min(cluster, len(ref.bimodal.temp_refs) - 1)
            return ref.bimodal.temp_refs[idx], ref.bimodal.resist_refs[idx], cluster
        return ref.temp_ref, ref.resist_ref, cluster

    def update(self, wafer: WaferData) -> None:
        """
        Update reference profiles using chips from a new wafer.

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
        chips = wafer.chips

        # Determine time-series length from first chip
        if ref.n_points == 0:
            ref.n_points = chips[0].n_points

        n_pts = ref.n_points

        # Compute mean resist level per chip for bimodal detection
        mean_resist = np.array([np.mean(c.resist_series) for c in chips])
        is_bimodal, labels, centers = _detect_bimodal(mean_resist, self._config)

        if is_bimodal:
            ref.bimodal.is_bimodal = True
            ref.bimodal.cluster_centers = centers

            for k in range(2):
                cluster_chips = [c for c, lbl in zip(chips, labels) if lbl == k]
                if not cluster_chips:
                    continue

                temps = [c.temp_series[:n_pts] for c in cluster_chips]
                resists = [c.resist_series[:n_pts] for c in cluster_chips]
                new_temp_ref = _pointwise_median(temps)
                new_resist_ref = _pointwise_median(resists)

                if ref.n_wafers_seen == 0 or len(ref.bimodal.temp_refs) <= k:
                    if len(ref.bimodal.temp_refs) <= k:
                        ref.bimodal.temp_refs.append(new_temp_ref)
                        ref.bimodal.resist_refs.append(new_resist_ref)
                    else:
                        ref.bimodal.temp_refs[k] = new_temp_ref
                        ref.bimodal.resist_refs[k] = new_resist_ref
                else:
                    ref.bimodal.temp_refs[k] = _ewma_update(
                        ref.bimodal.temp_refs[k], new_temp_ref, alpha
                    )
                    ref.bimodal.resist_refs[k] = _ewma_update(
                        ref.bimodal.resist_refs[k], new_resist_ref, alpha
                    )

            # Also maintain a global (averaged) reference for non-bimodal fallback
            all_temps = [c.temp_series[:n_pts] for c in chips]
            all_resists = [c.resist_series[:n_pts] for c in chips]
        else:
            ref.bimodal.is_bimodal = False
            all_temps = [c.temp_series[:n_pts] for c in chips]
            all_resists = [c.resist_series[:n_pts] for c in chips]

        new_temp_ref = _pointwise_median(all_temps)
        new_resist_ref = _pointwise_median(all_resists)

        if ref.n_wafers_seen == 0 or len(ref.temp_ref) == 0:
            ref.temp_ref = new_temp_ref
            ref.resist_ref = new_resist_ref
        else:
            ref.temp_ref = _ewma_update(ref.temp_ref, new_temp_ref, alpha)
            ref.resist_ref = _ewma_update(ref.resist_ref, new_resist_ref, alpha)

        ref.n_wafers_seen += 1

    def update_roughness_baseline(
        self,
        ref: GroupReference,
        wafer_mean_roughness_temp: float,
        wafer_mean_roughness_resist: float,
    ) -> None:
        """
        Update the online mean/std baseline for wafer-level roughness statistics
        using Welford's algorithm (called after feature extraction).

        Uses a separate counter (roughness_n) so it stays correct regardless of
        when update() was called relative to this method.
        """
        new_mean_t, new_m2_t, new_n = _welford_update(
            ref.roughness_mean_temp,
            ref.roughness_m2_temp,
            ref.roughness_n,
            wafer_mean_roughness_temp,
        )
        ref.roughness_mean_temp = new_mean_t
        ref.roughness_m2_temp = new_m2_t

        new_mean_r, new_m2_r, _ = _welford_update(
            ref.roughness_mean_resist,
            ref.roughness_m2_resist,
            ref.roughness_n,
            wafer_mean_roughness_resist,
        )
        ref.roughness_mean_resist = new_mean_r
        ref.roughness_m2_resist = new_m2_r

        ref.roughness_n = new_n
        if new_n >= 2:
            ref.roughness_std_temp = max(float(np.sqrt(new_m2_t / (new_n - 1))), 1e-9)
            ref.roughness_std_resist = max(float(np.sqrt(new_m2_r / (new_n - 1))), 1e-9)
