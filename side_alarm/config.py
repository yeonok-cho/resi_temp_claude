"""
Centralized configuration for LEFT/RIGHT side-asymmetry RESI/TEMP alarming.

Every threshold used by detect_resi / detect_temp / trend / fuse_alarm lives
here; no magic numbers are allowed in the decision modules. Mirrors the
nested-dataclass + DEFAULT_CONFIG pattern already used by
``temp_anomaly.temp_vibration_detection.config`` and
``hbm_anomaly_detection.config``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Shared division guard. Every ratio in this package divides by max(x, EPS).
EPS: float = 1e-8

LEFT = "LEFT"
RIGHT = "RIGHT"
SIDES = (LEFT, RIGHT)


@dataclass
class ColumnConfig:
    """
    Physical column names in the chip-level source table.

    Defaults match ``ds_catalog.aifpa_cow_rule_hist`` as read by
    ``temp_anomaly/scripts/_common.py``, extended with the two columns this
    package consumes: the resistance anomaly score (``mp_score``) and the
    resistance time series (``resi_raw``).
    """
    table: str = "ds_catalog.aifpa_cow_rule_hist"

    eqp_id: str = "eqp_id"
    product: str = "product"
    module_id: str = "module_id"      # carries the LEFT/RIGHT side value
    wafer_id: str = "wafer_id"
    x: str = "x"
    y: str = "y"
    write_time: str = "event_tmstp"

    # RESI: per-chip anomaly score, already computed upstream.
    # Same quantity as rmt_profile_detector.BaseProfile.score() — a non-negative
    # chi2-like statistic. This package never recomputes it (see spec §31.6).
    resi_score: str = "mp_score"

    # TEMP: per-chip roughness, already computed upstream. When this column is
    # absent, the loader can fall back to computing it from ``temp_raw`` using
    # the repo's own definition (see loader.compute_roughness_from_raw).
    temp_roughness: str = "temp_roughness"

    # Raw hex-encoded float32 time series (fallback / diagnostics only).
    temp_raw: str = "temp_raw"
    resi_raw: str = "resi_raw"

    def chip_columns(self) -> list[str]:
        """Columns the loader SELECTs for scoring (raw series excluded)."""
        return [
            self.eqp_id, self.product, self.module_id, self.wafer_id,
            self.x, self.y, self.write_time, self.resi_score,
        ]


@dataclass
class SideConfig:
    """Normalization of the raw module_id value onto LEFT / RIGHT."""
    left_values: tuple[str, ...] = ("left", "l", "lft", "lh")
    right_values: tuple[str, ...] = ("right", "r", "rgt", "rh")
    # Rows whose module_id maps to neither list are dropped, with a warning.
    drop_unmapped: bool = True


@dataclass
class BaselineConfig:
    """
    Historical healthy-wafer baseline, computed per (EQP_ID, SIDE).

    The baseline window always ENDS before the scoring window begins, which is
    what makes the result leakage-free (spec §24): a wafer at time t is only
    ever compared against data strictly older than t.
    """
    lookback_days: int = 30           # reference period length; never a hard-coded date
    min_baseline_wafers: int = 10     # per (eqp, side) wafer-sides required for a usable baseline
    min_baseline_chips: int = 200     # per (eqp, side) chip rows required for chip-level mean/std

    # Percentiles kept for the wafer-side statistics (spec §5, §9).
    resi_max_percentiles: tuple[float, ...] = (95.0, 99.0)
    resi_mean_percentiles: tuple[float, ...] = (90.0, 95.0)
    resi_std_percentiles: tuple[float, ...] = (95.0, 99.0)
    temp_percentiles: tuple[float, ...] = (95.0, 99.0)

    # Optional robust cleaning when no problem-wafer labels are available.
    # 0.0 = off (use every historical wafer-side).
    exclude_top_pct: float = 0.0


@dataclass
class ResiConfig:
    """RESI (resistance anomaly score) side-asymmetry thresholds — spec §6, §7."""
    alarm_std_ratio: float = 5.0
    warn_std_ratio: float = 2.0

    warn_3sd_count: int = 8
    alarm_5sd_count: int = 3
    alarm_9sd_count: int = 1
    check_5sd_count: int = 3

    # Sigma levels counted per wafer-side (spec §4.2).
    sd_levels: tuple[int, ...] = (3, 5, 7, 9)

    # Chip-level z-scores are taken against the equipment-side baseline
    # ("baseline") or against the wafer-side's own mean/std ("wafer").
    # "baseline" is the default: a failing heater inflates the wafer-side std,
    # which would suppress its own SD counts under "wafer".
    sd_reference: str = "baseline"

    # PROCESS_SHIFT: both sides elevated with little asymmetry (spec §7).
    process_shift_std_ratio: float = 2.0
    process_shift_mean_percentile: float = 95.0


@dataclass
class TempConfig:
    """TEMP (roughness) thresholds — spec §11, §12, §13, §14."""
    high_share_ratio: float = 2.0
    mean_ratio: float = 1.5
    min_high_count: int = 3

    # Percentile defining a "high" chip, per equipment and side.
    high_percentile: float = 99.0

    # Consecutive suspect wafers required to call an equipment trend.
    trend_k: int = 3

    # Quantization diagnostic (spec §14).
    quant_top3_share: float = 0.80
    quant_max_unique: int = 12

    # Legacy std-ratio corroboration, only for non-quantized equipment (spec §13).
    legacy_enabled: bool = True
    legacy_std_ratio: float = 2.0
    legacy_max_percentile: float = 95.0


@dataclass
class QualityConfig:
    """Minimum sample sizes — spec §26."""
    min_resi_chips: int = 10
    min_temp_chips: int = 10


@dataclass
class SideAlarmConfig:
    """Top-level configuration aggregating all sub-configs."""
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    side: SideConfig = field(default_factory=SideConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    resi: ResiConfig = field(default_factory=ResiConfig)
    temp: TempConfig = field(default_factory=TempConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    eps: float = EPS


DEFAULT_CONFIG = SideAlarmConfig()
