"""
Level / event vocabularies and the per-wafer record types exchanged between
the RESI detector, the TEMP detector and the fusion layer.

Detector *stages* work on pandas frames (one row per wafer-side, then one row
per wafer) for vectorized throughput; the fusion layer works on these small
records so its decision logic stays readable (spec §19, §32).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Per-signal severity levels (spec §6).
NORMAL = "NORMAL"
CHECK = "CHECK"
WARNING = "WARNING"
ALARM = "ALARM"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"   # spec §26 — never silently NORMAL

LEVEL_ORDER = {INSUFFICIENT_DATA: -1, NORMAL: 0, CHECK: 1, WARNING: 2, ALARM: 3}

#: Levels that count as "abnormal" when escalating in the fusion layer.
ABNORMAL_LEVELS = (WARNING, ALARM)

# RESI event types (spec §7).
RESI_EVENT_NORMAL = "NORMAL"
RESI_EVENT_LOCALIZED_SPOT = "LOCALIZED_SPOT"
RESI_EVENT_MODULE_DEGRADATION = "MODULE_DEGRADATION"
RESI_EVENT_MODULE_FAILURE = "MODULE_FAILURE"
RESI_EVENT_PROCESS_SHIFT = "PROCESS_SHIFT"

# TEMP event types (spec §12).
TEMP_EVENT_NORMAL = "NORMAL"
TEMP_EVENT_WAFER = "WAFER_EVENT"
TEMP_EVENT_TRANSIENT = "TRANSIENT"
TEMP_EVENT_TREND = "EQUIPMENT_TREND"

# Fused severity classes (spec §17).
SEV_NORMAL = "NORMAL"
SEV_CHECK = "CHECK"
SEV_WARNING = "WARNING"
SEV_RESI_ALARM = "RESI_ALARM"
SEV_TEMP_ALARM = "TEMP_ALARM"
SEV_SEVERE = "SEVERE"
SEV_SEVERE_CHECK = "SEVERE_CHECK"
SEV_CRITICAL = "CRITICAL"
SEV_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def max_level(a: str, b: str) -> str:
    """Return whichever of two levels is more severe."""
    return a if LEVEL_ORDER.get(a, 0) >= LEVEL_ORDER.get(b, 0) else b


# ---------------------------------------------------------------------------
# Per-wafer records
# ---------------------------------------------------------------------------

@dataclass
class ResiResult:
    """One wafer's RESI verdict (spec §15)."""
    eqp_id: str
    product: str
    wafer_id: str
    write_time: object = None
    level: str = NORMAL
    event: str = RESI_EVENT_NORMAL
    suspect_side: str | None = None
    other_side: str | None = None
    std_ratio: float = float("nan")
    mean_ratio: float = float("nan")
    max_ratio: float = float("nan")
    max_diff: float = float("nan")
    suspect_max: float = float("nan")
    suspect_n: int = 0
    sd_counts: dict[int, int] = field(default_factory=dict)
    side_comparison_available: bool = False
    reason: str = ""


@dataclass
class TempResult:
    """One wafer's TEMP verdict (spec §15)."""
    eqp_id: str
    product: str
    wafer_id: str
    write_time: object = None
    level: str = NORMAL
    event: str = TEMP_EVENT_NORMAL
    suspect_side: str | None = None
    other_side: str | None = None
    high_count: int = 0
    high_share: float = float("nan")
    other_high_share: float = float("nan")
    mean_ratio: float = float("nan")
    high_share_ratio: float = float("nan")
    trend_count: int = 0
    quantized: bool = False
    legacy_mode: bool = False
    side_comparison_available: bool = False
    reason: str = ""


@dataclass
class FusionResult:
    """Combined verdict — keeps both inputs visible (spec §20)."""
    eqp_id: str
    product: str
    wafer_id: str
    write_time: object = None
    final_severity: str = SEV_NORMAL
    final_event: str = NORMAL
    temp_supports_resi: bool = False
    cross_signal_direction_match: bool | None = None
    final_reason: str = ""
