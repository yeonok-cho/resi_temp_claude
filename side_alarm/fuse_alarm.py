"""
Severity fusion (spec §16-§19).

TEMP is corroborating evidence for RESI, never a veto: a normal TEMP result can
never cancel or downgrade a valid RESI alarm, and an abnormal TEMP stays
independently visible to operators even when RESI is clean. Neither detector's
result is ever overwritten by the other — both are carried through into the
final row (spec §15, §20).

The matrix of spec §19 is expressed as a short sequence of named predicates
rather than nested ifs.
"""
from __future__ import annotations

import pandas as pd

from .config import SideAlarmConfig, DEFAULT_CONFIG
from .data_models import (
    ALARM, WARNING, CHECK, NORMAL, INSUFFICIENT_DATA, ABNORMAL_LEVELS,
    ResiResult, TempResult, FusionResult,
    TEMP_EVENT_TREND,
    SEV_NORMAL, SEV_CHECK, SEV_WARNING, SEV_RESI_ALARM, SEV_TEMP_ALARM,
    SEV_SEVERE, SEV_SEVERE_CHECK, SEV_CRITICAL, SEV_INSUFFICIENT_DATA,
)
from .loader import WAFER_KEYS


def is_abnormal(level: str) -> bool:
    """WARNING and ALARM escalate; CHECK is a monitor case, not an anomaly."""
    return level in ABNORMAL_LEVELS


def sides_agree(resi: ResiResult, temp: TempResult) -> bool | None:
    """
    True when both detectors point at the same physical module.

    None when the question is not meaningful — either suspect side is unknown,
    or a detector saw nothing at all. Every wafer has a side with the larger
    spread, so reporting a "direction match" for two NORMAL verdicts would read
    as evidence where there is none. None is not disagreement, and the caller
    must never read it as confirmation.
    """
    if resi.suspect_side is None or temp.suspect_side is None:
        return None
    if resi.level == NORMAL or temp.level == NORMAL:
        return None
    return resi.suspect_side == temp.suspect_side


def _both_abnormal_severity(resi: ResiResult, temp: TempResult, same_side: bool | None) -> tuple[str, str]:
    """Severity and reason when RESI and TEMP are both abnormal (spec §18)."""
    if same_side is not True:
        # Opposite (or unknown) side: not thermal confirmation of the RESI
        # anomaly. Neither signal is suppressed (spec §18.3).
        return SEV_SEVERE_CHECK, (
            "RESI and TEMP both abnormal, but suspect sides disagree "
            f"(RESI={resi.suspect_side}, TEMP={temp.suspect_side})"
        )
    if resi.level == ALARM and temp.event == TEMP_EVENT_TREND:
        # Strong module anomaly + same-side thermal abnormality + persistence.
        return SEV_CRITICAL, (
            "RESI module alarm corroborated by same-side TEMP equipment trend "
            f"({temp.suspect_side}, persisted {temp.trend_count} wafers)"
        )
    return SEV_SEVERE, (
        f"RESI {resi.level} corroborated by same-side TEMP {temp.level} "
        f"({temp.suspect_side}, {temp.event}) — not yet a sustained equipment trend"
    )


def fuse_alarm(
    resi: ResiResult,
    temp: TempResult,
    config: SideAlarmConfig | None = None,
) -> FusionResult:
    """Combine one wafer's independent RESI and TEMP verdicts."""
    _ = config or DEFAULT_CONFIG
    same_side = sides_agree(resi, temp)
    resi_ab = is_abnormal(resi.level)
    temp_ab = is_abnormal(temp.level)

    if resi_ab and temp_ab:
        severity, reason = _both_abnormal_severity(resi, temp, same_side)
        supports = same_side is True
    elif resi_ab:
        severity = SEV_RESI_ALARM if resi.level == ALARM else SEV_WARNING
        reason = f"RESI {resi.level} ({resi.event}) with TEMP {temp.level}; " \
                 "TEMP does not downgrade a RESI alarm"
        supports = False
    elif temp_ab:
        severity = SEV_TEMP_ALARM
        reason = f"TEMP {temp.level} ({temp.event}) on {temp.suspect_side} " \
                 f"with RESI {resi.level}; TEMP reported independently"
        supports = False
    elif CHECK in (resi.level, temp.level):
        severity = SEV_CHECK
        reason = f"monitor case: RESI={resi.level}, TEMP={temp.level}"
        supports = False
    elif INSUFFICIENT_DATA in (resi.level, temp.level):
        severity = SEV_INSUFFICIENT_DATA
        reason = f"insufficient data: RESI={resi.level}, TEMP={temp.level}"
        supports = False
    else:
        severity = SEV_NORMAL
        reason = ""
        supports = False

    if resi_ab and temp_ab:
        final_event = f"{resi.event}+{temp.event}"
    elif resi_ab:
        final_event = resi.event
    elif temp_ab:
        final_event = temp.event
    else:
        final_event = NORMAL

    return FusionResult(
        eqp_id=resi.eqp_id,
        product=resi.product,
        wafer_id=resi.wafer_id,
        write_time=resi.write_time,
        final_severity=severity,
        final_event=final_event,
        temp_supports_resi=supports,
        cross_signal_direction_match=same_side,
        final_reason=reason,
    )


# ---------------------------------------------------------------------------
# Frame-level driver
# ---------------------------------------------------------------------------

RESI_OUTPUT_COLUMNS = [
    "EQP_ID", "PRODUCT", "WAFER_ID", "WRITE_TIME",
    "RESI_LEVEL", "RESI_EVENT", "RESI_SUSPECT_SIDE", "RESI_STD_RATIO",
    "SUSPECT_RESI_MAX", "SUSPECT_RESI_3SD", "SUSPECT_RESI_5SD", "SUSPECT_RESI_9SD",
    "SIDE_COMPARISON_AVAILABLE", "RESI_REASON",
]

TEMP_OUTPUT_COLUMNS = [
    "EQP_ID", "PRODUCT", "WAFER_ID", "WRITE_TIME",
    "TEMP_LEVEL", "TEMP_EVENT", "TEMP_SUSPECT_SIDE",
    "SUSPECT_TEMP_HIGH_COUNT", "SUSPECT_TEMP_HIGH_SHARE", "TEMP_MEAN_RATIO",
    "TEMP_TREND_COUNT", "TEMP_QUANTIZED", "ONLINE_RESULT", "TEMP_REASON",
]

FUSION_OUTPUT_COLUMNS = [
    "EQP_ID", "PRODUCT", "WAFER_ID", "WRITE_TIME",
    "RESI_LEVEL", "RESI_EVENT", "RESI_SUSPECT_SIDE", "RESI_STD_RATIO", "RESI_REASON",
    "TEMP_LEVEL", "TEMP_EVENT", "TEMP_SUSPECT_SIDE", "SUSPECT_TEMP_HIGH_SHARE",
    "TEMP_TREND_COUNT", "TEMP_REASON",
    "TEMP_SUPPORTS_RESI", "CROSS_SIGNAL_DIRECTION_MATCH",
    "FINAL_SEVERITY", "FINAL_EVENT", "FINAL_REASON",
]


def _to_resi_result(row) -> ResiResult:
    return ResiResult(
        eqp_id=row.EQP_ID, product=row.PRODUCT, wafer_id=row.WAFER_ID,
        write_time=row.WRITE_TIME,
        level=row.RESI_LEVEL, event=row.RESI_EVENT,
        suspect_side=row.RESI_SUSPECT_SIDE,
        std_ratio=getattr(row, "RESI_STD_RATIO", float("nan")),
        reason=getattr(row, "RESI_REASON", ""),
    )


def _to_temp_result(row) -> TempResult:
    return TempResult(
        eqp_id=row.EQP_ID, product=row.PRODUCT, wafer_id=row.WAFER_ID,
        write_time=row.WRITE_TIME,
        level=row.TEMP_LEVEL, event=row.TEMP_EVENT,
        suspect_side=row.TEMP_SUSPECT_SIDE,
        trend_count=int(getattr(row, "TEMP_TREND_COUNT", 0) or 0),
        reason=getattr(row, "TEMP_REASON", ""),
    )


def fuse_frames(
    resi_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    Outer-join the two independent result frames on (EQP_ID, PRODUCT, WAFER_ID)
    and fuse each wafer.

    An outer join is deliberate: a wafer measured on only one of the two signals
    still has to reach the operator with that signal's verdict intact.
    """
    cfg = config or DEFAULT_CONFIG
    keys = list(WAFER_KEYS)

    resi = resi_df.drop_duplicates(subset=keys)
    temp = temp_df.drop_duplicates(subset=keys)
    merged = resi.merge(temp, on=keys, how="outer", suffixes=("", "_TEMP"))

    if "WRITE_TIME" not in merged.columns:
        merged["WRITE_TIME"] = pd.NaT
    if "WRITE_TIME_TEMP" in merged.columns:
        merged["WRITE_TIME"] = merged["WRITE_TIME"].fillna(merged["WRITE_TIME_TEMP"])

    defaults = {
        "RESI_LEVEL": NORMAL, "RESI_EVENT": NORMAL, "RESI_SUSPECT_SIDE": None,
        "RESI_REASON": "", "TEMP_LEVEL": NORMAL, "TEMP_EVENT": NORMAL,
        "TEMP_SUSPECT_SIDE": None, "TEMP_REASON": "", "TEMP_TREND_COUNT": 0,
    }
    for col, default in defaults.items():
        if col not in merged.columns:
            merged[col] = default
        elif default is not None:
            merged[col] = merged[col].fillna(default)

    fused = [
        fuse_alarm(_to_resi_result(row), _to_temp_result(row), cfg)
        for row in merged.itertuples(index=False)
    ]

    merged["TEMP_SUPPORTS_RESI"] = [f.temp_supports_resi for f in fused]
    merged["CROSS_SIGNAL_DIRECTION_MATCH"] = [f.cross_signal_direction_match for f in fused]
    merged["FINAL_SEVERITY"] = [f.final_severity for f in fused]
    merged["FINAL_EVENT"] = [f.final_event for f in fused]
    merged["FINAL_REASON"] = [f.final_reason for f in fused]

    for col in FUSION_OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    return merged[FUSION_OUTPUT_COLUMNS]
