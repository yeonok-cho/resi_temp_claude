"""
RESI detector — heater/module abnormality from the LEFT/RIGHT distribution of
per-chip resistance anomaly scores (``mp_score``).

Design note (spec §4.1): a wafer is NOT classified on AnomMax alone. Absolute
maximum scores vary strongly by equipment and process generation, so the
primary signal is within-wafer LEFT vs RIGHT asymmetry — above all the ratio of
the two per-side score standard deviations. Absolute magnitude enters only
through equipment-side baseline percentiles.

This detector is fully independent of TEMP: it never reads a roughness value
and never requires a TEMP verdict to raise an alarm (spec §2, §33).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SideAlarmConfig, DEFAULT_CONFIG, LEFT, RIGHT
from .data_models import (
    NORMAL, CHECK, WARNING, ALARM, INSUFFICIENT_DATA,
    RESI_EVENT_NORMAL, RESI_EVENT_LOCALIZED_SPOT, RESI_EVENT_MODULE_DEGRADATION,
    RESI_EVENT_MODULE_FAILURE, RESI_EVENT_PROCESS_SHIFT,
)
from .loader import WAFER_KEYS, aggregate_wafer_side, pivot_sides
from .baseline import BASELINE_KEYS


# ---------------------------------------------------------------------------
# 1. Per wafer-side statistics (spec §4.2)
# ---------------------------------------------------------------------------

def calculate_resi_stats(
    chips: pd.DataFrame,
    resi_baseline: pd.DataFrame | None = None,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    N / mean / median / std / max / skew and the 3-5-7-9 sigma chip counts,
    one row per (EQP_ID, PRODUCT, WAFER_ID, SIDE).

    The sigma counts are taken against the equipment-side baseline chip
    distribution by default (``ResiConfig.sd_reference == "baseline"``). Using
    the wafer-side's own mean/std instead would be self-defeating: a failing
    heater inflates that side's std, which suppresses its own SD counts.
    """
    cfg = config or DEFAULT_CONFIG
    rcfg = cfg.resi
    score_col = cfg.columns.resi_score
    work = chips.copy()

    if rcfg.sd_reference == "baseline" and resi_baseline is not None and not resi_baseline.empty:
        ref = resi_baseline[list(BASELINE_KEYS) + ["RESI_CHIP_MEAN", "RESI_CHIP_STD"]]
        work = work.merge(ref, on=list(BASELINE_KEYS), how="left")
        ref_mean = work["RESI_CHIP_MEAN"]
        ref_std = work["RESI_CHIP_STD"]
    else:
        g = work.groupby(list(WAFER_KEYS) + ["SIDE"], sort=False)[score_col]
        ref_mean = g.transform("mean")
        ref_std = g.transform("std")

    z = (work[score_col] - ref_mean) / np.maximum(ref_std.fillna(0.0), cfg.eps)

    extra: dict[str, tuple[str, str]] = {}
    for k in rcfg.sd_levels:
        flag = f"_ge{k}sd"
        work[flag] = (z >= k).fillna(False)
        extra[f"RESI_{k}SD"] = (flag, "sum")

    stats = aggregate_wafer_side(work, score_col, "RESI", cfg, extra_aggs=extra)
    for k in rcfg.sd_levels:
        stats[f"RESI_{k}SD"] = stats[f"RESI_{k}SD"].fillna(0).astype(int)
    return stats


# ---------------------------------------------------------------------------
# 2. LEFT vs RIGHT comparison (spec §4.3)
# ---------------------------------------------------------------------------

def _side_value(wide: pd.DataFrame, base: str, suspect: pd.Series, take_suspect: bool) -> pd.Series:
    """Pick the suspect-side (or other-side) column value row by row."""
    left, right = wide[f"{base}_{LEFT}"], wide[f"{base}_{RIGHT}"]
    want_left = (suspect == LEFT) if take_suspect else (suspect == RIGHT)
    return left.where(want_left, right)


def compare_resi_sides(
    stats: pd.DataFrame,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    One row per wafer with the suspect side and its asymmetry ratios.

    The suspect side is the one with the LARGER RESI std. When only one side
    was measured, ``SIDE_COMPARISON_AVAILABLE`` is False and no ratio is
    produced — the single-side absolute checks still run (spec §27).
    """
    cfg = config or DEFAULT_CONFIG
    wide = pivot_sides(stats, "RESI")

    std_l, std_r = wide[f"RESI_STD_{LEFT}"], wide[f"RESI_STD_{RIGHT}"]
    n_l, n_r = wide[f"RESI_N_{LEFT}"], wide[f"RESI_N_{RIGHT}"]

    both = n_l.notna() & n_r.notna()
    # Larger std wins; with only one side present that side is the suspect.
    suspect = pd.Series(
        np.where(both, np.where(std_l.fillna(-1) >= std_r.fillna(-1), LEFT, RIGHT),
                 np.where(n_l.notna(), LEFT, RIGHT)),
        index=wide.index, dtype=object,
    )
    other = pd.Series(np.where(suspect == LEFT, RIGHT, LEFT), index=wide.index, dtype=object)

    out = wide[list(WAFER_KEYS) + ["WRITE_TIME"]].copy()
    out["RESI_SUSPECT_SIDE"] = suspect
    out["RESI_OTHER_SIDE"] = other.where(both, None)
    out["SIDE_COMPARISON_AVAILABLE"] = both

    for base in ["RESI_N", "RESI_MEAN", "RESI_MEDIAN", "RESI_STD", "RESI_MAX", "RESI_SKEW"] + \
                [f"RESI_{k}SD" for k in cfg.resi.sd_levels]:
        out[f"SUSPECT_{base}"] = _side_value(wide, base, suspect, True)
        out[f"OTHER_{base}"] = _side_value(wide, base, suspect, False).where(both)

    eps = cfg.eps
    out["RESI_STD_RATIO"] = out["SUSPECT_RESI_STD"] / np.maximum(out["OTHER_RESI_STD"], eps)
    out["RESI_MEAN_RATIO"] = out["SUSPECT_RESI_MEAN"] / np.maximum(out["OTHER_RESI_MEAN"], eps)
    out["RESI_MAX_RATIO"] = out["SUSPECT_RESI_MAX"] / np.maximum(out["OTHER_RESI_MAX"], eps)
    out["RESI_MAX_DIFF"] = out["SUSPECT_RESI_MAX"] - out["OTHER_RESI_MAX"]

    # Keep both raw sides for PROCESS_SHIFT, which is not a suspect-side test.
    for side in (LEFT, RIGHT):
        out[f"RESI_MEAN_{side}"] = wide[f"RESI_MEAN_{side}"]
        out[f"RESI_N_{side}"] = wide[f"RESI_N_{side}"]
    return out


# ---------------------------------------------------------------------------
# 3. Verdict (spec §6, §7)
# ---------------------------------------------------------------------------

def _merge_baseline_per_side(pairs: pd.DataFrame, baseline: pd.DataFrame,
                             cols: list[str]) -> pd.DataFrame:
    """Attach each side's own baseline columns as ``<col>_LEFT`` / ``<col>_RIGHT``."""
    out = pairs
    for side in (LEFT, RIGHT):
        sub = baseline[baseline["SIDE"] == side][["EQP_ID"] + cols + ["BASELINE_OK"]]
        sub = sub.rename(columns={c: f"{c}_{side}" for c in cols + ["BASELINE_OK"]})
        out = out.merge(sub, on="EQP_ID", how="left")
    return out


def detect_resi_anomaly(
    pairs: pd.DataFrame,
    resi_baseline: pd.DataFrame | None = None,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    Assign RESI_LEVEL / RESI_EVENT / RESI_REASON to each wafer.

    Levels are evaluated most-severe-first (spec §6):
      ALARM   : std ratio >= 5 AND (9SD >= 1 OR max > P99 OR 5SD >= 3)
      WARNING : std ratio >= 2 AND (3SD >= 8 OR max > P95)
      CHECK   : max > P99 OR 5SD >= 3
    """
    cfg = config or DEFAULT_CONFIG
    rcfg = cfg.resi
    out = pairs.copy()

    pct_cols = ["RESI_MAX_P95", "RESI_MAX_P99",
                f"RESI_MEAN_P{rcfg.process_shift_mean_percentile:g}"]
    if resi_baseline is not None and not resi_baseline.empty:
        out = _merge_baseline_per_side(out, resi_baseline, pct_cols)
    else:
        for side in (LEFT, RIGHT):
            for c in pct_cols + ["BASELINE_OK"]:
                out[f"{c}_{side}"] = np.nan if c != "BASELINE_OK" else False

    suspect = out["RESI_SUSPECT_SIDE"]
    is_left = suspect == LEFT

    def suspect_baseline(col: str) -> pd.Series:
        return out[f"{col}_{LEFT}"].where(is_left, out[f"{col}_{RIGHT}"])

    p95 = suspect_baseline("RESI_MAX_P95")
    p99 = suspect_baseline("RESI_MAX_P99")
    baseline_ok = suspect_baseline("BASELINE_OK").fillna(False).astype(bool)
    out["RESI_BASELINE_AVAILABLE"] = baseline_ok

    ratio = out["RESI_STD_RATIO"]
    sd9 = out[f"SUSPECT_RESI_{9}SD"].fillna(0)
    sd5 = out[f"SUSPECT_RESI_{5}SD"].fillna(0)
    sd3 = out[f"SUSPECT_RESI_{3}SD"].fillna(0)

    over_p99 = baseline_ok & (out["SUSPECT_RESI_MAX"] > p99)
    over_p95 = baseline_ok & (out["SUSPECT_RESI_MAX"] > p95)

    alarm = (ratio >= rcfg.alarm_std_ratio) & (
        (sd9 >= rcfg.alarm_9sd_count) | over_p99 | (sd5 >= rcfg.alarm_5sd_count)
    )
    warning = (ratio >= rcfg.warn_std_ratio) & (
        (sd3 >= rcfg.warn_3sd_count) | over_p95
    )
    check = over_p99 | (sd5 >= rcfg.check_5sd_count)

    # PROCESS_SHIFT: little asymmetry but BOTH sides elevated over their own
    # equipment baselines — a recipe/process move, not a heater failure (spec §7).
    mean_p = f"RESI_MEAN_P{rcfg.process_shift_mean_percentile:g}"
    both_elevated = (
        (out[f"RESI_MEAN_{LEFT}"] > out[f"{mean_p}_{LEFT}"])
        & (out[f"RESI_MEAN_{RIGHT}"] > out[f"{mean_p}_{RIGHT}"])
    ).fillna(False)
    process_shift = (
        out["SIDE_COMPARISON_AVAILABLE"]
        & (ratio < rcfg.process_shift_std_ratio)
        & both_elevated
    )

    insufficient = out["SUSPECT_RESI_N"].fillna(0) < cfg.quality.min_resi_chips

    out["RESI_LEVEL"] = np.select(
        [insufficient, alarm, warning, check | process_shift],
        [INSUFFICIENT_DATA, ALARM, WARNING, CHECK],
        default=NORMAL,
    )
    out["RESI_EVENT"] = np.select(
        [insufficient, alarm, warning, process_shift, check],
        [RESI_EVENT_NORMAL, RESI_EVENT_MODULE_FAILURE, RESI_EVENT_MODULE_DEGRADATION,
         RESI_EVENT_PROCESS_SHIFT, RESI_EVENT_LOCALIZED_SPOT],
        default=RESI_EVENT_NORMAL,
    )
    out["RESI_REASON"] = _resi_reasons(out, p95, p99, insufficient, process_shift, cfg)
    return out


def _resi_reasons(out: pd.DataFrame, p95: pd.Series, p99: pd.Series,
                  insufficient: pd.Series, process_shift: pd.Series,
                  cfg: SideAlarmConfig) -> pd.Series:
    """
    Human-readable reason carrying the measured value AND the threshold it
    crossed (spec §21). Built only for non-NORMAL rows — a wafer-level loop over
    a small subset, not a row-wise pass over chip data.
    """
    reasons = pd.Series("", index=out.index, dtype=object)
    interesting = out.index[(out["RESI_LEVEL"] != NORMAL)]
    rcfg = cfg.resi

    for i in interesting:
        row = out.loc[i]
        if insufficient.loc[i]:
            reasons.loc[i] = (
                f"suspect side chips={int(row['SUSPECT_RESI_N'] or 0)} "
                f"< min {cfg.quality.min_resi_chips}"
            )
            continue
        if process_shift.loc[i]:
            reasons.loc[i] = (
                f"std ratio={row['RESI_STD_RATIO']:.2f} < {rcfg.process_shift_std_ratio} "
                f"and both sides above baseline mean "
                f"(LEFT {row[f'RESI_MEAN_{LEFT}']:.4g}, RIGHT {row[f'RESI_MEAN_{RIGHT}']:.4g})"
            )
            continue

        side = row["RESI_SUSPECT_SIDE"]
        bits = []
        if pd.notna(row["RESI_STD_RATIO"]):
            bits.append(f"{side} std ratio={row['RESI_STD_RATIO']:.2f}")
        for k in rcfg.sd_levels:
            n = row.get(f"SUSPECT_RESI_{k}SD")
            if pd.notna(n) and n > 0:
                bits.append(f"{side} {k}SD={int(n)}")
        thr = p99.loc[i] if pd.notna(p99.loc[i]) else p95.loc[i]
        if pd.notna(thr) and pd.notna(row["SUSPECT_RESI_MAX"]):
            bits.append(f"{side} max={row['SUSPECT_RESI_MAX']:.4g} vs P99={thr:.4g}")
        if not row["SIDE_COMPARISON_AVAILABLE"]:
            bits.append("side comparison unavailable (one side only)")
        if not row["RESI_BASELINE_AVAILABLE"]:
            bits.append("equipment baseline unavailable")
        reasons.loc[i] = ", ".join(bits)

    return reasons


def run_resi_detector(
    chips: pd.DataFrame,
    resi_baseline: pd.DataFrame | None = None,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """stats -> side comparison -> verdict, in one call."""
    cfg = config or DEFAULT_CONFIG
    stats = calculate_resi_stats(chips, resi_baseline, cfg)
    pairs = compare_resi_sides(stats, cfg)
    return detect_resi_anomaly(pairs, resi_baseline, cfg)
