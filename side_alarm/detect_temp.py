"""
TEMP detector — thermal abnormality from per-chip act_temp roughness.

This detector is deliberately NOT a copy of the RESI logic (spec §8.1). RESI
z-score and std-ratio thresholds are not transferable, because 2026 TEMP
roughness is heavily quantized: values pile onto a handful of levels such as
0.03 / 0.04 / 0.05, a per-side std can be exactly 0, and a raw std ratio then
goes to infinity for a wafer that is perfectly healthy.

The primary screen is therefore quantization-aware: count the chips above the
equipment-side P99 and compare the two sides' SHARES. The legacy std-ratio
screen survives only as an optional corroboration path on equipment where the
quantization diagnostic says it is safe (spec §13).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SideAlarmConfig, DEFAULT_CONFIG, LEFT, RIGHT
from .data_models import INSUFFICIENT_DATA, NORMAL, TEMP_EVENT_NORMAL
from .loader import WAFER_KEYS, aggregate_wafer_side, pivot_sides
from .baseline import BASELINE_KEYS
from .trend import check_equipment_trend


# ---------------------------------------------------------------------------
# 1. Per wafer-side statistics (spec §10)
# ---------------------------------------------------------------------------

def calculate_temp_stats(
    chips: pd.DataFrame,
    temp_baseline: pd.DataFrame | None = None,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    N / mean / median / std / max plus TEMP_HIGH_COUNT and TEMP_HIGH_SHARE.

    "High" means above this equipment-and-side's own P99 — never a global
    threshold (spec §9). With no usable baseline the high count is 0 and the
    share is NaN, which the verdict step reports rather than scoring.
    """
    cfg = config or DEFAULT_CONFIG
    col = cfg.columns.temp_roughness
    work = chips.copy()

    if temp_baseline is not None and not temp_baseline.empty:
        p_col = f"TEMP_P{cfg.temp.high_percentile:g}"
        ref = temp_baseline[list(BASELINE_KEYS) + [p_col, "BASELINE_OK"]]
        work = work.merge(ref, on=list(BASELINE_KEYS), how="left")
        work["_is_high"] = (work[col] > work[p_col]).fillna(False)
        work["_baseline_ok"] = work["BASELINE_OK"].fillna(False)
    else:
        work["_is_high"] = False
        work["_baseline_ok"] = False

    stats = aggregate_wafer_side(
        work, col, "TEMP", cfg,
        extra_aggs={
            "TEMP_HIGH_COUNT": ("_is_high", "sum"),
            "TEMP_BASELINE_AVAILABLE": ("_baseline_ok", "max"),
        },
    )
    stats["TEMP_HIGH_COUNT"] = stats["TEMP_HIGH_COUNT"].fillna(0).astype(int)
    stats["TEMP_BASELINE_AVAILABLE"] = stats["TEMP_BASELINE_AVAILABLE"].fillna(False).astype(bool)

    n = stats["TEMP_N"]
    # TEMP_N == 0 must not raise; an empty side has no share, not a share of 0.
    stats["TEMP_HIGH_SHARE"] = np.where(n > 0, stats["TEMP_HIGH_COUNT"] / n.replace(0, np.nan), np.nan)
    return stats


# ---------------------------------------------------------------------------
# 2. LEFT vs RIGHT comparison (spec §11)
# ---------------------------------------------------------------------------

def _side_value(wide: pd.DataFrame, base: str, suspect: pd.Series, take_suspect: bool) -> pd.Series:
    left, right = wide[f"{base}_{LEFT}"], wide[f"{base}_{RIGHT}"]
    want_left = (suspect == LEFT) if take_suspect else (suspect == RIGHT)
    return left.where(want_left, right)


def compare_temp_sides(
    stats: pd.DataFrame,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    One row per wafer with the candidate suspect side (larger high share,
    mean breaking ties) and the ratios spec §11 tests.
    """
    cfg = config or DEFAULT_CONFIG
    wide = pivot_sides(stats, "TEMP")

    share_l, share_r = wide[f"TEMP_HIGH_SHARE_{LEFT}"], wide[f"TEMP_HIGH_SHARE_{RIGHT}"]
    mean_l, mean_r = wide[f"TEMP_MEAN_{LEFT}"], wide[f"TEMP_MEAN_{RIGHT}"]
    n_l, n_r = wide[f"TEMP_N_{LEFT}"], wide[f"TEMP_N_{RIGHT}"]
    both = n_l.notna() & n_r.notna()

    prefer_left = (share_l.fillna(-1) > share_r.fillna(-1)) | (
        (share_l.fillna(-1) == share_r.fillna(-1)) & (mean_l.fillna(-1) >= mean_r.fillna(-1))
    )
    suspect = pd.Series(
        np.where(both, np.where(prefer_left, LEFT, RIGHT),
                 np.where(n_l.notna(), LEFT, RIGHT)),
        index=wide.index, dtype=object,
    )
    other = pd.Series(np.where(suspect == LEFT, RIGHT, LEFT), index=wide.index, dtype=object)

    out = wide[list(WAFER_KEYS) + ["WRITE_TIME"]].copy()
    out["TEMP_SUSPECT_SIDE"] = suspect
    out["TEMP_OTHER_SIDE"] = other.where(both, None)
    out["SIDE_COMPARISON_AVAILABLE"] = both

    for base in ["TEMP_N", "TEMP_MEAN", "TEMP_MEDIAN", "TEMP_STD", "TEMP_MAX",
                 "TEMP_HIGH_COUNT", "TEMP_HIGH_SHARE", "TEMP_BASELINE_AVAILABLE"]:
        out[f"SUSPECT_{base}"] = _side_value(wide, base, suspect, True)
        out[f"OTHER_{base}"] = _side_value(wide, base, suspect, False).where(both)

    eps = cfg.eps
    out["TEMP_HIGH_SHARE_RATIO"] = out["SUSPECT_TEMP_HIGH_SHARE"] / np.maximum(
        out["OTHER_TEMP_HIGH_SHARE"].fillna(0.0), eps)
    out["TEMP_MEAN_RATIO"] = out["SUSPECT_TEMP_MEAN"] / np.maximum(
        out["OTHER_TEMP_MEAN"].fillna(0.0), eps)
    out["TEMP_STD_RATIO"] = out["SUSPECT_TEMP_STD"] / np.maximum(
        out["OTHER_TEMP_STD"].fillna(0.0), eps)
    return out


# ---------------------------------------------------------------------------
# 3. Suspect verdict (spec §11, §13)
# ---------------------------------------------------------------------------

def detect_temp_anomaly(
    pairs: pd.DataFrame,
    temp_baseline: pd.DataFrame | None = None,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    Flag TEMP_SUSPECT per wafer. Level and event are NOT assigned here — a
    single suspect wafer is not yet an equipment alarm; ``check_equipment_trend``
    decides that (spec §12).
    """
    cfg = config or DEFAULT_CONFIG
    tcfg = cfg.temp
    out = pairs.copy()

    quant_map = {}
    if temp_baseline is not None and not temp_baseline.empty and "TEMP_QUANTIZED" in temp_baseline:
        quant_map = {
            (r.EQP_ID, r.SIDE): bool(r.TEMP_QUANTIZED)
            for r in temp_baseline.itertuples(index=False)
        }
    out["TEMP_QUANTIZED"] = [
        quant_map.get((e, s), False)
        for e, s in zip(out["EQP_ID"], out["TEMP_SUSPECT_SIDE"])
    ]

    share_ok = out["SUSPECT_TEMP_HIGH_SHARE"].fillna(0.0) >= (
        tcfg.high_share_ratio * out["OTHER_TEMP_HIGH_SHARE"].fillna(0.0)
    )
    mean_ok = out["SUSPECT_TEMP_MEAN"].fillna(0.0) >= (
        tcfg.mean_ratio * out["OTHER_TEMP_MEAN"].fillna(0.0)
    )
    count_ok = out["SUSPECT_TEMP_HIGH_COUNT"].fillna(0) >= tcfg.min_high_count
    primary = share_ok & mean_ok & count_ok & out["SIDE_COMPARISON_AVAILABLE"]

    # Legacy corroboration: only where quantization is not dominant (spec §13).
    if tcfg.legacy_enabled and temp_baseline is not None and not temp_baseline.empty:
        p_col = f"TEMP_P{tcfg.legacy_max_percentile:g}"
        legacy_p = _suspect_baseline_value(out, temp_baseline, p_col)
        legacy = (
            ~out["TEMP_QUANTIZED"]
            & out["SIDE_COMPARISON_AVAILABLE"]
            & (out["TEMP_STD_RATIO"] >= tcfg.legacy_std_ratio)
            & (out["SUSPECT_TEMP_MAX"] > legacy_p)
        ).fillna(False)
    else:
        legacy = pd.Series(False, index=out.index)

    out["TEMP_LEGACY_CORROBORATE"] = legacy
    insufficient = out["SUSPECT_TEMP_N"].fillna(0) < cfg.quality.min_temp_chips
    out["TEMP_INSUFFICIENT_DATA"] = insufficient
    out["TEMP_SUSPECT"] = (primary | legacy) & ~insufficient
    return out


def _suspect_baseline_value(out: pd.DataFrame, baseline: pd.DataFrame, col: str) -> pd.Series:
    """Look up one baseline column for each row's suspect side."""
    lut = {(r.EQP_ID, r.SIDE): getattr(r, col) for r in baseline.itertuples(index=False)}
    return pd.Series(
        [lut.get((e, s), np.nan) for e, s in zip(out["EQP_ID"], out["TEMP_SUSPECT_SIDE"])],
        index=out.index, dtype=float,
    )


# ---------------------------------------------------------------------------
# 4. Reasons and the full TEMP stage
# ---------------------------------------------------------------------------

def build_temp_reasons(df: pd.DataFrame, config: SideAlarmConfig | None = None) -> pd.Series:
    """Reason text with measured shares, the other side, and persistence (spec §21)."""
    cfg = config or DEFAULT_CONFIG
    reasons = pd.Series("", index=df.index, dtype=object)

    for i in df.index[(df["TEMP_LEVEL"] != NORMAL) | df["TEMP_INSUFFICIENT_DATA"]]:
        row = df.loc[i]
        if row["TEMP_INSUFFICIENT_DATA"]:
            reasons.loc[i] = (
                f"suspect side chips={int(row['SUSPECT_TEMP_N'] or 0)} "
                f"< min {cfg.quality.min_temp_chips}"
            )
            continue
        side = row["TEMP_SUSPECT_SIDE"]
        other = row["TEMP_OTHER_SIDE"]
        bits = [
            f"{side} high(P{cfg.temp.high_percentile:g}) share="
            f"{row['SUSPECT_TEMP_HIGH_SHARE']:.3f} (count={int(row['SUSPECT_TEMP_HIGH_COUNT'] or 0)})"
        ]
        if pd.notna(row["OTHER_TEMP_HIGH_SHARE"]):
            bits.append(f"{other}={row['OTHER_TEMP_HIGH_SHARE']:.3f}")
        if pd.notna(row["TEMP_MEAN_RATIO"]):
            bits.append(f"mean ratio={row['TEMP_MEAN_RATIO']:.2f}")
        bits.append(f"persisted for {int(row['TEMP_TREND_COUNT'])} wafers "
                    f"(K={cfg.temp.trend_k})")
        if row["TEMP_LEGACY_CORROBORATE"]:
            bits.append("legacy std-ratio corroboration")
        if row["TEMP_QUANTIZED"]:
            bits.append("quantized equipment: std-ratio screen disabled")
        if not row["SUSPECT_TEMP_BASELINE_AVAILABLE"]:
            bits.append("equipment baseline unavailable")
        reasons.loc[i] = ", ".join(bits)

    return reasons


def run_temp_detector(
    chips: pd.DataFrame,
    temp_baseline: pd.DataFrame | None = None,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """stats -> side comparison -> suspect -> equipment trend -> reason."""
    cfg = config or DEFAULT_CONFIG
    stats = calculate_temp_stats(chips, temp_baseline, cfg)
    pairs = compare_temp_sides(stats, cfg)
    flagged = detect_temp_anomaly(pairs, temp_baseline, cfg)
    trended = check_equipment_trend(flagged, cfg)
    trended.loc[trended["TEMP_INSUFFICIENT_DATA"], "TEMP_LEVEL"] = INSUFFICIENT_DATA
    trended.loc[trended["TEMP_INSUFFICIENT_DATA"], "TEMP_EVENT"] = TEMP_EVENT_NORMAL
    trended["TEMP_REASON"] = build_temp_reasons(trended, cfg)
    return trended
