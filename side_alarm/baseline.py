"""
Historical healthy-wafer baselines, computed per (EQP_ID, SIDE).

Absolute thresholds are equipment-specific and side-specific: a global RESI or
TEMP threshold is never used, because both distributions vary strongly across
equipment and process generation (spec §5, §9).

Leakage prevention (spec §24)
-----------------------------
``split_baseline_scoring`` cuts the frame at ``baseline_end``: the baseline is
built only from rows STRICTLY OLDER than that instant, and only rows at or
after it are scored. So for any scored wafer at time t, every baseline row
satisfies ``timestamp < t``. Backtests slide ``baseline_end`` forward instead
of reusing one baseline built over the whole period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SideAlarmConfig, DEFAULT_CONFIG
from .loader import WAFER_KEYS

BASELINE_KEYS = ("EQP_ID", "SIDE")


def split_baseline_scoring(
    df: pd.DataFrame,
    baseline_end,
    lookback_days: int | None = None,
    time_col: str = "WRITE_TIME",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into (baseline rows, scoring rows).

    baseline : [baseline_end - lookback_days, baseline_end)
    scoring  : [baseline_end, end of frame]
    """
    ts = pd.to_datetime(df[time_col])
    cutoff = pd.Timestamp(baseline_end)

    is_baseline = ts < cutoff
    if lookback_days is not None:
        is_baseline &= ts >= cutoff - pd.Timedelta(days=lookback_days)

    return df[is_baseline].copy(), df[~(ts < cutoff)].copy()


def _percentiles(grouped, percentiles, prefix: str) -> pd.DataFrame:
    """Named percentile columns (e.g. RESI_MAX_P95) from one quantile call."""
    qs = [p / 100.0 for p in percentiles]
    out = grouped.quantile(qs).unstack()
    out.columns = [f"{prefix}_P{p:g}" for p in percentiles]
    return out


def _drop_worst(stats: pd.DataFrame, col: str, pct: float) -> pd.DataFrame:
    """Robust cleaning fallback when no problem-wafer labels exist (spec §5)."""
    if pct <= 0:
        return stats
    keep = stats.groupby(list(BASELINE_KEYS), sort=False)[col].transform(
        lambda s: s <= s.quantile(1.0 - pct)
    )
    return stats[keep]


def build_resi_baseline(
    chip_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    config: SideAlarmConfig | None = None,
    exclude_wafer_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Per (EQP_ID, SIDE) RESI baseline.

    Two families of numbers come out of this:

    * chip-level ``RESI_CHIP_MEAN`` / ``RESI_CHIP_STD`` — the reference for the
      3/5/7/9-sigma chip counts (spec §4.2).
    * wafer-side percentiles ``RESI_MAX_P95/P99``, ``RESI_MEAN_P90/P95``,
      ``RESI_STD_P95/P99`` — the absolute thresholds of spec §5.

    ``exclude_wafer_ids`` removes known problem wafers from both.
    """
    cfg = config or DEFAULT_CONFIG
    bcfg = cfg.baseline
    score_col = cfg.columns.resi_score

    chips = chip_df
    stats = stats_df
    if exclude_wafer_ids:
        chips = chips[~chips["WAFER_ID"].astype(str).isin(exclude_wafer_ids)]
        stats = stats[~stats["WAFER_ID"].astype(str).isin(exclude_wafer_ids)]

    stats = _drop_worst(stats, "RESI_MAX", bcfg.exclude_top_pct)

    chip_g = chips.groupby(list(BASELINE_KEYS), sort=False)[score_col]
    out = pd.DataFrame({
        "RESI_CHIP_MEAN": chip_g.mean(),
        "RESI_CHIP_STD": chip_g.std(),
        "N_CHIPS": chip_g.count(),
    })

    stat_g = stats.groupby(list(BASELINE_KEYS), sort=False)
    out = out.join([
        _percentiles(stat_g["RESI_MAX"], bcfg.resi_max_percentiles, "RESI_MAX"),
        _percentiles(stat_g["RESI_MEAN"], bcfg.resi_mean_percentiles, "RESI_MEAN"),
        _percentiles(stat_g["RESI_STD"], bcfg.resi_std_percentiles, "RESI_STD"),
        stat_g.size().rename("N_WAFER_SIDES"),
    ], how="outer")

    out["N_WAFER_SIDES"] = out["N_WAFER_SIDES"].fillna(0).astype(int)
    out["N_CHIPS"] = out["N_CHIPS"].fillna(0).astype(int)
    out["RESI_CHIP_STD"] = out["RESI_CHIP_STD"].fillna(0.0)
    out["BASELINE_OK"] = (
        (out["N_WAFER_SIDES"] >= bcfg.min_baseline_wafers)
        & (out["N_CHIPS"] >= bcfg.min_baseline_chips)
    )
    return out.reset_index()


def detect_temp_quantization(
    chip_df: pd.DataFrame,
    config: SideAlarmConfig | None = None,
) -> pd.DataFrame:
    """
    Quantization diagnostic per (EQP_ID, SIDE) — spec §14.

    2026 TEMP data is heavily quantized (roughness lands on a handful of values
    such as 0.03 / 0.04 / 0.05), which makes a per-side std collapse to 0 and a
    raw std ratio blow up to infinity. Flagging the equipment lets the TEMP
    detector stay on the quantization-aware high-share path and keeps the
    legacy std-ratio corroboration switched off.
    """
    cfg = config or DEFAULT_CONFIG
    tcfg = cfg.temp
    col = cfg.columns.temp_roughness

    rows = []
    for key, grp in chip_df.groupby(list(BASELINE_KEYS), sort=False):
        vals = grp[col].dropna()
        if vals.empty:
            continue
        counts = vals.value_counts()
        top3_share = float(counts.iloc[:3].sum() / len(vals))
        n_unique = int(counts.size)
        rows.append({
            "EQP_ID": key[0],
            "SIDE": key[1],
            "TEMP_UNIQUE_COUNT": n_unique,
            "TEMP_TOP3_SHARE": top3_share,
            "TEMP_QUANTIZED": bool(
                top3_share >= tcfg.quant_top3_share or n_unique <= tcfg.quant_max_unique
            ),
        })

    if not rows:
        return pd.DataFrame(
            columns=["EQP_ID", "SIDE", "TEMP_UNIQUE_COUNT", "TEMP_TOP3_SHARE", "TEMP_QUANTIZED"]
        )
    return pd.DataFrame(rows)


def build_temp_baseline(
    chip_df: pd.DataFrame,
    stats_df: pd.DataFrame | None = None,
    config: SideAlarmConfig | None = None,
    exclude_wafer_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Per (EQP_ID, SIDE) TEMP roughness baseline — spec §9.

    Percentiles are taken over CHIP-level roughness, because TEMP_HIGH_COUNT is
    defined as the number of chips above the equipment-side P99 (spec §10).
    ``APW01 LEFT`` and ``APW01 RIGHT`` therefore get independent P99s.
    """
    cfg = config or DEFAULT_CONFIG
    bcfg = cfg.baseline
    col = cfg.columns.temp_roughness

    chips = chip_df
    if exclude_wafer_ids:
        chips = chips[~chips["WAFER_ID"].astype(str).isin(exclude_wafer_ids)]

    g = chips.groupby(list(BASELINE_KEYS), sort=False)[col]
    out = pd.DataFrame({
        "TEMP_MEAN": g.mean(),
        "TEMP_STD": g.std(),
        "N_CHIPS": g.count(),
    })
    out = out.join(_percentiles(g, bcfg.temp_percentiles, "TEMP"))
    out["TEMP_STD"] = out["TEMP_STD"].fillna(0.0)
    out = out.reset_index()

    if stats_df is not None and not stats_df.empty:
        n_wafers = (
            stats_df.groupby(list(BASELINE_KEYS), sort=False)
            .size().rename("N_WAFER_SIDES").reset_index()
        )
        out = out.merge(n_wafers, on=list(BASELINE_KEYS), how="left")
    else:
        out["N_WAFER_SIDES"] = 0
    out["N_WAFER_SIDES"] = out["N_WAFER_SIDES"].fillna(0).astype(int)

    quant = detect_temp_quantization(chips, cfg)
    if not quant.empty:
        out = out.merge(quant, on=list(BASELINE_KEYS), how="left")
    else:
        out["TEMP_UNIQUE_COUNT"] = np.nan
        out["TEMP_TOP3_SHARE"] = np.nan
        out["TEMP_QUANTIZED"] = False
    out["TEMP_QUANTIZED"] = out["TEMP_QUANTIZED"].fillna(False).astype(bool)

    out["BASELINE_OK"] = (
        (out["N_CHIPS"] >= bcfg.min_baseline_chips)
        & (out["N_WAFER_SIDES"] >= bcfg.min_baseline_wafers)
    )
    return out
