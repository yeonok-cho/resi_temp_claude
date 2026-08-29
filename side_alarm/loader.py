"""
Chip-level ingestion and wafer x side aggregation.

The source table already carries both per-chip scores this package consumes:
``mp_score`` (RESI anomaly score) and a TEMP roughness column. Neither is
recomputed here — the upstream definitions are preserved (spec §31.6).

Everything below aggregates with a single ``groupby`` per signal rather than
``iterrows``/``apply``, so several million chip rows stay tractable (spec §32).
"""
from __future__ import annotations

import struct

import numpy as np
import pandas as pd

from .config import SideAlarmConfig, DEFAULT_CONFIG, LEFT, RIGHT


# ---------------------------------------------------------------------------
# Side normalization (spec §3)
# ---------------------------------------------------------------------------

def normalize_side(values: pd.Series, config: SideAlarmConfig | None = None) -> pd.Series:
    """
    Map the raw ``module_id`` column onto LEFT / RIGHT.

    Unrecognized values become NaN so the caller can drop or report them; the
    comparison never silently treats an unknown module as a side.
    """
    cfg = (config or DEFAULT_CONFIG).side
    norm = values.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=values.index, dtype=object)
    out[norm.isin(cfg.left_values)] = LEFT
    out[norm.isin(cfg.right_values)] = RIGHT
    return out


def attach_side(df: pd.DataFrame, config: SideAlarmConfig | None = None) -> pd.DataFrame:
    """Add a normalized ``SIDE`` column, dropping unmapped module_id rows."""
    cfg = config or DEFAULT_CONFIG
    out = df.copy()
    out["SIDE"] = normalize_side(out[cfg.columns.module_id], cfg)

    n_bad = int(out["SIDE"].isna().sum())
    if n_bad and cfg.side.drop_unmapped:
        bad_values = sorted(set(out.loc[out["SIDE"].isna(), cfg.columns.module_id].astype(str)))[:5]
        print(f"  [side] module_id 미매핑 {n_bad} rows 제외 (예: {bad_values})")
        out = out[out["SIDE"].notna()]
    return out


# ---------------------------------------------------------------------------
# ClickHouse ingestion
# ---------------------------------------------------------------------------

def build_query(start: str, end: str, config: SideAlarmConfig | None = None,
                extra_columns: list[str] | None = None) -> str:
    """SELECT the chip-level columns for a half-open-ish [start, end] window."""
    cfg = config or DEFAULT_CONFIG
    cols = cfg.columns.chip_columns() + list(extra_columns or [])
    seen: list[str] = []
    for c in cols:                        # preserve order, drop duplicates
        if c not in seen:
            seen.append(c)
    return f"""
    SELECT {', '.join(seen)}
    FROM {cfg.columns.table}
    WHERE {cfg.columns.write_time} > '{start}' AND {cfg.columns.write_time} <= '{end}'
    ORDER BY {cfg.columns.eqp_id}, {cfg.columns.product}, {cfg.columns.write_time}
    """


def query_chips(client, start: str, end: str, config: SideAlarmConfig | None = None,
                extra_columns: list[str] | None = None) -> pd.DataFrame:
    """Run build_query and return a DataFrame (empty frame on query failure)."""
    q = build_query(start, end, config, extra_columns)
    try:
        return client.query_dataframe(q)
    except Exception as e:                # noqa: BLE001 - surfaced to the operator
        print(f"  [쿼리 오류] {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# TEMP roughness fallback
# ---------------------------------------------------------------------------

def decode_hex_to_float32(hex_str) -> np.ndarray:
    """Same decoder as temp_anomaly/scripts/_common.py."""
    if not hex_str:
        return np.array([])
    try:
        raw = bytes.fromhex(hex_str)
    except (ValueError, TypeError):
        return np.array([])
    n = len(raw) // 4
    return np.array(
        [struct.unpack('<f', raw[i * 4:(i + 1) * 4])[0] for i in range(n)],
        dtype=np.float32,
    )


def compute_roughness_from_raw(df: pd.DataFrame, config: SideAlarmConfig | None = None) -> pd.DataFrame:
    """
    Fallback when the source table has no precomputed roughness column.

    Reproduces the repo's definition exactly: the reference profile is the
    pointwise median across the wafer-side's chips (as in
    ``reference_manager``), and roughness is ``std(diff(residual))`` (as in
    ``feature_extractor.compute_roughness``).

    This decodes one hex blob per chip and is therefore far slower than reading
    a stored column; prefer supplying ``columns.temp_roughness``.
    """
    cfg = config or DEFAULT_CONFIG
    cols = cfg.columns
    out = df.copy()
    out[cols.temp_roughness] = np.nan

    keys = [cols.eqp_id, cols.product, cols.wafer_id, "SIDE"]
    for _, idx in out.groupby(keys, sort=False).groups.items():
        series = [decode_hex_to_float32(v) for v in out.loc[idx, cols.temp_raw]]
        lengths = [len(s) for s in series if len(s) > 0]
        if not lengths:
            continue
        n = min(lengths)
        stack = np.stack([s[:n] for s in series if len(s) >= n])
        reference = np.median(stack, axis=0)

        values = []
        for s in series:
            if len(s) < 2:
                values.append(np.nan)
                continue
            residual = s[:n] - reference if len(s) >= n else s - reference[:len(s)]
            values.append(float(np.std(np.diff(residual))) if len(residual) >= 2 else np.nan)
        out.loc[idx, cols.temp_roughness] = values

    return out


def ensure_temp_roughness(df: pd.DataFrame, config: SideAlarmConfig | None = None,
                          allow_fallback: bool = True) -> pd.DataFrame:
    """Return a frame guaranteed to carry the TEMP roughness column."""
    cfg = config or DEFAULT_CONFIG
    cols = cfg.columns
    if cols.temp_roughness in df.columns:
        return df
    if not allow_fallback or cols.temp_raw not in df.columns:
        raise KeyError(
            f"'{cols.temp_roughness}' 컬럼이 없고 '{cols.temp_raw}'로 대체할 수도 없습니다. "
            f"사용 가능한 컬럼: {sorted(df.columns)}"
        )
    print(f"  [temp] '{cols.temp_roughness}' 없음 → '{cols.temp_raw}'에서 roughness 계산 (느림)")
    return compute_roughness_from_raw(df, cfg)


# ---------------------------------------------------------------------------
# Wafer x side aggregation (spec §4.2, §10)
# ---------------------------------------------------------------------------

WAFER_KEYS = ("EQP_ID", "PRODUCT", "WAFER_ID")


def _rename_keys(df: pd.DataFrame, config: SideAlarmConfig) -> pd.DataFrame:
    cols = config.columns
    return df.rename(columns={
        cols.eqp_id: "EQP_ID",
        cols.product: "PRODUCT",
        cols.wafer_id: "WAFER_ID",
        cols.write_time: "WRITE_TIME",
    })


def aggregate_wafer_side(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
    config: SideAlarmConfig | None = None,
    extra_aggs: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Collapse chip rows to one row per (EQP_ID, PRODUCT, WAFER_ID, SIDE).

    One groupby pass produces N / mean / median / std / max / skew plus the
    wafer-side's WRITE_TIME (the latest chip timestamp, which is what orders
    wafers for trend evaluation — spec §25).

    ``extra_aggs`` adds further named aggregations as ``{out_name: (col, how)}``,
    which is how the SD counts and TEMP high counts ride along in the same pass.
    """
    cfg = config or DEFAULT_CONFIG
    work = _rename_keys(df, cfg)
    keys = list(WAFER_KEYS) + ["SIDE"]

    aggs: dict[str, tuple[str, str]] = {
        f"{prefix}_N": (value_col, "count"),
        f"{prefix}_MEAN": (value_col, "mean"),
        f"{prefix}_MEDIAN": (value_col, "median"),
        f"{prefix}_STD": (value_col, "std"),
        f"{prefix}_MAX": (value_col, "max"),
        f"{prefix}_SKEW": (value_col, "skew"),
        "WRITE_TIME": ("WRITE_TIME", "max"),
    }
    if extra_aggs:
        aggs.update(extra_aggs)

    out = work.groupby(keys, sort=False, dropna=False).agg(**aggs).reset_index()
    # ddof=1 std is NaN for a single chip; 0.0 is the honest spread there and
    # keeps ratio math finite (the min-chip guard rejects such rows anyway).
    out[f"{prefix}_STD"] = out[f"{prefix}_STD"].fillna(0.0)
    return out


def pivot_sides(stats: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Turn one row per wafer-side into one row per wafer with _LEFT/_RIGHT columns.

    This is the join the existing code never performs: ``_common.df_to_wafers``
    puts module_id in the group key, so a physical wafer's two sides end up in
    different GroupKeys and never meet.
    """
    value_cols = [c for c in stats.columns if c.startswith(f"{prefix}_")]
    wide = stats.pivot_table(
        index=list(WAFER_KEYS),
        columns="SIDE",
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [f"{col}_{side}" for col, side in wide.columns]

    times = stats.groupby(list(WAFER_KEYS), sort=False)["WRITE_TIME"].max()
    wide = wide.join(times)

    for side in (LEFT, RIGHT):
        for col in value_cols:
            name = f"{col}_{side}"
            if name not in wide.columns:
                wide[name] = np.nan

    return wide.reset_index()


def prepare_chips(df: pd.DataFrame, config: SideAlarmConfig | None = None) -> pd.DataFrame:
    """Normalize module_id to SIDE and rename the identity columns to the
    canonical EQP_ID / PRODUCT / WAFER_ID / WRITE_TIME names used downstream."""
    cfg = config or DEFAULT_CONFIG
    out = attach_side(df, cfg)
    out = _rename_keys(out, cfg)
    out["WRITE_TIME"] = pd.to_datetime(out["WRITE_TIME"])
    out["WAFER_ID"] = out["WAFER_ID"].astype(str)
    return out


def keep_latest_measurement(chips: pd.DataFrame, config: SideAlarmConfig | None = None) -> pd.DataFrame:
    """
    Collapse re-measurements of the same chip to the latest one (spec §25).

    Keyed by (EQP_ID, PRODUCT, WAFER_ID, SIDE, x, y); the row with the largest
    WRITE_TIME wins. Skipped when the coordinate columns are absent.
    """
    cfg = config or DEFAULT_CONFIG
    coords = [cfg.columns.x, cfg.columns.y]
    if not all(c in chips.columns for c in coords):
        return chips
    keys = ["EQP_ID", "PRODUCT", "WAFER_ID", "SIDE"] + coords
    return (
        chips.sort_values("WRITE_TIME")
        .drop_duplicates(subset=keys, keep="last")
        .reset_index(drop=True)
    )
