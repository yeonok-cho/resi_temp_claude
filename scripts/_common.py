"""
Shared ClickHouse / temp_vibration_detection utilities used by the
build_baseline, run_anomaly_detection, and ad-hoc run scripts.

Keeping these in one place avoids drift between scripts that all need to:
  - decode the hex-encoded temp_raw column
  - build ChipData / WaferData objects from a DataFrame
  - group rows into per-GroupKey, chronologically-ordered wafer lists
"""
from __future__ import annotations

import struct
from datetime import datetime

import numpy as np
import pandas as pd
from clickhouse_driver import Client

from temp_vibration_detection.data_models import WaferData, ChipData, GroupKey

TABLE = "ds_catalog.aifpa_cow_rule_hist"


def decode_hex_to_float32(hex_str) -> np.ndarray:
    if not hex_str:
        return np.array([])
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError:
        return np.array([])
    n = len(raw) // 4
    return np.array(
        [struct.unpack('<f', raw[i * 4:(i + 1) * 4])[0] for i in range(n)],
        dtype=np.float32,
    )


def create_chip(row, bond_order: int) -> ChipData:
    return ChipData(
        equipment_id=row.get('eqp_id', 'unknown'),
        recipe=row.get('product', 'unknown'),
        head=row.get('module_id', row.get('head', 'unknown')),
        wafer_id=str(row.get('wafer_id', 'unknown')),
        x=int(row.get('x', 0)),
        y=int(row.get('y', 0)),
        bond_order=bond_order,
        temp_series=decode_hex_to_float32(row.get('temp_raw', '')),
    )


def df_to_wafers(df: pd.DataFrame) -> dict[GroupKey, list[tuple]]:
    """
    DataFrame -> {GroupKey: [(first_event_ts, WaferData), ...]}, sorted oldest-first.

    Groups rows by (eqp_id, product, module_id, wafer_id) so that wafers from
    different equipment/recipe/head combinations never collide even if they
    happen to share a wafer_id. Chips within each wafer are ordered by
    event_tmstp (bond order).
    """
    if df.empty:
        return {}

    group_cols = [c for c in ['eqp_id', 'product', 'module_id', 'wafer_id'] if c in df.columns]
    result: dict[GroupKey, list[tuple]] = {}

    for _, wafer_df in df.groupby(group_cols, sort=False):
        wafer_df = wafer_df.sort_values('event_tmstp')
        chips = []
        for _, row in wafer_df.iterrows():
            chip = create_chip(row, len(chips))
            if len(chip.temp_series) > 0:
                chips.append(chip)
        if not chips:
            continue
        first_ts = wafer_df['event_tmstp'].iloc[0]
        wafer_id_val = str(wafer_df['wafer_id'].iloc[0])
        gk = chips[0].group_key
        result.setdefault(gk, []).append(
            (first_ts, WaferData(wafer_id=wafer_id_val, chips=chips))
        )

    return {gk: sorted(lst, key=lambda x: x[0]) for gk, lst in result.items()}


def query_window(client: Client, start: str, end: str) -> pd.DataFrame:
    """Query all rows in [start, end] (inclusive), ordered for stable grouping."""
    q = f"""
    SELECT eqp_id, product, module_id, wafer_id,
           x, y, temp_raw, event_tmstp
    FROM {TABLE}
    WHERE event_tmstp > '{start}' AND event_tmstp <= '{end}'
    ORDER BY eqp_id, product, module_id, event_tmstp
    """
    try:
        return client.query_dataframe(q)
    except Exception as e:
        print(f"  [쿼리 오류] {e}")
        return pd.DataFrame()


def group_key_to_str(gk: GroupKey) -> str:
    return f"{gk.equipment_id}|{gk.recipe}|{gk.head}"
