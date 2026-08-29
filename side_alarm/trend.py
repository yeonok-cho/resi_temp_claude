"""
Equipment trend evaluation for TEMP suspects (spec §12, §25).

A single abnormal wafer must never become an equipment alarm on its own. This
module converts a per-wafer suspect flag into one of three events by counting
how many CONSECUTIVE wafers from the same equipment were suspect ON THE SAME
SIDE:

    count == 1              -> WAFER_EVENT      -> CHECK
    2 <= count < TREND_K    -> TRANSIENT        -> WARNING
    count >= TREND_K        -> EQUIPMENT_TREND  -> ALARM

The walk is strictly chronological and never looks forward, so the verdict on
each wafer is exactly what an online run would have produced at that moment
(``ONLINE_RESULT = True``). TREND_K comes from configuration; it is never
hard-coded in this logic.
"""
from __future__ import annotations

import pandas as pd

from .config import SideAlarmConfig, DEFAULT_CONFIG
from .data_models import (
    NORMAL, CHECK, WARNING, ALARM,
    TEMP_EVENT_NORMAL, TEMP_EVENT_WAFER, TEMP_EVENT_TRANSIENT, TEMP_EVENT_TREND,
)


def check_equipment_trend(
    wafers: pd.DataFrame,
    config: SideAlarmConfig | None = None,
    suspect_col: str = "TEMP_SUSPECT",
    side_col: str = "TEMP_SUSPECT_SIDE",
) -> pd.DataFrame:
    """
    Add TEMP_TREND_COUNT / TEMP_EVENT / TEMP_LEVEL / ONLINE_RESULT.

    A suspect wafer on the OTHER side restarts the run at 1 rather than
    extending it: persistence is only meaningful for one physical module.
    """
    cfg = config or DEFAULT_CONFIG
    k = cfg.temp.trend_k

    out = wafers.sort_values(["EQP_ID", "WRITE_TIME", "WAFER_ID"]).copy()

    counts: list[int] = []
    run_eqp = None
    run_side = None
    run_len = 0

    for row in out.itertuples(index=False):
        eqp = getattr(row, "EQP_ID")
        suspect = bool(getattr(row, suspect_col))
        side = getattr(row, side_col)

        if eqp != run_eqp:
            run_eqp, run_side, run_len = eqp, None, 0

        if not suspect:
            run_side, run_len = None, 0
        elif side == run_side:
            run_len += 1
        else:
            run_side, run_len = side, 1

        counts.append(run_len)

    out["TEMP_TREND_COUNT"] = counts

    def classify(n: int) -> tuple[str, str]:
        if n <= 0:
            return TEMP_EVENT_NORMAL, NORMAL
        if n >= k:
            return TEMP_EVENT_TREND, ALARM
        if n >= 2:
            return TEMP_EVENT_TRANSIENT, WARNING
        return TEMP_EVENT_WAFER, CHECK

    classified = [classify(n) for n in counts]
    out["TEMP_EVENT"] = [c[0] for c in classified]
    out["TEMP_LEVEL"] = [c[1] for c in classified]
    out["ONLINE_RESULT"] = True
    return out
