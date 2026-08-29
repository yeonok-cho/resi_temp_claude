"""
Batch RESI / TEMP side-alarm run.

    chip rows (mp_score, temp roughness)
        -> wafer x side aggregation
        -> independent RESI detector
        -> independent TEMP detector + equipment trend
        -> severity fusion

Both detectors run over the SAME chip frame but share no thresholds and no
state; the fusion layer is the only place their verdicts meet, and it only
ever escalates (spec §2, §16).

Leakage (spec §24): the baseline is built from ``--baseline-lookback-days``
of data ending at ``--baseline-end``, and only wafers at or after that instant
are scored, so every scored wafer is compared against strictly older data.

Usage
-----
  # against ClickHouse (CLICKHOUSE_HOST / _USER / _PASSWORD in the environment)
  python side_alarm/scripts/run_side_alarm.py \
      --start '2026-08-01 00:00:00' --end '2026-08-29 00:00:00' \
      --baseline-end '2026-08-22 00:00:00' \
      --output-dir /workspace/act_resi/sw_act_resi_anomaly/output

  # against an exported chip-level CSV (no DB needed)
  python side_alarm/scripts/run_side_alarm.py --input-csv chips.csv \
      --baseline-end '2026-08-22 00:00:00' --output-dir ./out
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from side_alarm.config import DEFAULT_CONFIG, SideAlarmConfig
from side_alarm.data_models import NORMAL
from side_alarm.loader import prepare_chips, keep_latest_measurement, query_chips, ensure_temp_roughness
from side_alarm.baseline import build_resi_baseline, build_temp_baseline, split_baseline_scoring
from side_alarm.detect_resi import calculate_resi_stats, run_resi_detector
from side_alarm.detect_temp import calculate_temp_stats, run_temp_detector
from side_alarm.fuse_alarm import (
    fuse_frames, RESI_OUTPUT_COLUMNS, TEMP_OUTPUT_COLUMNS,
)

log = logging.getLogger("side_alarm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RESI/TEMP side asymmetry alarm run")
    p.add_argument("--input-csv", type=str, default=None,
                   help="chip-level CSV 입력 (지정 시 ClickHouse 미사용)")
    p.add_argument("--start", type=str, default=None, help="조회 시작 (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--end", type=str, default=None, help="조회 종료 (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--baseline-end", type=str, required=True,
                   help="baseline 종료 시각. 이 시각 이후 wafer만 채점된다 (누수 방지)")
    p.add_argument("--baseline-lookback-days", type=int,
                   default=DEFAULT_CONFIG.baseline.lookback_days,
                   help=f"baseline 참조 기간 (default: {DEFAULT_CONFIG.baseline.lookback_days}일)")
    p.add_argument("--output-dir", type=str, default="./side_alarm_output")
    p.add_argument("--exclude-wafers", type=str, default=None,
                   help="baseline에서 제외할 문제 wafer_id 목록 파일 (한 줄에 하나)")
    p.add_argument("--keep-latest", action="store_true",
                   help="같은 칩의 재측정이 있으면 최신 것만 사용")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def load_chips(args: argparse.Namespace, config: SideAlarmConfig) -> pd.DataFrame:
    """Read chip rows from a CSV export or from ClickHouse."""
    if args.input_csv:
        log.info("CSV 입력: %s", args.input_csv)
        return pd.read_csv(args.input_csv)

    if not (args.start and args.end):
        raise SystemExit("--input-csv 를 쓰지 않으려면 --start / --end 가 필요합니다")

    try:
        from clickhouse_driver import Client
    except ImportError as e:
        raise SystemExit(f"clickhouse_driver 가 설치되어 있지 않습니다: {e}")

    host = os.environ.get("CLICKHOUSE_HOST")
    if not host:
        raise SystemExit("CLICKHOUSE_HOST 환경변수가 필요합니다")
    client = Client(
        host=host,
        user=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
    )
    log.info("ClickHouse 조회: [%s ~ %s]", args.start, args.end)
    return query_chips(client, args.start, args.end, config,
                       extra_columns=[config.columns.temp_roughness])


def log_wafer_verdicts(resi: pd.DataFrame, temp: pd.DataFrame, fused: pd.DataFrame) -> None:
    """
    Wafer-level summaries only — chip rows are never logged (spec §29).
    """
    for r in resi[resi["RESI_LEVEL"] != NORMAL].itertuples(index=False):
        ratio = r.RESI_STD_RATIO
        ratio_txt = f"{ratio:.2f}" if pd.notna(ratio) else "n/a"
        log.info("[RESI] %s %s suspect=%s std_ratio=%s level=%s",
                 r.EQP_ID, r.WAFER_ID, r.RESI_SUSPECT_SIDE, ratio_txt, r.RESI_LEVEL)

    for t in temp[temp["TEMP_LEVEL"] != NORMAL].itertuples(index=False):
        log.info("[TEMP] %s %s suspect=%s event=%s level=%s trend=%d",
                 t.EQP_ID, t.WAFER_ID, t.TEMP_SUSPECT_SIDE, t.TEMP_EVENT,
                 t.TEMP_LEVEL, int(t.TEMP_TREND_COUNT or 0))

    for f in fused[fused["FINAL_SEVERITY"] != NORMAL].itertuples(index=False):
        log.info("[FUSION] %s %s severity=%s temp_support=%s",
                 f.EQP_ID, f.WAFER_ID, f.FINAL_SEVERITY, f.TEMP_SUPPORTS_RESI)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(message)s")
    config = DEFAULT_CONFIG
    config.baseline.lookback_days = args.baseline_lookback_days
    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_chips(args, config)
    if raw.empty:
        log.warning("입력 데이터가 없습니다 — 종료")
        return

    chips = prepare_chips(raw, config)
    chips = ensure_temp_roughness(chips, config)
    if args.keep_latest:
        before = len(chips)
        chips = keep_latest_measurement(chips, config)
        log.info("재측정 정리: %d -> %d rows", before, len(chips))

    exclude: set[str] = set()
    if args.exclude_wafers and os.path.exists(args.exclude_wafers):
        with open(args.exclude_wafers) as f:
            exclude = {line.strip() for line in f if line.strip()}
        log.info("baseline 제외 wafer: %d개", len(exclude))

    hist, current = split_baseline_scoring(
        chips, args.baseline_end, config.baseline.lookback_days
    )
    log.info("baseline 구간 %d rows / 채점 구간 %d rows", len(hist), len(current))
    if hist.empty or current.empty:
        log.warning("baseline 또는 채점 구간이 비어 있습니다 — 종료")
        return

    # --- baselines, per (EQP_ID, SIDE) -------------------------------------
    resi_baseline = build_resi_baseline(
        hist, calculate_resi_stats(hist, None, config), config, exclude)
    temp_baseline = build_temp_baseline(
        hist, calculate_temp_stats(hist, None, config), config, exclude)
    log.info("baseline: RESI %d개 / TEMP %d개 (EQP x SIDE), quantized=%d",
             len(resi_baseline), len(temp_baseline),
             int(temp_baseline["TEMP_QUANTIZED"].sum()))

    # --- two independent detectors -----------------------------------------
    resi_result = run_resi_detector(current, resi_baseline, config)
    temp_result = run_temp_detector(current, temp_baseline, config)

    # --- fusion -------------------------------------------------------------
    fused = fuse_frames(resi_result, temp_result, config)

    log_wafer_verdicts(resi_result, temp_result, fused)

    def write(df: pd.DataFrame, name: str, columns: list[str] | None = None) -> None:
        path = os.path.join(args.output_dir, name)
        (df[columns] if columns else df).to_csv(path, index=False)
        log.info("저장: %s (%d rows)", path, len(df))

    write(resi_result, "resi_anomaly_result.csv",
          [c for c in RESI_OUTPUT_COLUMNS if c in resi_result.columns])
    write(temp_result, "temp_anomaly_result.csv",
          [c for c in TEMP_OUTPUT_COLUMNS if c in temp_result.columns])
    write(fused, "final_alarm_result.csv")
    write(resi_baseline, "equipment_resi_baseline.csv")
    write(temp_baseline, "equipment_temp_baseline.csv")

    counts = fused["FINAL_SEVERITY"].value_counts()
    log.info("=" * 60)
    log.info("채점 wafer: %d", len(fused))
    for level, n in counts.items():
        log.info("  %-18s %d", level, n)
    log.info("실행 시각: %s", datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
