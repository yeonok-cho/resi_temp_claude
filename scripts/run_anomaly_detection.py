"""
Incremental, idempotent temperature-vibration anomaly detection.

Designed to be invoked every 12h by an Airflow DAG (see dags/temp_vibration_dag.py),
but safe to run manually / re-run after a failure.

On each run
-----------
1. Load persisted pipeline state (reference profiles + inter-wafer EWMA) from
   STATE_DIR/baseline.json, if present.
2. Load the watermark (last processed event_tmstp) from STATE_DIR/watermark.json.
   - If the watermark file does NOT exist (very first run ever), bootstrap by
     scanning backwards week-by-week (up to --bootstrap-lookback-days) so that
     groups with enough recent history start "ready" immediately, instead of
     waiting --min-wafers x 12h to warm up from scratch.
   - Otherwise, only the NEW data since the watermark is queried — already
     "built" groups are never re-scanned, so a run only costs as much as the
     last 12h of data.
3. For every group_key, process new wafers in chronological order:
     - status "building"  (ref.n_wafers_seen < MIN_BASELINE_WAFERS before this
       wafer): pipeline.process_wafer() still runs (so reference / inter-wafer
       state keeps accumulating), but any returned events are discarded —
       the group hasn't earned a trustworthy baseline yet.
     - status "ready"     (ref.n_wafers_seen >= MIN_BASELINE_WAFERS): events
       are kept and reported.
   A group crosses from "building" to "ready" automatically, mid-run, the
   moment its wafer count passes the threshold — no separate "promotion" step.
4. Persist updated pipeline state + watermark, write this run's anomalies to
   CSV, and print a building/ready summary.

Usage
-----
  python scripts/run_anomaly_detection.py \\
      --state-dir /workspace/state \\
      --output-dir /workspace/act_resi/sw_act_resi_anomaly/output \\
      --min-wafers 5
"""
from __future__ import annotations

import sys
sys.path.append('/workspace')

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from clickhouse_driver import Client

from temp_vibration_detection.pipeline import TempVibrationPipeline
from temp_vibration_detection.config import DEFAULT_CONFIG
from temp_vibration_detection.data_models import AnomalyEvent

from _common import df_to_wafers, query_window, group_key_to_str

# ──────────────────────────────────────────────
# CLI 인자
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Incremental temperature vibration anomaly detection")
parser.add_argument("--state-dir", type=str, default="/workspace/state",
                    help="baseline.json / watermark.json 저장 위치")
parser.add_argument("--output-dir", type=str,
                    default="/workspace/act_resi/sw_act_resi_anomaly/output",
                    help="이번 run의 anomaly CSV / summary 출력 위치")
parser.add_argument("--min-wafers", type=int, default=5,
                    help="그룹이 'ready' 상태가 되기 위한 최소 wafer 수 (default: 5)")
parser.add_argument("--bootstrap-lookback-days", type=int, default=30,
                    help="watermark가 없는 최초 실행 시 최대 과거 탐색 일수 (default: 30)")
parser.add_argument("--bootstrap-chunk-days", type=int, default=7,
                    help="최초 실행 시 과거 탐색 청크 크기 (default: 7)")
args = parser.parse_args()

MIN_BASELINE_WAFERS: int = args.min_wafers

os.makedirs(args.state_dir, exist_ok=True)
BASELINE_PATH = os.path.join(args.state_dir, "baseline.json")
WATERMARK_PATH = os.path.join(args.state_dir, "watermark.json")

RUN_TS = datetime.now()
RUN_TIMESTAMP = RUN_TS.strftime("%Y%m%d_%H%M%S")
run_output_dir = os.path.join(args.output_dir, f"run_{RUN_TIMESTAMP}")
os.makedirs(run_output_dir, exist_ok=True)

# ──────────────────────────────────────────────
# ClickHouse 연결 (환경에 맞게 수정)
# ──────────────────────────────────────────────
client = Client(host=host, user=user, password=password)

# ──────────────────────────────────────────────
# 1. 파이프라인 상태 로드 (없으면 빈 상태로 시작)
# ──────────────────────────────────────────────
pipeline = TempVibrationPipeline(config=DEFAULT_CONFIG)
if os.path.exists(BASELINE_PATH):
    pipeline.load_baseline(BASELINE_PATH)
    print(f"기존 baseline 로드: {BASELINE_PATH}")
else:
    print("기존 baseline 없음 — 새로 시작")

# ──────────────────────────────────────────────
# 2. watermark 로드 (없으면 최초 실행 → 과거 부트스트랩)
# ──────────────────────────────────────────────
is_first_run = not os.path.exists(WATERMARK_PATH)
new_wafers: dict = {}  # {GroupKey: [(ts, WaferData), ...]} oldest-first

if is_first_run:
    print(f"watermark 없음 — 최초 실행: 최대 {args.bootstrap_lookback_days}일 과거 부트스트랩")
    today = date.today()
    chunk_end = today  # 오늘 포함
    earliest_allowed = today - timedelta(days=args.bootstrap_lookback_days)
    satisfied: set = set()

    n_chunks = (args.bootstrap_lookback_days + args.bootstrap_chunk_days - 1) // args.bootstrap_chunk_days
    for _ in range(n_chunks):
        chunk_start = chunk_end - timedelta(days=args.bootstrap_chunk_days - 1)
        if chunk_start < earliest_allowed:
            chunk_start = earliest_allowed

        s = f"{chunk_start} 00:00:00"
        e = RUN_TS.strftime("%Y-%m-%d %H:%M:%S") if chunk_end == today else f"{chunk_end} 23:59:59"
        print(f"  [{s} ~ {e}] 쿼리 중...", end=" ", flush=True)

        df_chunk = query_window(client, s, e)
        chunk_wafers = df_to_wafers(df_chunk)
        del df_chunk

        for gk, ts_list in chunk_wafers.items():
            new_wafers.setdefault(gk, [])
            new_wafers[gk] = ts_list + new_wafers[gk]  # prepend (older first)

        for gk, lst in new_wafers.items():
            if len(lst) >= MIN_BASELINE_WAFERS:
                satisfied.add(gk)

        print(f"groups={len(new_wafers)}, satisfied={len(satisfied)}/{len(new_wafers)}")

        if satisfied and satisfied == set(new_wafers):
            print("  → 모든 그룹 충족, 부트스트랩 조기 종료")
            break

        chunk_end = chunk_start - timedelta(days=1)
        if chunk_end < earliest_allowed:
            break

    watermark_start = RUN_TS  # 부트스트랩 끝난 시점부터 watermark 시작
else:
    with open(WATERMARK_PATH) as f:
        wm = json.load(f)
    watermark_start = datetime.fromisoformat(wm["last_processed_ts"])
    print(f"watermark: {watermark_start.isoformat()} 이후 데이터만 조회")

    s = watermark_start.strftime("%Y-%m-%d %H:%M:%S")
    e = RUN_TS.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{s} ~ {e}] 쿼리 중...", end=" ", flush=True)
    df_new = query_window(client, s, e)
    new_wafers = df_to_wafers(df_new)
    del df_new
    print(f"groups={len(new_wafers)}, "
          f"wafers={sum(len(v) for v in new_wafers.values())}")

print()

# ──────────────────────────────────────────────
# 3. 그룹별로 신규 wafer 순차 처리 (building/ready 자동 판별)
# ──────────────────────────────────────────────
all_events: list[AnomalyEvent] = []
group_summary = []  # 리포트용

for gk in sorted(new_wafers.keys()):
    ts_wafer_list = new_wafers[gk]
    ref = pipeline._ref_manager.get_or_create(gk)

    n_building = 0
    n_ready = 0
    group_events = 0

    for _, wafer in ts_wafer_list:
        was_building = ref.n_wafers_seen < MIN_BASELINE_WAFERS
        events = pipeline.process_wafer(wafer, update_reference=True)

        if was_building:
            n_building += 1
            # 아직 baseline 확립 전 — 탐지 결과는 신뢰하지 않고 버림
            continue

        n_ready += 1
        if events:
            all_events.extend(events)
            group_events += len(events)

    status = "ready" if ref.n_wafers_seen >= MIN_BASELINE_WAFERS else "building"
    group_summary.append({
        "group_key": group_key_to_str(gk),
        "status": status,
        "n_wafers_seen_total": ref.n_wafers_seen,
        "n_new_wafers_this_run": len(ts_wafer_list),
        "n_building_this_run": n_building,
        "n_ready_this_run": n_ready,
        "anomalies_this_run": group_events,
    })

    print(f"  {gk}  status={status}  n_wafers_seen={ref.n_wafers_seen}  "
          f"(이번 run: building={n_building}, ready={n_ready}, anomalies={group_events})")

print()

# ──────────────────────────────────────────────
# 4. 결과 저장
# ──────────────────────────────────────────────
# 4a. anomaly CSV
if all_events:
    anomaly_df = pd.DataFrame([{
        'eqp_id': e.group_key.equipment_id,
        'recipe': e.group_key.recipe,
        'head': e.group_key.head,
        'wafer_id': e.wafer_id,
        'anomaly_type': e.anomaly_type,
        'severity': e.severity,
        'details': str(e.details),
    } for e in all_events])
    anomaly_csv_path = os.path.join(run_output_dir, "anomalies.csv")
    anomaly_df.to_csv(anomaly_csv_path, index=False)
    print(f"anomaly 저장: {anomaly_csv_path} ({len(all_events)} events)")
else:
    print("이번 run에서 탐지된 anomaly 없음")

# 4b. 그룹 상태 요약
summary_path = os.path.join(run_output_dir, "group_summary.csv")
pd.DataFrame(group_summary).to_csv(summary_path, index=False)
print(f"그룹 상태 요약: {summary_path}")

# 4c. 파이프라인 상태 저장 (다음 run에서 이어서 사용)
pipeline.save_baseline(BASELINE_PATH)
print(f"baseline 상태 저장: {BASELINE_PATH}")

# 4d. watermark 갱신
with open(WATERMARK_PATH, "w") as f:
    json.dump({"last_processed_ts": RUN_TS.isoformat()}, f, indent=2)
print(f"watermark 갱신: {RUN_TS.isoformat()}")

# ──────────────────────────────────────────────
# 5. 전체 요약
# ──────────────────────────────────────────────
n_ready_groups = sum(1 for g in group_summary if g["status"] == "ready")
n_building_groups = len(group_summary) - n_ready_groups
print()
print("=" * 60)
print(f"groups 처리       : {len(group_summary)} "
      f"(ready={n_ready_groups}, building={n_building_groups})")
print(f"신규 wafer 총합   : {sum(g['n_new_wafers_this_run'] for g in group_summary)}")
print(f"탐지된 anomaly    : {len(all_events)}")
print(f"출력 디렉토리     : {run_output_dir}")
print("=" * 60)
