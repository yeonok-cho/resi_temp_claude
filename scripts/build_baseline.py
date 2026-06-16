"""
Build per-GroupKey baseline JSON for temp_vibration_detection.

Strategy per GroupKey
---------------------
최근 데이터부터 주(week) 단위로 역순 쿼리해서 그룹별로 MIN_BASELINE_WAFERS장이
채워지면 해당 그룹은 더 이상 과거를 조회하지 않는다. 최대 MAX_LOOKBACK_DAYS까지
탐색한다.

- 역사 wafer >= MIN_BASELINE_WAFERS → reference + inter-wafer 완전 baseline
- 역사 wafer < MIN_BASELINE_WAFERS  → 역사분 먼저 쌓고, 오늘 데이터로 reference만
  보완 (inter-wafer tracker 건너뜀). production wafer 유입 시 intra 탐지는 즉시 작동,
  inter 탐지는 production wafer가 쌓이면서 점진적으로 활성화.
- 역사 wafer = 0                    → 오늘 데이터로 reference만 구축 (intra only)

Output
------
A single JSON file loadable by TempVibrationPipeline.load_baseline(path).

Usage
-----
  python scripts/build_baseline.py --output /workspace/baseline.json
  python scripts/build_baseline.py --output /workspace/baseline.json --min-wafers 5
"""
from __future__ import annotations

import sys
sys.path.append('/workspace')

import argparse
import struct
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from clickhouse_driver import Client

from temp_vibration_detection.pipeline import TempVibrationPipeline
from temp_vibration_detection.data_models import WaferData, ChipData, GroupKey
from temp_vibration_detection.config import DEFAULT_CONFIG

# ──────────────────────────────────────────────
# ClickHouse 연결 (환경에 맞게 수정)
# ──────────────────────────────────────────────
client = Client(host=host, user=user, password=password)

# ──────────────────────────────────────────────
# CLI 인자
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Build temperature vibration baseline JSON")
parser.add_argument("--output", type=str, default="/workspace/baseline.json",
                    help="저장할 JSON 파일 경로 (default: /workspace/baseline.json)")
parser.add_argument("--min-wafers", type=int, default=5,
                    help="그룹당 최소 역사 wafer 수 (default: 5)")
parser.add_argument("--max-lookback-days", type=int, default=30,
                    help="최대 과거 탐색 일수 (default: 30)")
parser.add_argument("--chunk-days", type=int, default=7,
                    help="한 번에 쿼리할 일수 단위 (default: 7)")
args = parser.parse_args()

MIN_BASELINE_WAFERS: int = args.min_wafers
MAX_LOOKBACK_DAYS: int = args.max_lookback_days
CHUNK_DAYS: int = args.chunk_days

today = date.today()
TODAY_START = f"{today} 00:00:00"
TODAY_END   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"최소 wafer        : {MIN_BASELINE_WAFERS}")
print(f"최대 탐색 (일)    : {MAX_LOOKBACK_DAYS}")
print(f"쿼리 단위 (일)    : {CHUNK_DAYS}")
print(f"출력 경로         : {args.output}")
print()

# ──────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────

def decode_hex_to_float32(hex_str) -> np.ndarray:
    if not hex_str:
        return np.array([])
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError:
        return np.array([])
    n = len(raw) // 4
    return np.array(
        [struct.unpack('<f', raw[i*4:(i+1)*4])[0] for i in range(n)],
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
    """DataFrame → {GroupKey: [(first_ts, WaferData), ...]} 시간 오름차순."""
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


def query_window(start: str, end: str) -> pd.DataFrame:
    q = f"""
    SELECT eqp_id, product, module_id, wafer_id,
           x, y, temp_raw, event_tmstp
    FROM ds_catalog.aifpa_cow_rule_hist
    WHERE event_tmstp BETWEEN '{start}' AND '{end}'
    ORDER BY eqp_id, product, module_id, event_tmstp
    """
    try:
        return client.query_dataframe(q)
    except Exception as e:
        print(f"  [쿼리 오류] {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# 1. 역사 데이터: 주 단위로 역순 쿼리, 그룹별 5장 채워지면 중단
# ──────────────────────────────────────────────
# hist_wafers[gk] = 시간 오름차순 (oldest first)의 (ts, WaferData) 리스트
hist_wafers: dict[GroupKey, list[tuple]] = {}
satisfied: set[GroupKey] = set()  # 이미 5장 이상 확보된 그룹

chunk_end = today - timedelta(days=1)  # 어제부터 시작

for _ in range((MAX_LOOKBACK_DAYS + CHUNK_DAYS - 1) // CHUNK_DAYS):
    chunk_start = chunk_end - timedelta(days=CHUNK_DAYS - 1)
    # 최대 탐색 한계 초과 방지
    earliest_allowed = today - timedelta(days=MAX_LOOKBACK_DAYS)
    if chunk_start < earliest_allowed:
        chunk_start = earliest_allowed

    s = f"{chunk_start} 00:00:00"
    e = f"{chunk_end} 23:59:59"
    print(f"  [{s} ~ {e}] 쿼리 중...", end=" ", flush=True)

    df_chunk = query_window(s, e)
    chunk_wafers = df_to_wafers(df_chunk)
    del df_chunk

    # 이번 청크 결과를 hist_wafers에 prepend (이번 청크가 더 오래됨)
    new_groups = 0
    for gk, ts_list in chunk_wafers.items():
        if gk not in hist_wafers:
            hist_wafers[gk] = []
            new_groups += 1
        hist_wafers[gk] = ts_list + hist_wafers[gk]  # prepend (더 오래된 것 앞에)

    # 이번 라운드 후 만족 여부 갱신
    for gk, lst in hist_wafers.items():
        if len(lst) >= MIN_BASELINE_WAFERS:
            satisfied.add(gk)

    total_groups = len(hist_wafers)
    total_wafers = sum(len(v) for v in hist_wafers.values())
    print(f"groups={total_groups}, wafers={total_wafers}, satisfied={len(satisfied)}/{total_groups}")

    # 모든 알려진 그룹이 만족됐으면 조기 종료
    if satisfied and satisfied == set(hist_wafers):
        print("  → 모든 그룹 충족, 탐색 종료")
        break

    chunk_end = chunk_start - timedelta(days=1)
    if chunk_end < earliest_allowed:
        break

print()

# ──────────────────────────────────────────────
# 2. 오늘 데이터 쿼리 (역사 부족 그룹 보완 + 신규 그룹 발견)
# ──────────────────────────────────────────────
print(f"오늘 데이터 쿼리 중 ({TODAY_START} ~ {TODAY_END})...", end=" ", flush=True)
df_today = query_window(TODAY_START, TODAY_END)
today_wafers = df_to_wafers(df_today)
del df_today
print(f"groups={len(today_wafers)}")
print()

# ──────────────────────────────────────────────
# 3. 파이프라인 baseline 구축
# ──────────────────────────────────────────────
pipeline = TempVibrationPipeline(config=DEFAULT_CONFIG)

all_group_keys = set(hist_wafers) | set(today_wafers)
print(f"baseline 구축 (총 {len(all_group_keys)} groups)")

for gk in sorted(all_group_keys):
    hist_list  = [w for _, w in hist_wafers.get(gk, [])]
    today_list = [w for _, w in today_wafers.get(gk, [])]
    n_hist, n_today = len(hist_list), len(today_list)

    if n_hist >= MIN_BASELINE_WAFERS:
        # 역사 충분 → reference + inter-wafer 완전 baseline
        # inter-wafer.baseline_wafers(기본 10)보다 많으면 inter도 동결됨
        pipeline.initialize_baseline(gk, hist_list)
        st = pipeline._inter_tracker._states.get(gk)
        frozen = st.baseline_frozen if st else False
        print(f"  {gk}  hist={n_hist}  → full (inter frozen={frozen})")

    elif n_hist > 0:
        # 역사 부족 → 역사분으로 reference + inter 부분 구축,
        # 오늘 데이터는 reference 보완만 (inter-wafer 제외)
        pipeline.initialize_baseline(gk, hist_list)
        for wafer in today_list:
            pipeline._ref_manager.update(wafer)
        st = pipeline._inter_tracker._states.get(gk)
        frozen = st.baseline_frozen if st else False
        print(f"  {gk}  hist={n_hist}(<{MIN_BASELINE_WAFERS}) today={n_today}"
              f"  → ref 보완 (inter frozen={frozen})")

    else:
        # 역사 없음 → 오늘 데이터로 reference만
        if today_list:
            for wafer in today_list:
                pipeline._ref_manager.update(wafer)
            print(f"  {gk}  hist=0 today={n_today}  → reference only (intra만)")
        else:
            print(f"  {gk}  데이터 없음 → SKIP")

print()

# ──────────────────────────────────────────────
# 4. JSON 저장
# ──────────────────────────────────────────────
pipeline.save_baseline(args.output)

ref_groups    = len(pipeline._ref_manager._groups)
inter_frozen  = sum(1 for st in pipeline._inter_tracker._states.values() if st.baseline_frozen)
print(f"저장 완료: {args.output}")
print(f"  reference 구축  : {ref_groups} groups")
print(f"  inter 완전 동결 : {inter_frozen} groups")
print(f"  inter 미동결    : {ref_groups - inter_frozen} groups (intra만 즉시 작동)")
