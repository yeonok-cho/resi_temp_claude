"""
Build per-GroupKey baseline JSON for temp_vibration_detection.

Strategy per GroupKey
---------------------
1. Query wafers from (today-30d) to yesterday (historical window).
2. If >= MIN_BASELINE_WAFERS found  →  full baseline: reference profile + inter-wafer EWMA.
3. If <  MIN_BASELINE_WAFERS found  →  also query today's wafers and use them to
   supplement the **reference profile only** (not the inter-wafer tracker).
   When production wafers arrive, intra-wafer (CUSUM) fires immediately but
   inter-wafer (EWMA) accumulates from production data first.
4. Groups with zero wafers in both windows are skipped (no state written).

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
                    help="그룹당 최소 역사 wafer 수. 미달 시 오늘 데이터로 보완 (default: 5)")
args = parser.parse_args()

MIN_BASELINE_WAFERS: int = args.min_wafers

# ──────────────────────────────────────────────
# 날짜 범위 계산
# ──────────────────────────────────────────────
today = date.today()
yesterday = today - timedelta(days=1)
thirty_days_ago = today - timedelta(days=30)

HIST_START = f"{thirty_days_ago} 00:00:00"
HIST_END   = f"{yesterday} 23:59:59"
TODAY_START = f"{today} 00:00:00"
TODAY_END   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"역사 구간  : {HIST_START} ~ {HIST_END}")
print(f"오늘 구간  : {TODAY_START} ~ {TODAY_END}")
print(f"최소 wafer : {MIN_BASELINE_WAFERS}")
print(f"출력 경로  : {args.output}")
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


def df_to_wafers(df: pd.DataFrame) -> dict[GroupKey, list[tuple[datetime, WaferData]]]:
    """
    DataFrame → {GroupKey: [(first_event_ts, WaferData), ...]} (시간 순 정렬).
    칩은 event_tmstp 오름차순, wafer 목록도 first_ts 오름차순.
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

    # 각 그룹 내 wafer 시간 순 정렬
    return {
        gk: sorted(ts_list, key=lambda x: x[0])
        for gk, ts_list in result.items()
    }


def query_window(start: str, end: str) -> pd.DataFrame:
    """지정 기간의 전체 데이터를 DataFrame으로 반환."""
    q = f"""
    SELECT
        eqp_id, product, module_id, wafer_id,
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
# 1. 역사 데이터 쿼리 및 wafer 변환
# ──────────────────────────────────────────────
print("역사 데이터 쿼리 중...")
df_hist = query_window(HIST_START, HIST_END)
print(f"  → {len(df_hist):,} rows")

hist_wafers = df_to_wafers(df_hist)
del df_hist  # 메모리 해제

print(f"  → {len(hist_wafers)} groups, "
      f"총 {sum(len(v) for v in hist_wafers.values())} wafers\n")

# ──────────────────────────────────────────────
# 2. 오늘 데이터가 필요한 그룹 식별
# ──────────────────────────────────────────────
groups_need_today = {
    gk for gk, ts_list in hist_wafers.items()
    if len(ts_list) < MIN_BASELINE_WAFERS
}

# 역사 데이터가 아예 없는 그룹도 오늘 데이터에서 발견될 수 있으므로
# 오늘 데이터를 모두 쿼리 후 추가 그룹 포함

today_wafers: dict[GroupKey, list[tuple]] = {}
if True:  # 항상 오늘 데이터 쿼리 (새로운 그룹이 있을 수 있음)
    print("오늘 데이터 쿼리 중...")
    df_today = query_window(TODAY_START, TODAY_END)
    print(f"  → {len(df_today):,} rows")
    today_wafers = df_to_wafers(df_today)
    del df_today

    new_groups = set(today_wafers) - set(hist_wafers)
    groups_need_today |= new_groups

    print(f"  → {len(today_wafers)} groups")
    print(f"  역사 부족(<{MIN_BASELINE_WAFERS})으로 오늘 데이터 보완 대상: "
          f"{len(groups_need_today)} groups\n")

# ──────────────────────────────────────────────
# 3. 파이프라인 baseline 구축
# ──────────────────────────────────────────────
pipeline = TempVibrationPipeline(config=DEFAULT_CONFIG)

all_group_keys = set(hist_wafers) | set(today_wafers)
print(f"baseline 구축 시작 (총 {len(all_group_keys)} groups)\n")

for gk in sorted(all_group_keys):
    hist_list = [w for _, w in hist_wafers.get(gk, [])]
    today_list = [w for _, w in today_wafers.get(gk, [])]

    n_hist  = len(hist_list)
    n_today = len(today_list)

    if n_hist >= MIN_BASELINE_WAFERS:
        # ── 역사 wafer 충분: 정상 baseline (reference + inter-wafer)
        pipeline.initialize_baseline(gk, hist_list)
        st = pipeline._inter_tracker._states.get(gk)
        frozen = st.baseline_frozen if st else False
        print(f"  {gk}: hist={n_hist} → full baseline (inter frozen={frozen})")

    elif n_hist > 0:
        # ── 역사 wafer 부족: 역사로 reference + inter-wafer 부분 구축,
        #    오늘 데이터는 reference profile 보완만 (inter-wafer tracker 제외)
        pipeline.initialize_baseline(gk, hist_list)

        # 오늘 wafer → reference 보완 (inter-wafer 건너뜀)
        for wafer in today_list:
            pipeline._ref_manager.update(wafer)

        total = n_hist + n_today
        st = pipeline._inter_tracker._states.get(gk)
        frozen = st.baseline_frozen if st else False
        print(f"  {gk}: hist={n_hist}(<{MIN_BASELINE_WAFERS}), today={n_today} "
              f"→ ref 보완 (inter frozen={frozen}, 부족 시 production에서 누적)")

    else:
        # ── 역사 wafer 없음: 오늘 데이터로 reference만 구축
        if today_list:
            for wafer in today_list:
                pipeline._ref_manager.update(wafer)
            print(f"  {gk}: hist=0, today={n_today} → reference only (intra-wafer 탐지만)")
        else:
            print(f"  {gk}: 데이터 없음 → SKIP")

print()

# ──────────────────────────────────────────────
# 4. JSON 저장
# ──────────────────────────────────────────────
pipeline.save_baseline(args.output)
print(f"Baseline JSON 저장 완료: {args.output}")

# ── 요약 ──
ref_groups = set(pipeline._ref_manager._groups)
inter_frozen = {
    gk for gk, st in pipeline._inter_tracker._states.items()
    if st.baseline_frozen
}
print(f"\n요약")
print(f"  reference 구축  : {len(ref_groups)} groups")
print(f"  inter 완전 동결 : {len(inter_frozen)} groups")
print(f"  inter 미동결    : {len(ref_groups) - len(inter_frozen)} groups (intra-wafer 탐지만)")
