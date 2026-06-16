"""
Run Temperature Vibration Detection using temp_vibration_detection package

This script:
1. Queries act_temp data from ClickHouse
2. Uses the temp_vibration_detection package to detect vibration anomalies
3. Outputs results to CSV and generates plots
"""

import sys
sys.path.append('/workspace')

from clickhouse_driver import Client
import pandas as pd
import numpy as np
import os
import struct
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings(action='ignore')

from temp_vibration_detection.pipeline import TempVibrationPipeline
from temp_vibration_detection.data_models import WaferData, ChipData, GroupKey
from temp_vibration_detection.config import DEFAULT_CONFIG

# ClickHouse connection
client = Client(host=host, user=user, password=password)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

current_time = datetime.now()
end_time = current_time - timedelta(hours=2)
start_time = end_time - timedelta(hours=12)
START_DATE = start_time.strftime("%Y-%m-%d %H:%M:%S")
END_DATE = end_time.strftime("%Y-%m-%d %H:%M:%S")

BASE_OUTPUT_DIR = "/workspace/act_resi/sw_act_resi_anomaly/output"
output_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{RUN_TIMESTAMP}")
os.makedirs(output_dir, exist_ok=True)
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
eqp_csv_dir = os.path.join(output_dir, "eqp_results")
os.makedirs(eqp_csv_dir, exist_ok=True)

print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Output directory: {output_dir}")
print()

# -------------------------------------------------
# COMMAND LINE ARGUMENTS
# -------------------------------------------------

parser = argparse.ArgumentParser(description='Temperature Vibration Detection')

# [수정 1] --baseline-load / --baseline-save 추가
# 첫 실행: --baseline-start/end 로 구축 후 --baseline-save 로 저장
# 이후 실행: --baseline-load 로 바로 복원 (재쿼리 불필요)
parser.add_argument('--baseline-load', type=str,
                    help='기구축된 baseline JSON 파일 경로 (load하면 baseline-start/end 무시)')
parser.add_argument('--baseline-save', type=str,
                    help='baseline 구축 후 저장할 JSON 파일 경로')
parser.add_argument('--baseline-start', type=str, help='Baseline start (YYYY-MM-DD HH:MM:SS)')
parser.add_argument('--baseline-end', type=str, help='Baseline end (YYYY-MM-DD HH:MM:SS)')
parser.add_argument('--baseline-eqp', type=str, help='Baseline equipment ID (optional filter)')
parser.add_argument('--baseline-recipe', type=str, help='Baseline recipe (optional filter)')
parser.add_argument('--baseline-head', type=str, help='Baseline head/module (optional filter)')
args = parser.parse_args()

# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------

def decode_hex_to_float32(hex_str):
    if hex_str is None or hex_str == "":
        return np.array([])
    try:
        raw_bytes = bytes.fromhex(hex_str)
    except ValueError:
        return np.array([])
    n_floats = len(raw_bytes) // 4
    return np.array(
        [struct.unpack('<f', raw_bytes[i*4:(i+1)*4])[0] for i in range(n_floats)],
        dtype=np.float32,
    )


def create_chip_data_from_row(row, bond_order):
    temp_series = decode_hex_to_float32(row.get('temp_raw', ''))
    return ChipData(
        equipment_id=row.get('eqp_id', 'unknown'),
        recipe=row.get('product', 'unknown'),
        head=row.get('module_id', row.get('head', 'unknown')),
        wafer_id=row.get('wafer_id', 'unknown'),
        x=int(row.get('x', 0)),
        y=int(row.get('y', 0)),
        bond_order=bond_order,
        temp_series=temp_series,
    )


def plot_vibration_anomalies(wafer_data, anomalies, eqp_id, wafer_id, product, module_id, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    chips_sorted = wafer_data.chips_sorted_by_bond_order()

    ax1 = axes[0, 0]
    for i, chip in enumerate(chips_sorted):
        if len(chip.temp_series) > 0:
            ax1.plot(chip.temp_series, alpha=0.5, linewidth=1)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature")
    ax1.set_title(f"Temperature Profiles\n{eqp_id} | {wafer_id} | {product} | {module_id}")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    temp_roughness = [
        np.std(np.diff(chip.temp_series)) if len(chip.temp_series) > 1 else 0
        for chip in chips_sorted
    ]
    ax2.plot(temp_roughness, 'o-', markersize=4)
    ax2.set_xlabel("Bond Order")
    ax2.set_ylabel("Temperature Roughness")
    ax2.set_title("Temperature Roughness Sequence")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if anomalies:
        for i, anomaly in enumerate(anomalies[:5]):
            color = "red" if anomaly.severity > 2 else "yellow"
            ax3.text(0.05, 0.9 - i * 0.18,
                     f"{anomaly.anomaly_type}: {anomaly.severity:.3f}",
                     transform=ax3.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    else:
        ax3.text(0.5, 0.5, "No anomalies detected",
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    ax3.axis('off')
    ax3.set_title("Detected Anomalies")

    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = (
        f"Statistics Summary\n"
        f"==================\n"
        f"Equipment: {eqp_id}\n"
        f"Wafer: {wafer_id}\n"
        f"Product: {product}\n"
        f"Module: {module_id}\n\n"
        f"Total Chips: {len(chips_sorted)}\n"
        f"Avg Roughness: {np.mean(temp_roughness):.4f}\n"
        f"Max Roughness: {np.max(temp_roughness):.4f}\n\n"
        f"Anomalies: {len(anomalies)}\n"
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()
    filename = f"{eqp_id}_{wafer_id}_{product}_{module_id}_vibration_analysis.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    return filepath


# [수정 2] query_baseline_wafers: (eqp_id, product, module_id, wafer_id) 기준으로 그룹핑,
#          칩은 event_tmstp 순 정렬, wafer 목록도 시간 순 정렬
def query_baseline_wafers(
    client: Client,
    start_date: str,
    end_date: str,
    eqp_id: str | None = None,
    recipe: str | None = None,
    head: str | None = None,
) -> dict[GroupKey, list[WaferData]]:
    query = f"""
    SELECT
        eqp_id,
        product,
        module_id,
        wafer_id,
        x,
        y,
        temp_raw,
        event_tmstp
    FROM ds_catalog.aifpa_cow_rule_hist
    WHERE event_tmstp BETWEEN '{start_date}' AND '{end_date}'
    """
    if eqp_id:
        query += f" AND eqp_id = '{eqp_id}'"
    if recipe:
        query += f" AND product = '{recipe}'"
    if head:
        query += f" AND module_id = '{head}'"
    query += " ORDER BY eqp_id, product, module_id, event_tmstp"

    try:
        df = client.query_dataframe(query)
    except Exception as e:
        print(f"Error querying baseline wafers: {e}")
        return {}

    if df.empty or 'temp_raw' not in df.columns:
        return {}

    # 그룹핑 키: eqp_id × product × module_id × wafer_id
    # → 동일한 wafer_id를 가진 다른 장비·레시피가 뒤섞이지 않도록
    group_cols = [c for c in ['eqp_id', 'product', 'module_id', 'wafer_id'] if c in df.columns]

    # 각 (group_key, wafer)별 첫 번째 타임스탬프를 기록해 시간 순 정렬에 사용
    wafers_with_ts: dict[GroupKey, list[tuple]] = {}  # {GroupKey: [(first_ts, WaferData)]}

    for _, wafer_df in df.groupby(group_cols, sort=False):
        wafer_df = wafer_df.sort_values('event_tmstp')

        chips = []
        for _, row in wafer_df.iterrows():
            chip = create_chip_data_from_row(row, len(chips))
            if len(chip.temp_series) > 0:
                chips.append(chip)

        if not chips:
            continue

        wafer_id_val = str(wafer_df['wafer_id'].iloc[0])
        first_ts = wafer_df['event_tmstp'].iloc[0]
        group_key = chips[0].group_key

        wafers_with_ts.setdefault(group_key, []).append(
            (first_ts, WaferData(wafer_id=wafer_id_val, chips=chips))
        )

    # 각 group_key 내에서 wafer를 시간 순으로 정렬
    return {
        gk: [w for _, w in sorted(ts_list, key=lambda x: x[0])]
        for gk, ts_list in wafers_with_ts.items()
    }


# -------------------------------------------------
# MAIN PROCESSING
# -------------------------------------------------

print("Discovering equipment IDs...")
try:
    df_discover = client.query_dataframe(f"""
        SELECT DISTINCT eqp_id
        FROM ds_catalog.aifpa_cow_rule_hist
        WHERE event_tmstp >= '{START_DATE}' AND event_tmstp <= '{END_DATE}'
        ORDER BY eqp_id
    """)
    eqp_ids = df_discover['eqp_id'].tolist()
    print(f"Found {len(eqp_ids)} equipment IDs")
except Exception as e:
    print(f"Error discovering equipment IDs: {e}")
    eqp_ids = []
print()

pipeline = TempVibrationPipeline(config=DEFAULT_CONFIG)

# [수정 3] baseline 초기화 로직:
#   A) --baseline-load 가 있으면 JSON 에서 바로 복원 (재쿼리 불필요)
#   B) --baseline-start/end 가 있으면 쿼리 후 group_key 별 initialize_baseline,
#      그리고 --baseline-save 가 있으면 JSON 으로 저장
if args.baseline_load:
    print(f"Loading baseline from {args.baseline_load} ...")
    pipeline.load_baseline(args.baseline_load)
    print("Baseline loaded\n")

elif args.baseline_start and args.baseline_end:
    print(f"Building baseline from {args.baseline_start} to {args.baseline_end}")
    baseline_wafers = query_baseline_wafers(
        client=client,
        start_date=args.baseline_start,
        end_date=args.baseline_end,
        eqp_id=args.baseline_eqp,
        recipe=args.baseline_recipe,
        head=args.baseline_head,
    )

    if baseline_wafers:
        for group_key, wafers in baseline_wafers.items():
            print(f"  {group_key}: {len(wafers)} wafers")
            pipeline.initialize_baseline(group_key, wafers)
        print(f"Baseline built for {len(baseline_wafers)} groups\n")
    else:
        print("  No baseline wafers found\n")

    if args.baseline_save:
        pipeline.save_baseline(args.baseline_save)
        print(f"Baseline saved to {args.baseline_save}\n")

# -------------------------------------------------
# PRODUCTION WAFER PROCESSING
# -------------------------------------------------

results = []

for eqp_id in eqp_ids:
    print(f"Processing {eqp_id} ...")

    query = f"""
    SELECT *
    FROM ds_catalog.aifpa_cow_rule_hist
    WHERE event_tmstp >= '{START_DATE}'
      AND event_tmstp <= '{END_DATE}'
      AND eqp_id = '{eqp_id}'
    ORDER BY event_tmstp
    """

    try:
        df = client.query_dataframe(query)

        if len(df) == 0:
            print(f"  No data - SKIPPING")
            results.append({'eqp_id': eqp_id, 'status': 'no_data',
                            'total_rows': 0, 'total_wafers': 0, 'total_chips': 0,
                            'vibration_anomalies': 0, 'normal_wafers': 0})
            continue

        if 'temp_raw' not in df.columns:
            print(f"  temp_raw column missing - SKIPPING")
            results.append({'eqp_id': eqp_id, 'status': 'no_temp_column',
                            'total_rows': len(df), 'total_wafers': 0, 'total_chips': 0,
                            'vibration_anomalies': 0, 'normal_wafers': 0})
            continue

        df['temp_decoded'] = df['temp_raw'].apply(decode_hex_to_float32)
        df = df[df['temp_decoded'].apply(lambda x: len(x) > 0)]

        if len(df) == 0:
            results.append({'eqp_id': eqp_id, 'status': 'no_valid_temp_data',
                            'total_rows': 0, 'total_wafers': 0, 'total_chips': 0,
                            'vibration_anomalies': 0, 'normal_wafers': 0})
            continue

        group_cols = [c for c in ['eqp_id', 'product', 'module_id', 'wafer_id'] if c in df.columns]

        eqp_anomalies = []
        eqp_total_chips = 0
        eqp_total_wafers = 0
        eqp_normal_wafers = 0

        for key_vals, g in df.groupby(group_cols, sort=False):
            key_dict = dict(zip(group_cols, key_vals))
            product = key_dict.get('product', 'unknown')
            wafer = key_dict.get('wafer_id', 'unknown')
            module_id = key_dict.get('module_id', 'unknown')

            g = g.sort_values('event_tmstp')

            chips = []
            for idx, (_, row) in enumerate(g.iterrows()):
                chip = create_chip_data_from_row(row, bond_order=idx)
                if len(chip.temp_series) > 0:
                    chips.append(chip)

            if not chips:
                continue

            eqp_total_chips += len(chips)
            eqp_total_wafers += 1

            wafer_data = WaferData(wafer_id=wafer, chips=chips)
            anomalies = pipeline.process_wafer(wafer_data, update_reference=True)

            if anomalies:
                eqp_anomalies.extend(anomalies)
                print(f"  Wafer {wafer}: {len(anomalies)} anomalies")
            else:
                eqp_normal_wafers += 1

            plot_vibration_anomalies(wafer_data, anomalies, eqp_id, wafer, product, module_id, plots_dir)

        results.append({
            'eqp_id': eqp_id, 'status': 'success',
            'total_rows': len(df),
            'total_wafers': eqp_total_wafers,
            'total_chips': eqp_total_chips,
            'vibration_anomalies': len(eqp_anomalies),
            'normal_wafers': eqp_normal_wafers,
        })

        if eqp_anomalies:
            anomaly_df = pd.DataFrame([{
                'eqp_id': a.group_key.equipment_id,
                'recipe': a.group_key.recipe,
                'head': a.group_key.head,
                'wafer_id': a.wafer_id,
                'anomaly_type': a.anomaly_type,
                'severity': a.severity,
                'details': str(a.details),
            } for a in eqp_anomalies])
            csv_path = os.path.join(eqp_csv_dir, f"{eqp_id}_vibration_anomalies_{RUN_TIMESTAMP}.csv")
            anomaly_df.to_csv(csv_path, index=False)
            print(f"  Saved anomalies: {csv_path}")

        print(f"  wafers={eqp_total_wafers}, chips={eqp_total_chips}, anomalies={len(eqp_anomalies)}")
        print()

    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        results.append({'eqp_id': eqp_id, 'status': 'error', 'error': str(e),
                        'total_rows': 0, 'total_wafers': 0, 'total_chips': 0,
                        'vibration_anomalies': 0, 'normal_wafers': 0})

summary_df = pd.DataFrame(results)
summary_path = os.path.join(output_dir, f"vibration_detection_summary_{RUN_TIMESTAMP}.csv")
summary_df.to_csv(summary_path, index=False)

print("=" * 60)
print("DONE")
print(f"Equipment processed : {len(eqp_ids)}")
print(f"Total wafers        : {summary_df['total_wafers'].sum()}")
print(f"Total chips         : {summary_df['total_chips'].sum()}")
print(f"Vibration anomalies : {summary_df['vibration_anomalies'].sum()}")
print(f"Output              : {output_dir}")
print("=" * 60)
