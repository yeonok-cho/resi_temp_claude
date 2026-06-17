"""
Airflow DAG: run incremental temperature-vibration anomaly detection every 12h.

Just calls scripts/run_anomaly_detection.py, which is itself idempotent
(watermark-based) — safe to retry after a failed run without reprocessing
or double-counting data. See that script's module docstring for details.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/workspace/temp_anomaly"
STATE_DIR = f"{PROJECT_ROOT}/state"
OUTPUT_DIR = "/workspace/act_resi/sw_act_resi_anomaly/output"
MIN_WAFERS = 5

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="temp_vibration_anomaly_detection",
    description="Incremental temp-vibration anomaly detection (watermark-based, safe to retry)",
    default_args=default_args,
    schedule_interval=timedelta(hours=12),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["anomaly-detection", "temp-vibration"],
) as dag:

    run_detection = BashOperator(
        task_id="run_anomaly_detection",
        bash_command=(
            f"python {PROJECT_ROOT}/scripts/run_anomaly_detection.py "
            f"--state-dir {STATE_DIR} "
            f"--output-dir {OUTPUT_DIR} "
            f"--min-wafers {MIN_WAFERS}"
        ),
    )
