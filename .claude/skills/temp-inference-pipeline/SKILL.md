---
name: temp-inference-pipeline
description: How temperature-only wafers get scored end to end and deployed to production in this repo — TempVibrationPipeline.process_wafer() in temp_anomaly/temp_vibration_detection/pipeline.py, PM-event baseline resets, baseline persistence (save_baseline/load_baseline JSON), and the ClickHouse + Airflow DAG production deployment (watermark-based incremental runs) under temp_anomaly/scripts/ and temp_anomaly/dags/. Use this whenever the user wants to run temperature vibration detection on a wafer or sequence, debug why a wafer produced no/unexpected temp AnomalyEvents, reproduce an end-to-end run on synthetic data, or extend the production ingestion/DAG scripts. Load temp-data-preprocessing and temp-detection-models first for what happens inside each step. For the joint temp+resist pipeline, use resi-inference-pipeline instead.
---

# Running the temperature-only pipeline (inference)

`TempVibrationPipeline` (`temp_anomaly/temp_vibration_detection/pipeline.py`) is the
package that is **actually deployed to production** in this repo — it's the only
pipeline with real ClickHouse ingestion scripts and an Airflow DAG. It's a stateful,
sequential streamer: each `process_wafer()` call both scores the new wafer and updates
the state future calls depend on, same contract as the joint
`AnomalyDetectionPipeline` (see `resi-inference-pipeline`) but temperature-only.

## 1. Core orchestration — `pipeline.py`

`TempVibrationPipeline.__init__` (line 40) builds one `ReferenceManager` and one
`InterWaferTracker`, sharing an `AnomalyConfig`. One instance serves every `GroupKey`.

`process_wafer(wafer, update_reference=True, pm_event=False)` (line 62):

1. `if pm_event:` call `self.reset_baseline(key)` (line 45) **first** — wipes both the
   reference profile (`ReferenceManager.reset`) and the inter-wafer EWMA
   (`InterWaferTracker.reset`) for this group, since a Preventive Maintenance event can
   shift the equipment's physical baseline (see `temp-data-preprocessing` §3 and
   `temp-detection-models` §2). Intra-wafer detection is untouched — it never depended
   on cross-wafer state to begin with.
2. **Cold-start check**: `if not ref.is_initialized()`, build the reference only
   (if `update_reference`) and return `[]` — same rule as the joint pipeline: *a
   group's (or a post-PM group's) first wafer never produces events.*
3. **Feature extraction**: `extract_chip_features(chip, ref.temp_ref)` per chip →
   `WaferFeatures`.
4. **Run both detectors**: `detect_intra_wafer_drift` then
   `InterWaferTracker.update_and_detect` on `summary["mean"]` — see
   `temp-detection-models` for the algorithms.
5. **Update the reference** (`update_reference=True`) — after detection, so the
   triggering wafer doesn't get folded into the baseline that should have flagged it.

`process_wafer_sequence(wafers)` (line 267) loops `process_wafer` over a list and
returns `dict[wafer_id, list[AnomalyEvent]]`.

## 2. Lifecycle helpers not present in the joint pipeline

- **`initialize_baseline(group_key, wafers)`** (line 135) — resets the group, then
  replays a list of known-good historical wafers through `process_wafer`, discarding
  any events. Use this to bootstrap a group from history before it starts seeing live
  wafers (should contain ≥ `inter_wafer.baseline_wafers` wafers to fully freeze the
  inter-wafer baseline; fewer still works, the baseline just keeps accumulating from
  live data).
- **`save_baseline(path)`** (line 159) — serializes every group's `temp_ref`,
  `n_wafers_seen`, and `InterWaferState` fields to JSON, plus the config used to build
  it (for auditability). **Note the JSON round-trip is lossy on purpose**: it persists
  `temp_ref`/`n_wafers_seen`/`n_points` for the reference and the five `InterWaferState`
  scalar fields — it does **not** persist intra-wafer state (there is none to persist;
  intra-wafer is self-calibrating per wafer, see `temp-detection-models` §1).
- **`load_baseline(path)`** (line 225) — restores state from that JSON. Existing
  in-memory groups not present in the file are left untouched (additive load, not a
  full reset).

```python
from temp_vibration_detection.pipeline import TempVibrationPipeline
from temp_vibration_detection.simulator import make_normal_wafer_sequence

pipeline = TempVibrationPipeline()
wafers = make_normal_wafer_sequence(n_wafers=12, n_chips=50, n_points=300, seed=1)
pipeline.initialize_baseline(("EQ01", "R001", "H1"), wafers)
pipeline.save_baseline("/tmp/temp_baseline.json")

pipeline2 = TempVibrationPipeline()
pipeline2.load_baseline("/tmp/temp_baseline.json")
```

## 3. Reproducing a full run on synthetic data

```python
from temp_vibration_detection.pipeline import TempVibrationPipeline
from temp_vibration_detection.simulator import (
    make_normal_wafer_sequence, make_intra_vibration_wafer,
)
import numpy as np

pipeline = TempVibrationPipeline()
baseline = make_normal_wafer_sequence(n_wafers=12, n_chips=100, n_points=500, seed=10)
for w in baseline:
    pipeline.process_wafer(w)

anomaly_wafer = make_intra_vibration_wafer(
    wafer_id="INTRA_ANOM", n_chips=100, n_points=500,
    onset_fraction=0.5, max_noise_multiplier=6.0,
    rng=np.random.default_rng(99),
)
events = pipeline.process_wafer(anomaly_wafer)
assert any(e.anomaly_type == "intra_drift" for e in events)
```

Validate end to end: `python -m pytest
temp_anomaly/temp_vibration_detection/tests/test_pipeline.py -v`.

## 4. Production deployment — ClickHouse + Airflow

Three scripts under `temp_anomaly/scripts/` (share `_common.py` ingestion helpers, see
`temp-data-preprocessing` §2b):

- **`build_baseline.py`** — one-shot CLI: `python scripts/build_baseline.py --output
  baseline.json --min-wafers 5`. Scans backwards week-by-week per `GroupKey` (module
  docstring, lines 4-14) until `min-wafers` history is found or `max-lookback-days`
  (default 30) is exhausted, then supplements with today's data; groups with
  insufficient history still get a reference (intra-wafer detection works immediately),
  with the inter-wafer baseline filling in as production data arrives.
- **`run_anomaly_detection.py`** — the real production entry point, **idempotent and
  incrementally re-runnable** (docstring, lines 1-37):
  1. Load `state/baseline.json` + `state/watermark.json` (last processed
     `event_tmstp`), if present.
  2. No watermark (first run): bootstrap backwards in `bootstrap-chunk-days` (default
     7) chunks up to `bootstrap-lookback-days` (default 30), stopping early once every
     discovered group has `min-wafers` history.
  3. Otherwise: query only `event_tmstp > watermark` — already-processed data is never
     re-scanned.
  4. Per group: wafers processed while `ref.n_wafers_seen < min-wafers` are still fed
     through `process_wafer` (state accumulates) but their events are **discarded**
     (status `"building"`); once past `min-wafers` the group is `"ready"` and its
     events are kept. The flip happens automatically mid-run.
  5. Persist `baseline.json` + `watermark.json`; write `anomalies.csv` +
     `group_summary.csv` for this run.
  Run: `python scripts/run_anomaly_detection.py --state-dir /workspace/state
  --output-dir /workspace/output --min-wafers 5`.
- **`temp_anomaly/dags/temp_vibration_dag.py`** — 46-line Airflow DAG that shells out
  to `run_anomaly_detection.py` every 12h (`schedule_interval=timedelta(hours=12)`),
  relying entirely on that script's idempotency for safe retries (`retries=2`,
  `max_active_runs=1`).

There is currently **no equivalent ClickHouse/Airflow deployment for the joint
temp+resist pipeline** — if you need to productionize `AnomalyDetectionPipeline`
(resistance included), this script trio is the pattern to replicate; see
`resi-inference-pipeline` for what would need to change (decode a `resist_raw` column
too, swap `TempVibrationPipeline` for `AnomalyDetectionPipeline`, call `pm_event`-style
resets via the joint `ReferenceManager`/trackers directly since the joint pipeline
doesn't currently expose a `reset_baseline` convenience method).

## Reproducibility checklist

- The pipeline is stateful — to reproduce a specific wafer's result, replay all prior
  wafers for that group **in the same order** from a fresh `TempVibrationPipeline()`,
  don't jump straight to the wafer of interest.
- Expect zero events on a group's first wafer ever, and on the first wafer after any
  `pm_event=True` call.
- Prefer `save_baseline`/`load_baseline` round-trips over re-deriving a baseline from
  scratch when reproducing a specific production incident.
- End-to-end validation: `python -m pytest
  temp_anomaly/temp_vibration_detection/tests/test_pipeline.py -v`.
