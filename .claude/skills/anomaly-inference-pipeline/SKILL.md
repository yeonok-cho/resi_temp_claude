---
name: anomaly-inference-pipeline
description: How wafers actually get scored end to end in this repo — AnomalyDetectionPipeline.process_wafer()/TempVibrationPipeline.process_wafer(), the AnomalyEvent output schema, baseline persistence (save_baseline/load_baseline), PM-event baseline resets, and the production ClickHouse + Airflow DAG deployment (watermark-based incremental runs). Use this whenever the user wants to run the pipeline on a wafer or wafer sequence, wire preprocessing and detectors together, debug why a wafer produced no/unexpected AnomalyEvents, reproduce an end-to-end detection run on synthetic data, or understand/extend the production scripts under temp_anomaly/scripts/ and temp_anomaly/dags/. Load anomaly-data-preprocessing and anomaly-detection-models first if you need details on what happens inside a given step.
---

# Running the pipeline (inference)

"Inference" here means: given a new `WaferData` (chips just finished bonding), produce
the list of `AnomalyEvent`s for it, using whatever reference/detector state has
accumulated from wafers seen so far. This is a **stateful, sequential, streaming**
process, not a batch train/predict split — every `process_wafer()` call both scores the
new wafer *and* updates the state that future calls depend on.

## 1. Core orchestration — `hbm_anomaly_detection/pipeline.py`

`AnomalyDetectionPipeline.__init__` (line 45) constructs one instance of every stateful
component from `anomaly-detection-models`, all sharing the same `AnomalyConfig`:
`ReferenceManager`, `InterWaferTracker`, `SpikeRateTracker`, `CouplingDetector`,
`MultivariateInterWaferTracker`, `AutoencoderAnomalyDetector`. **One pipeline instance
can serve every `GroupKey`** — each component internally keys its state by `GroupKey`,
so you don't need one pipeline per equipment/recipe/head.

`process_wafer(wafer, update_reference=True)` (line 54), step by step:

1. **Cold-start check** (line 79-86): `ref = self._ref_manager.get_or_create(key)`; if
   `not ref.is_initialized()`, this is the group's first wafer ever — build the
   reference (if `update_reference`) and **return `[]` with no detection**. This avoids
   false alarms from a reference/roughness-baseline that doesn't exist yet. Concretely:
   *the first wafer of any new group never produces events, by design.*
2. **Feature extraction** (line 88-101): for each chip, `ReferenceManager.
   get_references(chip, ref)` picks the right (possibly bimodal-cluster-specific)
   reference, then `extract_chip_features` and `extract_residual_vector` build the
   `ChipFeatures` and the autoencoder's residual vector respectively (see
   `anomaly-data-preprocessing`).
3. **Run every detector** (line 111-165), collecting `AnomalyEvent`s into one flat list:
   `intra_wafer` (both channels) → `spike_detector` (per-chip + rate) → `inter_wafer`
   (both channels) → `multivariate_inter_wafer` (joint) → `coupling_detector` →
   `autoencoder`. Order doesn't affect correctness (detectors are independent), but
   matches the module's own step numbering (3a-3e) if you're reading the source.
4. **Update reference + roughness baseline** (line 167-179) — **after** detection, so
   the wafer that triggered an alarm doesn't also get baked into the baseline that
   should have flagged it. Only happens if `update_reference=True`.

`process_wafer_sequence(wafers, n_warmup=5)` (line 183) is a thin loop calling
`process_wafer` for each wafer in order and collecting results into
`dict[wafer_id, list[AnomalyEvent]]` — note `n_warmup` is accepted but not actually
used to suppress anything in the current implementation; warmup behavior is entirely
driven by each detector's own `baseline_wafers`/`warmup_chips` config.

## 2. Output schema — `AnomalyEvent` (`data_models.py:100`)

```python
AnomalyEvent(
    group_key: GroupKey,      # which equipment/recipe/head
    wafer_id: str,
    anomaly_type: str,        # "intra_drift" | "inter_drift" | "inter_drift_mv"
                               # | "spike" | "coupling" | "deep_anomaly"
    channel: str,              # "temp" | "resist" | "both"
    severity: float,           # detector-specific statistic — NOT comparable across
                               # anomaly_types (a CUSUM peak and a T² value are on
                               # different scales; see anomaly-detection-models)
    details: dict,             # detector-specific — e.g. chip_x/chip_y/bond_order for
                               # per-chip events, threshold/baseline values for context
)
```
`str(event)` gives a one-line human-readable summary (line 109). Every detector's
`details` dict is documented per-detector in `anomaly-detection-models`.

## 3. Reproducing a full run on synthetic data

This is the pattern used throughout `tests/test_pipeline.py` — the canonical
reproducibility reference. Minimal end-to-end example combining all three skills:

```python
from hbm_anomaly_detection.pipeline import AnomalyDetectionPipeline
from hbm_anomaly_detection.simulator import (
    make_normal_wafer_sequence, make_intra_vibration_wafer,
)
import numpy as np

pipeline = AnomalyDetectionPipeline()  # DEFAULT_CONFIG

# 1. Warm up: baseline wafers, no alarms expected (reference + detector baselines build)
baseline = make_normal_wafer_sequence(n_wafers=12, n_chips=100, n_points=500, seed=10)
for w in baseline:
    pipeline.process_wafer(w)

# 2. Feed an anomalous wafer and inspect events
anomaly_wafer = make_intra_vibration_wafer(
    wafer_id="INTRA_ANOM", n_chips=100, n_points=500,
    onset_fraction=0.5, max_noise_multiplier=6.0,
    rng=np.random.default_rng(99),
)
events = pipeline.process_wafer(anomaly_wafer)
assert any(e.anomaly_type == "intra_drift" for e in events)
```

To validate a change reproduces the expected end-to-end behavior, run the real test
suite rather than hand-checking:
`python -m pytest hbm_anomaly_detection/tests/test_pipeline.py -v`
— it covers false-positive rate on clean data, detection of every injected anomaly
type, and autoencoder fit/retrain lifecycle (11 test classes, ~40 tests).

## 4. Production variant — `temp_anomaly/temp_vibration_detection/pipeline.py`

`TempVibrationPipeline` is the temperature-only subset actually deployed against real
data (no resistance channel, no coupling/autoencoder detectors — just intra + inter
drift on temperature roughness). It adds operational features `AnomalyDetectionPipeline`
doesn't have:

- **`reset_baseline(group_key)`** (line 45) + **`pm_event=True`** param on
  `process_wafer` (line 66-92): after a Preventive Maintenance event, the equipment's
  physical baseline can shift, so the reference profile and inter-wafer EWMA are wiped
  and rebuilt from scratch — but intra-wafer detection (self-calibrating per wafer, see
  `anomaly-detection-models` §1) is untouched, since it never depended on cross-wafer
  state anyway.
- **`initialize_baseline(group_key, wafers)`** (line 135): pre-warm a group's state
  from a batch of known-good historical wafers in one call, discarding any events —
  used to bootstrap a group before it starts seeing live production wafers.
- **`save_baseline(path)` / `load_baseline(path)`** (lines 159, 225): serialize/restore
  every group's reference profile + inter-wafer EWMA state to/from JSON, so a
  production process doesn't have to re-derive baselines from scratch on every restart.
  The JSON also embeds the config used to build it (line 213) for auditability.

## 5. Production deployment — ClickHouse ingestion + Airflow

Three scripts under `temp_anomaly/scripts/` (all import `_common.py` — see
`anomaly-data-preprocessing` §2b — for ClickHouse decode/grouping helpers):

- **`build_baseline.py`** — one-shot CLI to construct a `baseline.json` from historical
  data before going live. Strategy (module docstring, lines 4-14): scan backwards
  week-by-week per `GroupKey` until `min-wafers` (default 5) history is found or
  `max-lookback-days` (default 30) is exhausted, then supplement with today's data.
  Groups with insufficient history still get a temperature-only reference (intra-wafer
  detection works immediately; inter-wafer baseline fills in as production data
  arrives). Run: `python scripts/build_baseline.py --output baseline.json --min-wafers 5`.
- **`run_anomaly_detection.py`** — the actual production entry point, designed to be
  **idempotent and incrementally re-runnable** (module docstring, lines 1-37):
  1. Load `state/baseline.json` (pipeline state) and `state/watermark.json` (last
     processed `event_tmstp`), if present.
  2. If no watermark (first run ever): bootstrap by scanning backwards in
     `bootstrap-chunk-days` chunks (default 7) up to `bootstrap-lookback-days` (default
     30), stopping early once every discovered group has `min-wafers` history.
  3. Otherwise: query only `event_tmstp > watermark` — **already-processed data is
     never re-scanned**, so steady-state runs cost only the last interval's data.
  4. Process each group's new wafers in order; if the group's wafer count is still
     below `min-wafers` at the time a wafer is processed, that wafer's events are
     discarded (status `"building"`) — only `"ready"` groups' events are kept. A group
     flips from building→ready mid-run automatically, no separate promotion step.
  5. Persist updated `baseline.json` + `watermark.json`, write `anomalies.csv` and
     `group_summary.csv` for the run.
  Run: `python scripts/run_anomaly_detection.py --state-dir /workspace/state
  --output-dir /workspace/output --min-wafers 5`.
- **`temp_anomaly/dags/temp_vibration_dag.py`** — a 46-line Airflow DAG that just shells
  out to `run_anomaly_detection.py` every 12 hours (`schedule_interval=timedelta(hours=12)`),
  relying entirely on the script's own idempotency for safe retries
  (`retries=2`, `max_active_runs=1`).

If you're extending the `hbm_anomaly_detection` (temp+resistance) pipeline for
production use rather than `temp_anomaly`, this is the pattern to replicate — there is
currently no ClickHouse ingestion script for the resistance channel, only for
temperature.

## Reproducibility checklist

- A pipeline instance's behavior depends entirely on prior `process_wafer` calls (it's
  stateful) — when reproducing a bug or a specific detection, always replay wafers in
  the **same order** from a fresh `AnomalyDetectionPipeline()`/`TempVibrationPipeline()`,
  don't assume you can jump straight to the interesting wafer.
- Expect **zero** events on any group's first wafer ever (cold start) and generally
  fewer/no events during each detector's own baseline period — see
  `anomaly-detection-models` for each detector's specific warm-up length.
- For production-state reproduction, `save_baseline`/`load_baseline` round-trip the
  full pipeline state as JSON — prefer loading a saved baseline over re-deriving one
  when debugging a specific production incident.
- End-to-end validation: `python -m pytest hbm_anomaly_detection/tests/test_pipeline.py -v`
  and `python -m pytest temp_anomaly/temp_vibration_detection/tests/test_pipeline.py -v`.
