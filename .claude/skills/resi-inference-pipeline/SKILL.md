---
name: resi-inference-pipeline
description: How wafers with a resistance channel get scored end to end in this repo — AnomalyDetectionPipeline.process_wafer() in hbm_anomaly_detection/pipeline.py (the joint temp+resist pipeline; there is no resist-only pipeline class). Use this whenever the user wants to run resistance/bimodal/coupling/spike/deep-anomaly detection on a wafer or sequence, debug why a wafer produced no/unexpected resist AnomalyEvents, reproduce an end-to-end resistance detection run on synthetic data, or figure out what it would take to productionize resistance detection against real ClickHouse data (which currently has no ingestion script). Load resi-data-preprocessing and resi-detection-models first for what happens inside each step. For the temperature-only production pipeline, use temp-inference-pipeline instead.
---

# Running the resistance (joint) pipeline (inference)

There is **no standalone resistance-only pipeline class** in this repo — resistance is
only ever scored as part of `AnomalyDetectionPipeline`
(`hbm_anomaly_detection/pipeline.py`), which processes temperature and resistance
together for every wafer. This skill documents that pipeline from the resistance side;
see `temp-inference-pipeline` for the temperature-only `TempVibrationPipeline`, which
*is* standalone and *is* what's actually deployed to production today.

## 1. Core orchestration — `pipeline.py`

`AnomalyDetectionPipeline.__init__` (line 45) builds one instance each of
`ReferenceManager`, `InterWaferTracker`, `SpikeRateTracker`, `CouplingDetector`,
`MultivariateInterWaferTracker`, `AutoencoderAnomalyDetector` — every resistance-facing
detector from `resi-detection-models` lives here. One pipeline instance serves every
`GroupKey`.

`process_wafer(wafer, update_reference=True)` (line 54), resistance-relevant steps:

1. **Cold-start check** (line 79): first-ever wafer for a group builds the reference
   (including running bimodal detection for the first time — see
   `resi-data-preprocessing` §3) but skips detection entirely, returning `[]`.
2. **Feature extraction** (line 88): for each chip,
   `ReferenceManager.get_references(chip, ref)` picks the resistance-cluster-correct
   `(temp_ref, resist_ref, cluster)` triple, then `extract_chip_features` computes
   `resist_roughness`/`resist_spike`/`resist_cluster`, and `extract_residual_vector`
   builds the joint `[temp|resist]` autoencoder input.
3. **Resistance-touching detectors** (lines 111-165):
   - `detect_intra_wafer_drift(..., channel="resist")` — CUSUM on the resist roughness
     sequence within this wafer.
   - `detect_chip_spikes` + `SpikeRateTracker.update_and_detect` — resistance-only,
     see `resi-detection-models` §1.
   - `InterWaferTracker.update_and_detect(..., channel="resist")` — separately from
     the `channel="temp"` call right next to it in the source.
   - `MultivariateInterWaferTracker.update_and_detect` — fed
     `[temp_roughness_mean, resist_roughness_mean]` jointly.
   - `CouplingDetector.process_wafer` — re-clusters by resistance, compares temp
     profiles across clusters.
   - `AutoencoderAnomalyDetector.process_wafer` — scores the joint `[temp|resist]`
     residual vector.
4. **Update reference + roughness baseline** (line 167) — after detection, using this
   wafer's resistance (and temperature) data; includes re-running bimodal detection,
   so a group's bimodal/unimodal status can change wafer to wafer if the underlying
   process does.

## 2. Output schema, resistance angle — `AnomalyEvent` (`data_models.py:100`)

`channel` disambiguates which events are resistance-relevant:
- `channel="resist"`: `spike` (per-chip and wafer-rate), `inter_drift` from the
  resist-channel `InterWaferTracker` call, `intra_drift` from the resist-channel
  `detect_intra_wafer_drift` call.
- `channel="both"`: `coupling`, `deep_anomaly`, `inter_drift_mv` — joint detectors that
  need resistance but aren't resistance-only (see `resi-detection-models` §3-5).

To pull just the resistance-relevant events out of a `process_wafer` result:

```python
resist_events = [e for e in events if e.channel in ("resist", "both")]
```

## 3. Reproducing a full run on synthetic resistance data

```python
from hbm_anomaly_detection.pipeline import AnomalyDetectionPipeline
from hbm_anomaly_detection.simulator import make_normal_wafer_sequence, make_spike_wafer
import numpy as np

pipeline = AnomalyDetectionPipeline()

baseline = make_normal_wafer_sequence(n_wafers=10, n_chips=100, n_points=500, seed=50)
for w in baseline:
    pipeline.process_wafer(w)

spike_wafer = make_spike_wafer(
    wafer_id="SPIKE_W", n_chips=100, n_points=500,
    spike_fraction=0.20, spike_amplitude=8.0,
    rng=np.random.default_rng(200),
)
events = pipeline.process_wafer(spike_wafer)
assert any(e.anomaly_type == "spike" for e in events)
```

For bimodal/coupling reproduction, replace the baseline with
`make_coupling_breakdown_sequence` (see `resi-data-preprocessing` §2a) and inspect
`e.anomaly_type == "coupling"` events in the broken-coupling portion of the sequence.

Validate end to end: `python -m pytest hbm_anomaly_detection/tests/test_pipeline.py -v`
(covers spike, coupling, joint T², and autoencoder integration alongside temperature).

## 4. Production deployment — the gap

`temp_anomaly/scripts/` + `temp_anomaly/dags/temp_vibration_dag.py` give
`TempVibrationPipeline` a full ClickHouse-ingestion + watermark-incremental +
Airflow-DAG deployment (see `temp-inference-pipeline` §4) — **no equivalent exists for
`AnomalyDetectionPipeline`**. To productionize resistance detection, replicate that
script trio with these changes:

1. Ingestion (`_common.py` equivalent): decode a `resist_raw` column alongside
   `temp_raw` (see `resi-data-preprocessing` §2b — the column doesn't currently exist
   in the queried table) and pass it as `resist_series` when building `ChipData`.
2. Pipeline: swap `TempVibrationPipeline` for `AnomalyDetectionPipeline` everywhere.
3. State persistence: `AnomalyDetectionPipeline` does **not** currently have
   `save_baseline`/`load_baseline`/`initialize_baseline`/`reset_baseline` methods —
   `TempVibrationPipeline` added those (see `temp-inference-pipeline` §2) but they were
   never ported to the joint pipeline. You'd need to either add them (mirroring
   `TempVibrationPipeline`'s implementation, but also serializing `BimodalState`,
   `SpikeRateTracker`, `CouplingDetector`, `MultivariateInterWaferTracker`, and
   `AutoencoderAnomalyDetector` state — considerably more than the temp-only version
   persists) or accept re-deriving state from a lookback window on every process
   restart.
4. Watermark/incremental logic and the Airflow DAG shell-out pattern from
   `run_anomaly_detection.py` carry over unchanged in structure.

## Reproducibility checklist

- Stateful and sequential — replay wafers for a group in order from a fresh
  `AnomalyDetectionPipeline()` when reproducing a specific result.
- Expect zero events on a group's first wafer, and generally fewer/no events during
  each detector's own baseline period (`SpikeConfig` has none — see
  `resi-detection-models` §1; `CouplingConfig.baseline_wafers`,
  `AutoencoderConfig.baseline_wafers`, `MultivariateInterWaferConfig.baseline_wafers`
  all default to 8-10).
- A group's bimodal status can flip between wafers — don't assume `resist_cluster`
  labels are stable identifiers across the whole history; always re-derive from
  `ReferenceManager.assign_cluster` for the wafer you're inspecting.
- End-to-end validation: `python -m pytest hbm_anomaly_detection/tests/test_pipeline.py -v`.
