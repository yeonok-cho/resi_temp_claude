---
name: anomaly-data-preprocessing
description: How this repo turns raw wafer-bonding time series (real ClickHouse rows or synthetic test data) into the ChipData/WaferData objects, reference profiles, and roughness/spike features that the anomaly detectors consume. Use this whenever the user asks to ingest or decode temp_raw/resist_raw columns, build ChipData/WaferData from a DataFrame, generate synthetic wafers for testing, build or update reference profiles (EWMA, bimodal clustering), extract roughness/spike/residual features, or debug why a wafer's features/reference look wrong — for the HBM CoW bonding (hbm_anomaly_detection) or temperature-vibration (temp_anomaly) pipelines in this repo. Also use it before touching feature_extractor.py, reference_manager.py, simulator.py, or the ClickHouse ingestion scripts under temp_anomaly/scripts/.
---

# Data preprocessing for the wafer-bonding anomaly pipelines

This repo detects anomalies in per-chip temperature/resistance time series recorded
during HBM CoW (Chip-on-Wafer) bonding. Preprocessing turns raw per-chip time series
into two things the detectors need: a **reference profile** (what "normal" looks like
for this equipment/recipe/head) and **per-chip features** (roughness, spikes, residual
vectors). Everything here is shared, near-identically, between
`hbm_anomaly_detection/` (the full reference implementation — temp + resistance,
bimodal clustering, autoencoder) and `temp_anomaly/temp_vibration_detection/` (a
temperature-only subset used in production against ClickHouse). File paths below are
`hbm_anomaly_detection/...` unless noted; the `temp_anomaly` equivalents have the same
names/line shapes minus the resistance channel.

Read `anomaly-detection-models` for what happens to these features next, and
`anomaly-inference-pipeline` for how a wafer flows through the whole system end to end.

## 1. Data model — what a "chip" and "wafer" are

`hbm_anomaly_detection/data_models.py`

- `GroupKey(equipment_id, recipe, head)` (line 12) — the unit that shares one baseline.
  Every reference profile, EWMA tracker, and trained model is keyed by this tuple, so
  different equipment/recipe/head combinations never share statistics.
- `ChipData` (line 20) — one bonded chip: `temp_series` and `resist_series` are
  equal-length `np.ndarray`s (one sample per time point during bonding), plus
  `x`, `y`, `bond_order` (0-based position in the wafer's bonding sequence) and the
  three `GroupKey` fields. `chip.group_key` and `chip.n_points` are derived properties.
- `WaferData` (line 46) — `wafer_id` + `list[ChipData]` for one track-out event.
  `chips_sorted_by_bond_order()` is used everywhere detection needs sequence order.
- `ChipFeatures` / `WaferFeatures` (lines 62, 73) — the *output* of this preprocessing
  stage (see §4).

Reproduce the shape by hand:

```python
import numpy as np
from hbm_anomaly_detection.data_models import ChipData, WaferData

chip = ChipData(
    equipment_id="EQ01", recipe="R001", head="H1", wafer_id="W001",
    x=3, y=5, bond_order=0,
    temp_series=np.linspace(25, 300, 500),
    resist_series=np.full(500, 0.5),
)
wafer = WaferData(wafer_id="W001", chips=[chip])
assert wafer.group_key == chip.group_key
```

## 2. Two ways data enters the pipeline

### 2a. Synthetic data (`hbm_anomaly_detection/simulator.py`) — use this to reproduce anything

There is no real fab data in this repo; every test, demo, and this skill's examples run
on the simulator. It generates physically-shaped profiles (temperature: ramp → hold →
cool-down via `_temp_profile`, line 20; resistance: either `_resist_profile_peak_jumps`
line 48, or `_resist_profile_ramp_hold` line 74) plus Gaussian noise, then injects
specific anomaly types on top:

| Function | Line | Produces |
|---|---|---|
| `make_normal_wafer(...)` | 253 | Clean wafer, no anomaly |
| `make_intra_vibration_wafer(...)` | 291 | Noise std ramps up progressively within one wafer's bond sequence (for `intra_wafer` detector) |
| `make_spike_wafer(...)` | 341 | A fraction of chips get an injected resistance spike (for `spike_detector`) |
| `make_bimodal_wafer(coupling_broken=...)` | 392 | Two resistance clusters; `coupling_broken=True` also offsets cluster-1 temperature (for `coupling_detector`) |
| `make_normal_wafer_sequence(n_wafers=...)` | 452 | List of normal wafers for baseline warm-up |
| `make_inter_wafer_drift_sequence(n_normal, n_drift, drift_multiplier_final)` | 481 | Normal wafers, then wafers with gradually increasing resist noise (for `inter_wafer` detector) |
| `make_coupling_breakdown_sequence(n_normal, n_broken, temp_offset)` | 529 | Bimodal wafers, then wafers where coupling breaks |

Every generator takes an `rng: np.random.Generator` (or a `seed` on the `*_sequence`
helpers) — **always pass one explicitly** for reproducible output; the tests do this
(`np.random.default_rng(42)` etc.) and so should you when demonstrating or debugging
a detector.

```python
from hbm_anomaly_detection.simulator import make_normal_wafer_sequence, make_spike_wafer
import numpy as np

baseline = make_normal_wafer_sequence(n_wafers=15, n_chips=100, n_points=500, seed=42)
spike_wafer = make_spike_wafer(
    wafer_id="SPIKE_W", n_chips=100, spike_fraction=0.15, spike_amplitude=5.0,
    rng=np.random.default_rng(7),
)
```

### 2b. Real data (ClickHouse) — `temp_anomaly/scripts/_common.py`

Production ingestion reads hex-encoded float32 arrays out of ClickHouse and builds the
exact same `ChipData`/`WaferData` objects:

- Table: `ds_catalog.aifpa_cow_rule_hist`, columns `eqp_id, product, module_id,
  wafer_id, x, y, temp_raw, event_tmstp` (`_common.py:21`, query in `query_window`
  line 85).
- `decode_hex_to_float32(hex_str)` (line 24) — each sample is a little-endian float32;
  the column is `hex(bytes)`, decoded via `struct.unpack('<f', ...)` in 4-byte chunks.
  Returns `np.array([], dtype=float32)` on empty/malformed input rather than raising —
  callers filter out chips with `len(temp_series) == 0`.
- `create_chip(row, bond_order)` (line 38) maps DataFrame columns → `ChipData` fields:
  `eqp_id→equipment_id`, `product→recipe`, `module_id→head` (falls back to `head` or
  `"unknown"`).
- `df_to_wafers(df)` (line 51) groups rows by `(eqp_id, product, module_id, wafer_id)`
  — **not** just `wafer_id`, so two equipment lines never collide on a coincidentally
  shared wafer_id — sorts chips within a wafer by `event_tmstp` (this *is* bond order),
  and sorts wafers within a group by each wafer's first timestamp (oldest first, which
  every downstream `process_wafer` call requires).

If you need to reproduce or extend real-data ingestion, copy this pattern rather than
inventing a new one — `build_baseline.py` and `run_anomaly_detection.py` both duplicate
`_common.py`'s functions inline (technical debt, not intentional divergence) so treat
`_common.py` as the source of truth.

## 3. Reference profile construction — `hbm_anomaly_detection/reference_manager.py`

The reference profile is "what normal looks like" per `GroupKey`, and is the baseline
against which every chip's residual is computed. Config: `ReferenceConfig` in
`config.py:8` (`ewma_alpha=0.1`, `bimodal_min_chips=10`, `bimodal_separation_threshold=0.15`).

`ReferenceManager.update(wafer)` (line 170) on each new wafer:

1. **Bimodal check** — `_detect_bimodal` (line 68) runs 1-D k=2 k-means
   (`_kmeans2`, line 49) on each chip's mean resistance level. Bimodal is declared when
   `(center1 - center0) / std(levels) > bimodal_separation_threshold`. This matters
   because some equipment bonds two physically different chip types per wafer with
   different resistance baselines (see `coupling_detector` in the model skill).
2. **Pointwise median reference** — `_pointwise_median` (line 89) stacks all chips'
   series (per cluster, if bimodal) and takes the per-timepoint median — robust to a
   few anomalous chips within the wafer.
3. **EWMA update** — `_ewma_update` (line 97): `ref = (1-α)*ref + α*new_median`, so the
   reference adapts slowly to legitimate process drift while damping single-wafer noise.
   The **first** wafer for a group hard-sets the reference (no blending) — see
   `ref.n_wafers_seen == 0` branch, line 238.
4. **Roughness baseline (Welford)** — `update_roughness_baseline` (line 247) tracks the
   running mean/std of wafer-level roughness using Welford's online algorithm
   (`_welford_update`, line 106) — this feeds the `inter_wafer` detector, not the
   reference profile itself.

`ReferenceManager.get_references(chip, ref)` (line 155) is how a detector/feature
extractor gets the right reference for a chip: it calls `assign_cluster` (nearest
bimodal centroid) then returns `(temp_ref, resist_ref, cluster_label)`.

Reproduce a reference from scratch:

```python
from hbm_anomaly_detection.reference_manager import ReferenceManager
from hbm_anomaly_detection.simulator import make_normal_wafer_sequence
from hbm_anomaly_detection.data_models import GroupKey

rm = ReferenceManager()
for w in make_normal_wafer_sequence(n_wafers=20, n_chips=30, n_points=500, seed=7):
    rm.update(w)

ref = rm.get(GroupKey("EQ01", "R001", "H1"))
assert ref.is_initialized()
print(ref.temp_ref.shape, ref.n_wafers_seen)  # (500,) 20
```
This mirrors `tests/test_pipeline.py::TestReferenceManager.test_ewma_smoothing`, which
asserts the converged reference has RMSE < 20°C against the true generator profile —
a good sanity check when you change EWMA parameters.

## 4. Feature extraction — `hbm_anomaly_detection/feature_extractor.py`

Given a chip and its reference, `extract_chip_features(chip, temp_reference,
resist_reference, resist_cluster, config)` (line 36):

1. Skips (returns all-zero `ChipFeatures`) if `chip.n_points < config.min_points`
   (`FeatureConfig.min_points`, default 10).
2. Aligns lengths (`ref_len = min(len(temp_reference), len(resist_reference), n)`) —
   reference and chip series can differ by a point due to rounding.
3. Computes the **residual**: `chip.series - reference` (this residual, not the raw
   series, is what every downstream detector actually looks at).
4. `compute_roughness(residual)` (line 16) = `std(diff(residual))` — first-order
   difference standard deviation. This is deliberately **insensitive to DC offset and
   slow trends** in the residual (see `tests/test_pipeline.py::TestFeatureExtractor.
   test_roughness_insensitive_to_dc_offset`) and only captures high-frequency
   oscillation — i.e., vibration, not miscalibration.
5. `compute_spike(residual)` (line 29) = `max(abs(residual))` — catches single-point
   outliers that roughness (a difference-based statistic) would dilute.

`WaferFeatures` (`data_models.py:73`) aggregates a wafer's `ChipFeatures` in bond order
via `roughness_sequence(channel)`, `spike_mask(threshold)`, and `summary_stats(channel)`
(mean/max/p90) — these are the inputs the intra-/inter-wafer detectors consume.

For the autoencoder model, preprocessing goes one step further:
`detector/autoencoder.py::extract_residual_vector(chip, temp_ref, resist_ref,
target_len)` (line 37) computes the same residuals, then block-averages
(`_downsample`, line 25) each channel down to `target_len` points and concatenates them
into a single `2*target_len`-length vector — this is what gets fed into the model
(see `anomaly-detection-models`).

```python
from hbm_anomaly_detection.feature_extractor import extract_chip_features
feat = extract_chip_features(chip, ref.temp_ref, ref.resist_ref, resist_cluster=0)
# feat.temp_roughness, feat.resist_roughness, feat.temp_spike, feat.resist_spike
```

## Reproducibility checklist

- Always pass an explicit `rng`/`seed` to simulator functions.
- Use `AnomalyConfig`/`ReferenceConfig`/`FeatureConfig` dataclasses (`config.py`) rather
  than hardcoding thresholds — `tests/test_pipeline.py::_test_config()` (line 38) is a
  working example of a full config override tuned for fast, small-N test runs.
- The reference for a group is only usable after its *first* wafer
  (`ref.is_initialized()`); pipelines skip detection on that first wafer by design (see
  `anomaly-inference-pipeline`) — don't expect features/detections on wafer #1 of a
  fresh group.
- To validate preprocessing changes, run:
  `python -m pytest hbm_anomaly_detection/tests/test_pipeline.py -k "FeatureExtractor or ReferenceManager" -v`
