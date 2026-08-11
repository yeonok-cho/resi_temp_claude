---
name: resi-data-preprocessing
description: How this repo turns raw resistance (resist_series) time series into bimodal reference profiles and resistance roughness/spike features — the resistance-specific parts of hbm_anomaly_detection (there is no standalone resist-only package; resistance lives alongside temperature in the joint hbm_anomaly_detection package). Use this whenever the user asks about resist_series/resist_ref, bimodal resistance clustering, resist_roughness/resist_spike features, generating synthetic spike/bimodal wafers, or building/extending real-data ingestion for a resist_raw column (which does not yet exist in this repo). Also use it before touching hbm_anomaly_detection/feature_extractor.py, reference_manager.py, or simulator.py's resistance-related functions. For the temperature channel, use temp-data-preprocessing instead.
---

# Resistance-channel data preprocessing

Unlike temperature (`temp_anomaly/`, a genuinely standalone temp-only package — see
`temp-data-preprocessing`), **resistance has no standalone package**: it only exists
inside `hbm_anomaly_detection/`, the joint package, alongside temperature. Everything
below is resistance-specific code living in files that also handle temperature; where
a function/class handles both channels, only the resistance-relevant behavior is
described here (see `temp-data-preprocessing` for the temperature-side view of the
same files).

## 1. Data model — `hbm_anomaly_detection/data_models.py`

- `ChipData.resist_series: np.ndarray` (line 34) — same shape/length contract as
  `temp_series`: one resistance sample per bonding time point.
- `ChipFeatures` (line 62) carries `resist_roughness`, `resist_spike`, and
  `resist_cluster: int` (0 or 1, or 0 if the equipment isn't bimodal) — the bimodal
  cluster label is resistance-specific; there is no temperature equivalent.
- `WaferFeatures.roughness_sequence(channel="resist")` / `.spike_mask(threshold)` /
  `.summary_stats(channel="resist")` (lines 78-96) — pass `channel="resist"` explicitly;
  the default in some call sites is `"resist"` but don't rely on that, be explicit.
- `AnomalyEvent.channel` (line 105) is `"resist"` for resistance-only detections
  (spike, resist-side inter_wafer) or `"both"` for joint detections that depend on
  resistance (coupling, deep_anomaly, inter_drift_mv) — see `resi-detection-models`.

## 2. Two ways resistance data enters the pipeline

### 2a. Synthetic — `hbm_anomaly_detection/simulator.py`

Resistance-specific generators (the temperature generators are shared — see
`temp-data-preprocessing` §2a for `_temp_profile`):

| Function | Line | Produces |
|---|---|---|
| `_resist_profile_peak_jumps(n_points, peak_positions, peak_height)` | 48 | Two sharp Gaussian-shaped peaks — the default "normal" resist shape |
| `_resist_profile_ramp_hold(n_points, jump_frac, hold_level)` | 74 | Gradual ramp then a sharp jump to a hold level — the alternate/second bimodal-cluster shape |
| `_inject_spike(series, spike_pos_frac, spike_amplitude)` | 103 | Adds a single positive spike at a random-ish position |
| `make_spike_chip(spike_amplitude=5.0)` | 203 | One chip with an injected resistance spike (for `spike_detector`) |
| `make_spike_wafer(spike_fraction=0.15, spike_amplitude=5.0)` | 341 | A random fraction of chips in the wafer get a spike |
| `make_bimodal_wafer(coupling_broken=False, temp_offset_group1=10.0)` | 392 | Alternates chips between `peak_jumps` (cluster 0) and `ramp_hold` (cluster 1) resistance shapes; `coupling_broken=True` additionally offsets cluster 1's **temperature** (for `coupling_detector` — see `resi-detection-models`) |
| `make_coupling_breakdown_sequence(n_normal, n_broken, temp_offset)` | 529 | Bimodal-normal wafers, then wafers where coupling breaks |

Always pass an explicit `rng`/`seed`.

```python
from hbm_anomaly_detection.simulator import make_spike_wafer, make_bimodal_wafer
import numpy as np

spike_wafer = make_spike_wafer(
    wafer_id="SPIKE_W", n_chips=100, spike_fraction=0.15, spike_amplitude=5.0,
    rng=np.random.default_rng(7),
)
bimodal_wafer = make_bimodal_wafer(n_chips=100, coupling_broken=False, rng=np.random.default_rng(3))
```

### 2b. Real data — **not yet implemented in this repo**

`temp_anomaly/scripts/_common.py` (the only real-data ingestion code that exists) reads
`eqp_id, product, module_id, wafer_id, x, y, temp_raw, event_tmstp` from
`ds_catalog.aifpa_cow_rule_hist` and decodes only `temp_raw` — **there is no
`resist_raw` column read anywhere in this repo**, and no ingestion script populates
`ChipData.resist_series` from real data. If you need to build this:

1. Confirm (with whoever owns the ClickHouse schema) whether `resist_raw` already
   exists as a same-format hex-encoded float32 column on `ds_catalog.aifpa_cow_rule_hist`
   or a related table.
2. Mirror `_common.py`'s pattern exactly: reuse `decode_hex_to_float32` (already
   channel-agnostic — it just decodes hex to a float32 array, see
   `temp-data-preprocessing` §2b) on the new column, add `resist_raw` to
   `query_window`'s `SELECT`, and pass the decoded array as `resist_series=...` in
   `create_chip`.
3. Switch the consuming pipeline from `TempVibrationPipeline` to the joint
   `AnomalyDetectionPipeline` (`hbm_anomaly_detection.pipeline`), since that's the only
   pipeline that does anything with `resist_series` — see `resi-inference-pipeline`.

## 3. Reference profile & bimodal clustering — `hbm_anomaly_detection/reference_manager.py`

Resistance introduces a concept temperature doesn't need: **bimodal clustering**, because
some equipment bonds two physically different chip types per wafer with distinct
resistance baselines (module docstring, line 4).

`ReferenceManager.update(wafer)` (line 170), resistance-specific steps:

1. `_detect_bimodal(mean_resist_levels, config)` (line 68): needs
   `≥ bimodal_min_chips` (default 10) chips; runs 1-D k=2 k-means
   (`_kmeans2`, line 49) on each chip's `mean(resist_series)`, then declares bimodal
   when `(center1 - center0) / std(levels) > bimodal_separation_threshold` (default
   0.15).
2. If bimodal: `BimodalState` (line 18) stores `cluster_centers` and **two** separate
   reference profiles per channel — `bimodal.temp_refs[0/1]`, `bimodal.resist_refs[0/1]`
   — each built the same way as the unimodal reference (pointwise median → EWMA,
   `ewma_alpha` default 0.1) but restricted to that cluster's chips.
3. A global (all-chips-averaged) fallback reference (`ref.temp_ref`/`ref.resist_ref`)
   is *also* always maintained, used when the group isn't currently bimodal.
4. `ReferenceManager.assign_cluster(chip, ref)` (line 144) and
   `get_references(chip, ref)` (line 155) — given a chip, find its nearest bimodal
   centroid by `mean(chip.resist_series)` and return the matching
   `(temp_ref, resist_ref, cluster_label)` pair. This is what feature extraction and
   the autoencoder call to get the right reference per chip.

```python
from hbm_anomaly_detection.reference_manager import ReferenceManager
from hbm_anomaly_detection.simulator import make_bimodal_wafer
from hbm_anomaly_detection.data_models import GroupKey
import numpy as np

rm = ReferenceManager()
rm.update(make_bimodal_wafer(n_chips=60, n_points=300, rng=np.random.default_rng(3)))
ref = rm.get(GroupKey("EQ01", "R001", "H1"))
assert ref.bimodal.is_bimodal
print(ref.bimodal.cluster_centers)  # two distinct mean resistance levels
```

## 4. Feature extraction — `hbm_anomaly_detection/feature_extractor.py`

`extract_chip_features(chip, temp_reference, resist_reference, resist_cluster, config)`
(line 36) computes, for the resistance side: `resist_res = chip.resist_series[:ref_len]
- resist_reference[:ref_len]`, then:

- `compute_roughness(resist_res)` (line 16) → `resist_roughness` = `std(diff(residual))`
  — high-frequency oscillation only, insensitive to DC offset/slow trend (feeds
  `intra_wafer`/`inter_wafer` when run with `channel="resist"`).
- `compute_spike(resist_res)` (line 29) → `resist_spike` = `max(abs(residual))` —
  catches single-point outliers (feeds `spike_detector`, which is resistance-only —
  see `resi-detection-models`).

The autoencoder's input also needs a resistance contribution:
`detector/autoencoder.py::extract_residual_vector` (line 37) computes the same
`resist_res`, block-averages it to `input_length` points (`_downsample`, line 25), and
concatenates it as the **second half** of the joint `2*input_length` input vector (the
temperature residual is the first half — see `temp-detection-models` §3).

```python
from hbm_anomaly_detection.feature_extractor import extract_chip_features
tref, rref, cluster = rm.get_references(chip, ref)
feat = extract_chip_features(chip, tref, rref, resist_cluster=cluster)
# feat.resist_roughness, feat.resist_spike, feat.resist_cluster
```

## Reproducibility checklist

- Always pass an explicit `rng`/`seed` to simulator functions.
- Bimodal detection needs `≥ ReferenceConfig.bimodal_min_chips` chips per wafer (default
  10) or it silently stays unimodal — don't build tiny test wafers and expect bimodal
  behavior.
- `ReferenceManager.get_references(chip, ref)` is the only correct way to fetch a
  chip's reference — don't read `ref.resist_ref` directly once a group is bimodal, it's
  only the unimodal fallback.
- Validate preprocessing changes:
  `python -m pytest hbm_anomaly_detection/tests/test_pipeline.py -k "FeatureExtractor or ReferenceManager" -v`.
