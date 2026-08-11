---
name: temp-detection-models
description: How the temperature-channel anomaly detectors work in this repo — the self-calibrating CUSUM intra-wafer drift detector and the frozen-baseline EWMA inter-wafer drift detector in temp_anomaly/temp_vibration_detection/detector/, plus how temperature feeds into the joint (temp+resist) autoencoder, Hotelling's T² (MEWMA), and coupling-breakdown detectors in hbm_anomaly_detection/detector/. Use this whenever the user asks how temperature drift/vibration detection is calibrated, wants to tune its CUSUM/EWMA thresholds, wants to add a new temperature detector, or asks what a temp AnomalyEvent.anomaly_type/severity means. Load temp-data-preprocessing first for where temp_roughness comes from. For resistance-only or joint models from the resistance side, use resi-detection-models.
---

# Temperature-channel detection models

`temp_anomaly/temp_vibration_detection/detector/` ships exactly two models, both
operating on `temp_roughness` (see `temp-data-preprocessing` §4). Both follow the
repo-wide pattern: a stateful class keyed by `GroupKey`, with a baseline/warm-up period
that **freezes**, after which new observations are scored and emit `AnomalyEvent`s
(`data_models.py:87`) — `anomaly_type` is `"intra_drift"` or `"inter_drift"`, and (since
this package is temperature-only) there is no `channel` field to disambiguate.

If you're working with the resistance channel or need the full joint pipeline, the
equivalent/generic versions of these two detectors also exist in
`hbm_anomaly_detection/detector/{intra_wafer,inter_wafer}.py`, parameterized by a
`channel` argument — see `resi-detection-models`.

## 1. Intra-wafer drift — `detector/intra_wafer.py`

Detects roughness increasing progressively **within one wafer's bond sequence**.
Self-calibrating — no cross-wafer baseline needed, so it works from the very first
wafer of a new group. `detect_intra_wafer_drift(wafer_features, group_key, config)`
(line 47):

1. Needs at least `warmup_chips * 2` chips (default `warmup_chips=10`); otherwise
   returns `[]`.
2. Estimates "normal" from the first `warmup_chips` of *this* wafer:
   `warmup_mean`, `warmup_sample_std` (lines 79-80).
3. Floors the std at `max(sample_std, warmup_mean * 0.15, 1e-9)` (`_MIN_CV = 0.15`,
   line 22) so a coincidentally homogeneous warmup window can't make the chart
   hypersensitive to normal chip-to-chip variation.
4. Z-scores the remaining chips against `(warmup_mean, effective_std)`, then runs a
   one-sided CUSUM (`_cusum_chart`, line 25):
   `S[i] = max(0, S[i-1] + z[i] - cusum_k)`, alarm when `S[i] > cusum_h`
   (`IntraWaferConfig`: `cusum_k=0.5`, `cusum_h=4.0`, `config.py:24`).
5. Emits one `AnomalyEvent(anomaly_type="intra_drift", severity=<CUSUM peak>)` with
   `details={"cusum_peak", "alarm_chip_index", "n_chips", "warmup_mean",
   "effective_std"}` if triggered.

This is line-for-line the same algorithm as `hbm_anomaly_detection/detector/
intra_wafer.py`, just without the `channel` parameter (this package only ever scores
temperature) and without the unused `baseline_mean`/`baseline_std` compatibility
parameters the joint version still carries.

## 2. Inter-wafer drift — `detector/inter_wafer.py`

Detects a **slow, multi-wafer** upward trend in wafer-mean roughness that no single
wafer shows. `InterWaferTracker.update_and_detect(group_key, wafer_id,
wafer_roughness_mean)` (line 72):

1. Feeds `wafer_roughness_mean` into an EWMA every call (`ewma_lambda`, default 0.1).
2. During the first `baseline_wafers` calls (default 10), also accumulates a Welford
   mean/std of the **raw** (non-EWMA) values; after that, `state.baseline_frozen =
   True` permanently — this is the mechanism preventing a real drift from adaptively
   raising its own detection ceiling (module docstring, lines 8-10).
3. Once frozen: alarm when `ewma_value > baseline_mean + alert_sigma * baseline_std`
   (`alert_sigma` default 3.0). `severity` = sigma distance from baseline mean.

`InterWaferTracker.reset(group_key)` (line 62) discards a group's EWMA state entirely
— call this together with `ReferenceManager.reset()` after a PM event (see
`temp-data-preprocessing` §3 and `temp-inference-pipeline` §1) so the baseline
re-accumulates from the post-maintenance process instead of blending old and new
behavior.

```python
from temp_vibration_detection.detector.inter_wafer import InterWaferTracker
from temp_vibration_detection.config import InterWaferConfig

tracker = InterWaferTracker(InterWaferConfig(baseline_wafers=5, alert_sigma=2.5))
for i, roughness in enumerate([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 5.0]):  # spike on wafer 7
    events = tracker.update_and_detect(("EQ01", "R001", "H1"), f"W{i}", roughness)
    if events:
        print(events[0])
```

## 3. How temperature feeds the joint (temp+resist) models

These three detectors only exist in `hbm_anomaly_detection/` (the joint package) — they
are **not** present in `temp_anomaly`, which explicitly excludes them. Full mechanics
(config, formulas) are also documented in `resi-detection-models` from the resistance
side; this section is the temperature-side view of the same code.

**Joint inter-wafer drift (Hotelling's T² / MEWMA)** — `hbm_anomaly_detection/
detector/multivariate_inter_wafer.py`. Runs a 2-D MEWMA over `[temp_roughness_mean,
resist_roughness_mean]` and alarms when
`T² = diffᵀ · Σ⁻¹ · diff > chi2.ppf(1 - alert_alpha, df=2)` against the frozen baseline
covariance. The point of tracking temp and resist *jointly* rather than as two
independent `InterWaferTracker`s (§2 above, run once per channel in the joint
pipeline) is that it also catches a shift in the *correlation* between the channels —
e.g. temp roughness rising while resist roughness stays flat — which two independent
univariate charts cannot see (module docstring, lines 6-13). `anomaly_type =
"inter_drift_mv"`, `channel = "both"`.

**Deep/joint signature (autoencoder)** — `hbm_anomaly_detection/detector/
autoencoder.py`. `extract_residual_vector` downsamples the temp residual to
`input_length` points (default 25) and concatenates it with the equally-sized
downsampled resist residual into one `2*input_length`-dim vector; the temp and resist
halves are **independently std-normalized** before training (different natural
scales — `_normalize`, line 267) so temperature doesn't dominate or get drowned out by
resistance's typically smaller magnitude. The hand-rolled NumPy MLP autoencoder
(`SimpleAutoencoder`, no torch/sklearn) is trained on this joint vector; reconstruction
error is the anomaly score. `anomaly_type = "deep_anomaly"`, `channel = "both"`.

**Coupling breakdown** — `hbm_anomaly_detection/detector/coupling_detector.py`. Clusters
chips by resistance level (bimodal equipment only), then checks whether the two
clusters' **mean temperature profiles** still match (they should — the assumption is
that temperature is a shared process step independent of which resistance mode a chip
lands in). A large L2 distance between the two temp profiles, sustained past an EWMA
baseline, means that assumption broke. `anomaly_type = "coupling"`, `channel = "both"`.
Since clustering is resistance-driven, see `resi-detection-models` for the clustering
mechanics; this detector's actual signal is a temperature-profile comparison.

## Config — `temp_anomaly/temp_vibration_detection/config.py`

`AnomalyConfig` (line 40) aggregates `ReferenceConfig`, `FeatureConfig`,
`IntraWaferConfig`, `InterWaferConfig` — no `SpikeConfig`/`CouplingConfig`/
`AutoencoderConfig`/`MultivariateInterWaferConfig` exist in this package (those are
joint-package-only; see `resi-detection-models`). `DEFAULT_CONFIG` (line 48) is the
production default.

## Adding a new temperature detector

Follow the shape of `InterWaferTracker`: a `@dataclass State` with whatever running
statistics you need, a `@dataclass Tracker` holding `_states: dict[GroupKey, State]`,
and `update_and_detect(group_key, wafer_id, ...) -> list[AnomalyEvent]`. Add a config
dataclass to `config.py` and a field on `AnomalyConfig`, wire it into
`TempVibrationPipeline.process_wafer` (see `temp-inference-pipeline`), and add a
matching synthetic scenario generator to `simulator.py` plus a test class in
`tests/test_pipeline.py` (a "no false positives on normal data" test + a "detects the
injected anomaly" test, mirroring the existing `TestIntraWaferDrift`/
`TestInterWaferDrift` classes).

## Reproducibility checklist

- Construct `IntraWaferConfig`/`InterWaferConfig` explicitly rather than hardcoding
  sigma/CUSUM values — see `tests/test_pipeline.py::_test_config()` for a working
  example tuned for fast synthetic runs.
- To validate a specific detector: `python -m pytest
  temp_anomaly/temp_vibration_detection/tests/test_pipeline.py -k
  "TestIntraWaferDrift or TestInterWaferDrift" -v`.
