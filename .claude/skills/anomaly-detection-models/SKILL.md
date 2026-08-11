---
name: anomaly-detection-models
description: How each anomaly-detection model/detector in this repo works — CUSUM intra-wafer drift, EWMA and Hotelling's T² (MEWMA) inter-wafer drift, MAD-based spike detection, bimodal temperature-coupling breakdown, the hand-rolled NumPy autoencoder, and the Marchenko-Pastur PCA (RMT) detector. Use this whenever the user asks how a detector is trained/calibrated, wants to tune detection thresholds or sensitivity, wants to add a new detector, asks what a given AnomalyEvent.anomaly_type means or how its severity is computed, or wants to reproduce/retrain the autoencoder model in hbm_anomaly_detection/detector/autoencoder.py or rmt_profile_detector/. Load anomaly-data-preprocessing first if you need to know where a detector's input features come from.
---

# The detection models

Every "model" in this repo follows the same shape: a stateful class keyed by
`GroupKey`, with a **baseline period** (accumulate statistics, raise no alarms) that
**freezes** after enough wafers, after which new observations are scored against the
frozen baseline and emit `AnomalyEvent` objects (`data_models.py:100`) when they exceed
a threshold. This freezing is deliberate — see `inter_wafer.py`'s docstring (line 8):
if the baseline kept adaptively updating, a genuine slow drift could raise its own
detection threshold and evade detection forever. Read `anomaly-data-preprocessing`
first for what `ChipFeatures`/`WaferFeatures`/residual vectors are; this skill covers
what happens to them. All six detectors below live in `hbm_anomaly_detection/detector/`
and are wired together in `pipeline.py` (see `anomaly-inference-pipeline`).

## Detector reference table

| Detector | File | Method | `anomaly_type` | Config |
|---|---|---|---|---|
| Intra-wafer drift | `detector/intra_wafer.py` | Self-calibrating CUSUM | `intra_drift` | `IntraWaferConfig` |
| Inter-wafer drift (univariate) | `detector/inter_wafer.py` | EWMA + frozen Gaussian control limit | `inter_drift` | `InterWaferConfig` |
| Inter-wafer drift (joint) | `detector/multivariate_inter_wafer.py` | MEWMA + Hotelling's T² | `inter_drift_mv` | `MultivariateInterWaferConfig` |
| Resistance spikes | `detector/spike_detector.py` | Median+MAD per-chip; z-score wafer rate | `spike` | `SpikeConfig` |
| Coupling breakdown | `detector/coupling_detector.py` | Cluster-mean profile L2 distance + EWMA | `coupling` | `CouplingConfig` |
| Deep/joint signature | `detector/autoencoder.py` | NumPy MLP autoencoder, reconstruction error | `deep_anomaly` | `AutoencoderConfig` |
| (alt. package) MP-score | `rmt_profile_detector/` | PCA + Marchenko-Pastur bulk-edge partition | `mp_anomaly_temp`/`mp_anomaly_resist` | `MPConfig` |

All configs are sub-dataclasses of `AnomalyConfig` in `config.py:84`; `DEFAULT_CONFIG`
(line 96) is the production default. `tests/test_pipeline.py::_test_config()` (line 38)
shows a full override tuned for small/fast synthetic runs — use it as a template when
demonstrating a detector rather than guessing parameter values.

## 1. Intra-wafer drift — `detector/intra_wafer.py`

Detects vibration/roughness *increasing progressively within one wafer's bond
sequence*. Deliberately **not** dependent on any cross-wafer baseline — it estimates
"normal" from the first `warmup_chips` (default 10) chips of the *same* wafer
(`warmup_mean`, `warmup_sample_std`, line 88-89), floors the std at
`max(sample_std, warmup_mean * 0.15, 1e-9)` (the `_MIN_CV = 0.15` constant, line 22) so
a lucky ultra-homogeneous warmup can't make it hypersensitive, then z-scores the rest
of the sequence and runs a one-sided CUSUM (`_cusum_chart`, line 25):
`S[i] = max(0, S[i-1] + z[i] - k)`, alarm when `S[i] > h`. `k` (`cusum_k`, default 0.5)
is the allowance (~detectable shift size in sigma), `h` (`cusum_h`, default 4.0) is the
decision interval. `severity` in the emitted event is the CUSUM peak value.

Because it self-calibrates per wafer, it works from the very first anomalous wafer of
a group with no separate training step — this is the one detector you can trust from
day one.

## 2. Inter-wafer drift, univariate — `detector/inter_wafer.py`

Detects a *slow, multi-wafer* upward trend in a channel's roughness that no single
wafer would show. `InterWaferTracker.update_and_detect(key, wafer_id, wafer_roughness_mean, channel)`
(line 67): feeds the wafer's mean roughness into an EWMA (`ewma_lambda`, default 0.1)
every call; during the first `baseline_wafers` (default 10) calls it also accumulates a
Welford mean/std of the raw values (not the EWMA) — after that the baseline **freezes**
(`state.baseline_frozen = True`, checked every call thereafter) and alarms fire when
`ewma_value > baseline_mean + alert_sigma * baseline_std` (default `alert_sigma=3.0`).
`severity` = sigma distance from baseline mean. Run once per channel (`"temp"` and
`"resist"` independently) from `pipeline.py`.

## 3. Inter-wafer drift, joint — `detector/multivariate_inter_wafer.py`

The univariate tracker above can miss a drift where temp roughness rises *while*
resist roughness stays flat, if that shifts the *correlation* between channels rather
than either channel's marginal distribution — see the module docstring (lines 6-13).
`MultivariateInterWaferTracker` runs a 2-D MEWMA over
`[temp_roughness_mean, resist_roughness_mean]` and computes Hotelling's
`T² = diffᵀ · Σ⁻¹ · diff` against the frozen baseline covariance (`baseline_cov`,
regularized by `_COV_REG = 1e-6` for invertibility). Alarm when
`T² > chi2.ppf(1 - alert_alpha, df=2)` (`alert_alpha` default 0.01) — a theoretical
threshold, not something you calibrate from data. `severity` = the raw T² value.

## 4. Resistance spikes — `detector/spike_detector.py`

Two levels:
- **Per-chip** (`detect_chip_spikes`, line 90): within one wafer, threshold =
  `median(resist_spike) + residual_sigma * 1.4826 * MAD(resist_spike)`
  (`_compute_spike_threshold`, line 17) — median/MAD are robust to the spiked chips
  themselves as long as <50% of the wafer is anomalous, so the threshold doesn't get
  dragged up by the very spikes it's trying to catch. `1.4826` is the standard
  MAD→sigma consistency factor for a Gaussian.
- **Wafer-level rate trend** (`SpikeRateTracker`, line 36): tracks the fraction of
  spiked chips per wafer across wafers seen so far; once ≥5 wafers observed, z-scores
  the latest rate against the mean/std of *all previous* rates (`arr[:-1]`, i.e.
  leave-one-out, not a frozen baseline) and alarms above `wafer_spike_rate_sigma`
  (default 3.0).

## 5. Coupling breakdown — `detector/coupling_detector.py`

Only applicable to equipment whose resistance is **bimodal** (two chip types per
wafer — see `anomaly-data-preprocessing` §3). The assumption being checked: even
though resistance differs by cluster, the *temperature* profile should be identical
across clusters. `CouplingDetector.process_wafer` (line 83): re-clusters resistance
per-wafer (`_kmeans2_resist`, line 33; returns early if `separation <=
bimodal_separation_threshold` — i.e. unimodal wafers produce no events at all), computes
each cluster's mean temperature profile (`_mean_profile`), then their normalized L2
distance (`_profile_distance`, line 27). That scalar distance feeds the same
EWMA-then-frozen-baseline pattern as detector #2 (`_update_and_detect`, line 125),
alarming when `z = (ewma_distance - baseline_mean) / baseline_std > coupling_sigma`
(default 3.0).

## 6. Deep/joint signature — `detector/autoencoder.py`

The only model here that is literally *trained*, and the only one with no external ML
dependency — hand-rolled NumPy, no torch/sklearn.

**Architecture** (`SimpleAutoencoder`, line 61): `input → tanh(hidden) → tanh(bottleneck)
→ tanh(hidden) → linear(input)`, i.e. a 4-layer MLP autoencoder. Input is the
`2*input_length`-dim residual vector from `extract_residual_vector` (see preprocessing
skill §4) — `input_length` defaults to 25, so input dim is 50. `hidden_dim=16`,
`bottleneck_dim=4` by default (`AutoencoderConfig`, `config.py:67`). Weights are
He-initialized (`_init`, line 74) with a fixed `seed`.

**Training** (`fit`, line 102): full-batch **hand-rolled Adam** (β1=0.9, β2=0.999) on
MSE reconstruction loss, backprop written out explicitly layer by layer (lines 113-140)
— there's no autograd. `epochs=150`, `lr=0.02` by default.

**Per-group lifecycle** (`AutoencoderAnomalyDetector.process_wafer`, line 185, and
`_fit`, line 245):
1. Buffer `residual_vectors` from each wafer into `state.training_data` until
   `n_wafers_seen >= baseline_wafers` (default 8).
2. `_fit`: per-channel std-normalize (`scale_temp`/`scale_resist` computed from the
   buffered data, `_normalize`, line 267 — temp and resist residuals have very
   different natural scales, so they're normalized independently before
   concatenation), fit the MLP, then set `threshold = percentile(train_errors,
   threshold_percentile)` (default 99th percentile) — **the threshold is
   data-derived, not theoretical**, unlike detector #3 or the RMT model below.
3. Once fitted, every subsequent chip is scored by `reconstruction_error` and an event
   (`deep_anomaly`, `severity = err/threshold`) is emitted if it exceeds the threshold.
4. If `retrain_interval_wafers > 0` (default `0` = never), buffering resumes after the
   initial fit and the model is periodically re-fit (fresh scale + threshold too) on
   the most recent window — see `tests/test_pipeline.py::TestDeepAutoencoderDetector.
   test_periodic_retrain` for the exact expected behavior.

Reproduce training standalone (outside the pipeline):

```python
import numpy as np
from hbm_anomaly_detection.detector.autoencoder import SimpleAutoencoder

rng = np.random.default_rng(0)
X = rng.normal(0, 1, size=(50, 10))   # 50 samples, 10-dim residual vectors

ae = SimpleAutoencoder(input_dim=10, hidden_dim=8, bottleneck_dim=2, seed=0)
err_before = ae.reconstruction_error(X).mean()
ae.fit(X, epochs=200, lr=0.05)
err_after = ae.reconstruction_error(X).mean()
assert err_after < err_before
```

## 7. Alternative model — `rmt_profile_detector/` (Marchenko-Pastur PCA)

A standalone, drop-in alternative signature detector using **Random Matrix Theory**
instead of a trained autoencoder — no gradient descent, no data-driven threshold.
`base_profile.py` docstring (lines 1-28) has the full derivation. Given baseline
residual matrix `X ∈ R^{n×p}`:

1. `BaseProfile.fit(X, min_noise_rank)` (line 89): eigendecompose the sample covariance
   `Σ = XᵀX/(n-1)`. Under a pure-noise null, eigenvalues should lie below the
   Marchenko-Pastur bulk edge `λ+ = σ²(1+√γ)²` where `γ = p/n`. Eigenvalues **above**
   `λ+` are "signal" (structured common-mode variation, e.g. the shared bonding
   profile shape); the rest form the "noise subspace" (`noise_vecs_`,
   `noise_eigenvalues_`).
2. `score(x)` (line 132): project a new residual into the noise subspace and compute
   `Σᵢ (cᵢ² / λᵢ)` — under the null this is exactly `chi²(k_noise)` distributed.
3. `threshold(alert_alpha)` (line 146): `chi2.ppf(1 - alert_alpha, df=k_noise)` —
   **purely theoretical**, no percentile calibration against training data needed
   (contrast with the autoencoder's `threshold_percentile`).

`MPProfileDetector` (`mp_detector.py`) wraps this in the same
buffer-then-fit-then-score lifecycle as the autoencoder, per group and per channel
(`temp`/`resist` scored independently, unlike the autoencoder which scores them
jointly). It imports `ChipData`/`GroupKey`/`AnomalyEvent` directly from
`hbm_anomaly_detection` (see `mp_detector.py:28`) — it's designed to slot into the same
pipeline as an additional detector, not to run standalone.

## Adding a new detector

Follow the established shape so it composes with the others in `pipeline.py`:
1. A `@dataclass State` holding whatever running statistics you need (see
   `InterWaferState`, `CouplingState`, `AutoencoderState` for examples of the
   accumulate-then-freeze pattern).
2. A `@dataclass Tracker`/`Detector` with `_states: dict[GroupKey, State]` and either
   `update_and_detect(group_key, wafer_id, ...) -> list[AnomalyEvent]` (stat-based) or
   `process_wafer(group_key, wafer_id, chips, ...) -> list[AnomalyEvent]` (needs raw
   chip access), returning `AnomalyEvent`s with a new `anomaly_type` string.
3. Add a config dataclass to `config.py` and a field on `AnomalyConfig`.
4. Wire it into `AnomalyDetectionPipeline.__init__`/`process_wafer` (see
   `anomaly-inference-pipeline`).
5. Add a synthetic scenario generator to `simulator.py` if the anomaly needs specific
   synthetic data, then a `TestYourDetector` class in `tests/test_pipeline.py` mirroring
   the existing ones (a "no false positives on normal data" test + a "detects the
   injected anomaly" test).

## Reproducibility checklist

- Detector thresholds come from `AnomalyConfig` — never hardcode a sigma/percentile
  when demonstrating; construct the config explicitly (see `_test_config()` in
  `tests/test_pipeline.py:38` for a complete, working example).
- The autoencoder and RMT model need `seed` fixed for reproducible weights/output.
- To validate a specific model's behavior: `python -m pytest
  hbm_anomaly_detection/tests/test_pipeline.py -k "<DetectorClassName>" -v`, e.g.
  `-k TestSimpleAutoencoder` or `-k TestMultivariateInterWaferDrift`.
