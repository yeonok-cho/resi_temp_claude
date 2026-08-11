---
name: resi-detection-models
description: How the resistance-channel anomaly detectors work in this repo — the MAD-based per-chip and wafer-rate spike detector (resistance-only), plus how resistance feeds into the bimodal coupling-breakdown detector, the joint NumPy autoencoder, and Hotelling's T² (MEWMA) inter-wafer drift, all in hbm_anomaly_detection/detector/. Use this whenever the user asks how resistance spike/coupling detection is calibrated, wants to tune resist_spike thresholds, wants to add a new resistance detector, asks what a resist/both AnomalyEvent.anomaly_type/severity means, or wants to reproduce/retrain the autoencoder or the Marchenko-Pastur (rmt_profile_detector) model on resistance data. Load resi-data-preprocessing first for where resist_roughness/resist_spike come from. For temperature-only or joint models from the temperature side, use temp-detection-models.
---

# Resistance-channel detection models

All detectors below live in `hbm_anomaly_detection/detector/` — there is no
standalone resistance-only package (contrast with temperature's `temp_anomaly/`, see
`temp-detection-models`). Every detector follows the repo-wide pattern: stateful,
keyed by `GroupKey`, baseline period that **freezes**, then scoring against the frozen
baseline emits `AnomalyEvent`s. Read `resi-data-preprocessing` first for
`resist_roughness`/`resist_spike`/`resist_cluster`.

| Detector | File | Resistance role | `anomaly_type` | `channel` |
|---|---|---|---|---|
| Spike detection | `detector/spike_detector.py` | Resistance-only | `spike` | `"resist"` |
| Inter-wafer drift (univariate) | `detector/inter_wafer.py` | Generic tracker, run with `channel="resist"` | `inter_drift` | `"resist"` |
| Intra-wafer drift | `detector/intra_wafer.py` | Generic tracker, run with `channel="resist"` | `intra_drift` | `"resist"` |
| Coupling breakdown | `detector/coupling_detector.py` | Joint — resistance drives clustering | `coupling` | `"both"` |
| Deep/joint signature | `detector/autoencoder.py` | Joint — resistance is half the input | `deep_anomaly` | `"both"` |
| Inter-wafer drift (joint) | `detector/multivariate_inter_wafer.py` | Joint — resistance is one axis | `inter_drift_mv` | `"both"` |
| (alt. package) MP-score | `rmt_profile_detector/` | Independent resist-channel model | `mp_anomaly_resist` | `"resist"` |

## 1. Spike detection (resistance-only) — `detector/spike_detector.py`

The only detector in this repo that is purely resistance-specific end to end — there is
no temperature spike detector wired up anywhere (see `temp-data-preprocessing` §4).

**Per-chip** — `detect_chip_spikes(wafer_features, group_key, config)` (line 90):
threshold = `median(resist_spike_values) + residual_sigma * 1.4826 *
MAD(resist_spike_values)` (`_compute_spike_threshold`, line 17; `1.4826` is the
standard MAD→sigma consistency factor for a Gaussian; `residual_sigma` default 5.0).
Median/MAD are robust to the spiked chips themselves as long as fewer than 50% of the
wafer is anomalous — the threshold doesn't get dragged up by the very spikes it's
meant to catch. Emits one event per chip exceeding threshold:
`severity = resist_spike value`, `details = {chip_x, chip_y, bond_order,
spike_value, threshold}`.

**Wafer-level rate trend** — `SpikeRateTracker.update_and_detect(group_key, wafer_id,
spike_rate)` (line 49): tracks the fraction of spiked chips per wafer; once ≥5 wafers
observed, z-scores the latest rate against the mean/std of *all previous* rates
(`arr[:-1]`, a leave-one-out comparison, **not** a frozen baseline like the other
trend detectors), alarms above `wafer_spike_rate_sigma` (default 3.0).

```python
from hbm_anomaly_detection.detector.spike_detector import detect_chip_spikes
events, spike_rate = detect_chip_spikes(wafer_features, group_key)
# events: list[AnomalyEvent], spike_rate: fraction of chips flagged
```

## 2. Inter-/intra-wafer drift on the resistance channel

`detector/inter_wafer.py` and `detector/intra_wafer.py` are **generic** across
channels (unlike `temp_anomaly`'s single-channel copies — see `temp-detection-models`
§1-2 for the full algorithm description, which applies identically here). The only
difference for resistance: the joint pipeline calls them with `channel="resist"`
(`inter_wafer.py`'s `update_and_detect(..., channel="resist")`,
`intra_wafer.py`'s `detect_intra_wafer_drift(..., channel="resist")`), and the input is
`resist_roughness` instead of `temp_roughness`. `AnomalyEvent.channel` is set to
`"resist"` accordingly, distinguishing these events from the temperature-side ones
emitted by the same tracker classes.

```python
from hbm_anomaly_detection.detector.inter_wafer import InterWaferTracker
from hbm_anomaly_detection.config import InterWaferConfig

tracker = InterWaferTracker(InterWaferConfig(baseline_wafers=5, alert_sigma=2.5))
events = tracker.update_and_detect(group_key, wafer_id, resist_roughness_mean, channel="resist")
```

## 3. Coupling breakdown (joint, resistance-driven) — `detector/coupling_detector.py`

Only applicable to bimodal equipment. `CouplingDetector.process_wafer` (line 83)
re-clusters chips by resistance level every wafer (`_kmeans2_resist`, line 33) —
returns `[]` immediately if `separation <= bimodal_separation_threshold` (i.e.
resistance isn't bimodal this wafer). For bimodal wafers, it then computes each
cluster's **mean temperature profile** and their L2 distance
(`_mean_profile`/`_profile_distance`, lines 19-30) — the actual anomaly signal is a
temperature comparison, but resistance is what defines the two groups being compared.
That distance feeds an EWMA-then-frozen-baseline chart (`_update_and_detect`, line 125,
`coupling_sigma` default 3.0), identical in structure to `InterWaferTracker`. Full
mechanics also documented from the temperature side in `temp-detection-models` §3.

## 4. Deep/joint signature (autoencoder) — `detector/autoencoder.py`

Hand-rolled NumPy MLP autoencoder (`SimpleAutoencoder`: `input → tanh(hidden) →
tanh(bottleneck) → tanh(hidden) → linear(input)`, full-batch hand-rolled Adam, no
torch/sklearn). Resistance's contribution: `extract_residual_vector` (see
`resi-data-preprocessing` §4) puts the downsampled resist residual in the **second
half** of the `2*input_length`-dim input vector (temp residual is the first half).
`AutoencoderConfig` (`config.py:67`): `input_length=25` per channel (so input dim 50
total), `hidden_dim=16`, `bottleneck_dim=4`, `baseline_wafers=8`,
`threshold_percentile=99.0`.

Per-group lifecycle (`AutoencoderAnomalyDetector.process_wafer`, line 185): buffer
residual vectors until `baseline_wafers` wafers seen, then `_fit` (line 245) —
**independently std-normalizes** the temp half and resist half before training
(`_normalize`, line 267; they have very different natural magnitudes), fits the MLP,
sets `threshold = percentile(train_reconstruction_errors, threshold_percentile)` (this
threshold is **data-derived**, not theoretical). After fitting, every chip is scored by
reconstruction error; `severity = err / threshold`.

Reproduce training standalone:

```python
import numpy as np
from hbm_anomaly_detection.detector.autoencoder import SimpleAutoencoder

rng = np.random.default_rng(0)
X = rng.normal(0, 1, size=(50, 10))  # stand-in for 50 chips' [temp|resist] residual vectors
ae = SimpleAutoencoder(input_dim=10, hidden_dim=8, bottleneck_dim=2, seed=0)
ae.fit(X, epochs=200, lr=0.05)
```

## 5. Inter-wafer drift, joint (Hotelling's T²) — `detector/multivariate_inter_wafer.py`

2-D MEWMA over `[temp_roughness_mean, resist_roughness_mean]`. Resistance is one of
the two axes of the joint vector; `T² = diffᵀ · Σ⁻¹ · diff` against the frozen
baseline covariance catches a shift in resist roughness alone, temp roughness alone,
**or** a shift in the correlation between the two that neither channel's independent
`InterWaferTracker` (§2 above) would flag (module docstring, lines 6-13). Alarm when
`T² > chi2.ppf(1 - alert_alpha, df=2)` (`alert_alpha` default 0.01, theoretical, no
data calibration). Full mechanics also documented in `temp-detection-models` §3.

## 6. Alternative model — `rmt_profile_detector/` (Marchenko-Pastur PCA, resist channel)

`MPProfileDetector` (`mp_detector.py`) scores temp and resist **independently** (unlike
the autoencoder, which scores them jointly) using a PCA + Marchenko-Pastur bulk-edge
model (`base_profile.py`) — see `theory recap` docstring there (lines 1-28). For the
resistance channel: `BaseProfile.fit(X_resist)` eigendecomposes the resist residual
covariance, partitions eigenvalues into "signal" (> Marchenko-Pastur bulk edge `λ+`)
and "noise" subspaces, and `score(x)` is the noise-subspace chi-square statistic —
`chi2.ppf(1-alert_alpha, df=k_noise)` is a **theoretical** threshold requiring no
percentile calibration (contrast with the autoencoder's `threshold_percentile`).
`anomaly_type = "mp_anomaly_resist"`, emitted per chip. This package imports
`ChipData`/`GroupKey`/`AnomalyEvent` directly from `hbm_anomaly_detection`
(`mp_detector.py:28`) — it's meant to slot in as an *additional* detector, not replace
the autoencoder.

## Config — `hbm_anomaly_detection/config.py`

Resistance-relevant sub-configs of `AnomalyConfig` (line 84): `SpikeConfig` (line 41,
`residual_sigma=5.0`, `wafer_spike_rate_sigma=3.0`), `CouplingConfig` (line 48,
`coupling_sigma=3.0`, `baseline_wafers=10`, `ewma_lambda=0.2`), `ReferenceConfig`'s
bimodal fields (`bimodal_min_chips=10`, `bimodal_separation_threshold=0.15`), plus
`AutoencoderConfig` and `MultivariateInterWaferConfig` (shared with temperature).

## Adding a new resistance detector

Same pattern as `temp-detection-models`' guidance: a `@dataclass State`, a
`@dataclass Tracker`/`Detector` with `_states: dict[GroupKey, State]`, and
`update_and_detect`/`process_wafer` returning `list[AnomalyEvent]` with
`channel="resist"` (or `"both"` if it also needs temperature). Add a config dataclass
to `config.py`, wire it into `AnomalyDetectionPipeline.process_wafer` (see
`resi-inference-pipeline`), add a synthetic scenario to `simulator.py`, and a
`TestYourDetector` class to `tests/test_pipeline.py`.

## Reproducibility checklist

- Detector thresholds come from `AnomalyConfig` sub-configs — construct them
  explicitly (see `tests/test_pipeline.py::_test_config()` for a complete example).
- Spike detection's per-chip threshold is computed **per wafer** from that wafer's own
  spike-value distribution — it is not a cross-wafer baseline, so don't expect it to
  need a warm-up period the way inter-wafer/coupling/autoencoder do.
- The autoencoder and RMT model need `seed` fixed for reproducible weights/output.
- Validate: `python -m pytest hbm_anomaly_detection/tests/test_pipeline.py -k
  "TestSpikeDetection or TestCouplingBreakdown or TestMultivariateInterWaferDrift or
  TestSimpleAutoencoder or TestDeepAutoencoderDetector" -v`.
