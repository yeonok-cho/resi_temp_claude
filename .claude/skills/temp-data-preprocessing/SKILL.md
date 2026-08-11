---
name: temp-data-preprocessing
description: How this repo turns raw temperature time series (real ClickHouse temp_raw rows or synthetic test data) into ChipData/WaferData objects, a temperature reference profile, and roughness features — the temp_anomaly/temp_vibration_detection package. Use this whenever the user asks to decode/ingest the temp_raw column, build ChipData/WaferData from a DataFrame for temperature-only detection, generate synthetic temperature wafers for testing, build or update the temperature reference profile (EWMA), extract temperature roughness features, or debug why a wafer's temperature features/reference look wrong. Also use it before touching temp_anomaly/temp_vibration_detection/feature_extractor.py, reference_manager.py, simulator.py, or temp_anomaly/scripts/_common.py. For the resistance channel, use resi-data-preprocessing instead.
---

# Temperature-channel data preprocessing

`temp_anomaly/temp_vibration_detection/` is a deliberately **temperature-only** subset
of the full HBM CoW anomaly system — its `config.py` docstring says so explicitly
(lines 4-6): "resistance, bimodal clustering, spike, and coupling detection are
intentionally excluded." This is the package actually deployed to production against
ClickHouse (see `temp-inference-pipeline`). If you need the resistance channel or the
joint temp+resist detectors, see `resi-data-preprocessing` / `resi-detection-models`
(they live in `hbm_anomaly_detection/`, the fuller sibling package).

## 1. Data model — `temp_anomaly/temp_vibration_detection/data_models.py`

- `GroupKey(equipment_id, recipe, head)` (line 15) — same shape as the joint package;
  every reference profile and EWMA tracker is keyed by this tuple.
- `ChipData` (line 22) — **only** `temp_series: np.ndarray`; there is no
  `resist_series` field at all in this package (contrast with
  `hbm_anomaly_detection.data_models.ChipData`, which has both).
- `WaferData` (line 44), `ChipFeatures` (line 60, just `temp_roughness`),
  `WaferFeatures` (line 67, `roughness_sequence()` / `summary_stats()` with no
  `channel` argument since there's only one channel).
- `AnomalyEvent` (line 87) — `anomaly_type` is `"intra_drift"` or `"inter_drift"` only,
  and there is **no `channel` field** (unlike the joint package's `AnomalyEvent`,
  which needs `channel` to disambiguate temp/resist/both).

```python
import numpy as np
from temp_vibration_detection.data_models import ChipData, WaferData

chip = ChipData(
    equipment_id="EQ01", recipe="R001", head="H1", wafer_id="W001",
    x=3, y=5, bond_order=0,
    temp_series=np.linspace(25, 300, 500),
)
wafer = WaferData(wafer_id="W001", chips=[chip])
```

## 2. Two ways temperature data enters the pipeline

### 2a. Synthetic — `temp_anomaly/temp_vibration_detection/simulator.py`

Same shape as the joint package's simulator but temperature-only (no resist profile
generators, no bimodal/spike wafer generators):

| Function | Line | Produces |
|---|---|---|
| `_temp_profile(n_points, ...)` | 19 | Ramp-up → hold → cool-down base curve (`t_start=25`→`t_peak=300`) |
| `make_normal_chip(...)` | 48 | One clean chip: base profile + Gaussian noise (`noise_std_temp=0.5`) |
| `make_vibration_chip(..., noise_multiplier)` | 78 | Elevated-noise chip (multiplies `noise_std_temp`) |
| `make_normal_wafer(...)` | 120 | Clean wafer, `n_chips` on a square grid (`_make_chip_grid`, line 113) |
| `make_intra_vibration_wafer(onset_fraction, max_noise_multiplier)` | 154 | Noise ramps up progressively in the wafer's back half — for `intra_wafer` detector |
| `make_normal_wafer_sequence(n_wafers, seed)` | 206 | List of normal wafers, for baseline warm-up |
| `make_inter_wafer_drift_sequence(n_normal, n_drift, drift_multiplier_final)` | 233 | Normal wafers, then wafers with gradually increasing temp noise — for `inter_wafer` detector |

Always pass an explicit `rng`/`seed` for reproducibility (every generator accepts one).

```python
from temp_vibration_detection.simulator import make_normal_wafer_sequence
baseline = make_normal_wafer_sequence(n_wafers=15, n_chips=100, n_points=500, seed=42)
```

### 2b. Real data — `temp_anomaly/scripts/_common.py`

The only real-data ingestion path in this repo, and it only handles temperature:

- Table: `ds_catalog.aifpa_cow_rule_hist`, columns `eqp_id, product, module_id,
  wafer_id, x, y, temp_raw, event_tmstp` (`query_window`, line 85-98). **There is no
  `resist_raw` column read anywhere in this repo** — see `resi-data-preprocessing` for
  what building that would require.
- `decode_hex_to_float32(hex_str)` (line 24) — the `temp_raw` column is a hex string of
  packed little-endian float32 samples; decoded via `struct.unpack('<f', ...)` in
  4-byte chunks. Returns `np.array([])` on empty/bad input.
- `create_chip(row, bond_order)` (line 38) — maps `eqp_id→equipment_id`,
  `product→recipe`, `module_id→head` (fallback `row.get('head', 'unknown')`).
- `df_to_wafers(df)` (line 51) — groups by `(eqp_id, product, module_id, wafer_id)`
  (not just `wafer_id`, to avoid collisions across equipment), sorts chips within a
  wafer by `event_tmstp` (= bond order), sorts wafers within a group by first
  timestamp. Chips with zero decoded samples are dropped.
- `query_baseline_wafers(...)` in `scripts/run_temp_vibration_detection.py` (line 169)
  and the near-identical inline copy in `scripts/build_baseline.py` (line 107) do the
  same thing with optional `eqp_id`/`recipe`/`head` filters — **`_common.py` is the
  canonical version; the others are inline duplicates (technical debt), so prefer
  extending `_common.py` if you're adding ingestion logic.**

## 3. Reference profile — `temp_anomaly/temp_vibration_detection/reference_manager.py`

Deliberately simpler than the joint package's `ReferenceManager` — **no bimodal
clustering** (module docstring, lines 4-8, says this is a resistance-specific concern
excluded here).

`ReferenceManager.update(wafer)` (line 76):
1. `_pointwise_median(temps)` (line 32) — per-timepoint median across the wafer's chips.
2. First wafer for a group: reference is hard-set to this median
   (`ref.n_wafers_seen == 0`, line 100). Every subsequent wafer: EWMA blend,
   `ref.temp_ref = (1-α)*ref.temp_ref + α*new_median` (`_ewma_update`, line 40;
   `ReferenceConfig.ewma_alpha` default 0.1).
3. `GroupReference.is_initialized()` (line 28) gates detection — same cold-start rule
   as the joint package (see `temp-inference-pipeline`).

`ReferenceManager.reset(key)` (line 65) discards a group's reference entirely — used
after a PM (Preventive Maintenance) event, since the equipment's physical temperature
profile can shift and old EWMA history would bias the new reference. The **next**
wafer for that group is treated as if it were the group's first ever wafer (hard-set,
no blending).

```python
from temp_vibration_detection.reference_manager import ReferenceManager
from temp_vibration_detection.simulator import make_normal_wafer_sequence
from temp_vibration_detection.data_models import GroupKey

rm = ReferenceManager()
for w in make_normal_wafer_sequence(n_wafers=20, n_chips=30, n_points=500, seed=7):
    rm.update(w)

ref = rm.get(GroupKey("EQ01", "R001", "H1"))
assert ref.is_initialized()
```

## 4. Feature extraction — `temp_anomaly/temp_vibration_detection/feature_extractor.py`

Only one feature: `extract_chip_features(chip, temp_reference, config)` (line 29)
computes the residual (`chip.temp_series[:ref_len] - temp_reference[:ref_len]`) then
`compute_roughness(residual)` (line 16) = `std(diff(residual))` — same
DC-offset-insensitive, high-frequency-only statistic as the joint package (see
`resi-data-preprocessing` §4 for the shared derivation). **There is no
`compute_spike` in this package** — temperature spike detection isn't implemented
here; only the joint `hbm_anomaly_detection` package computes `temp_spike`
(it's not currently wired into any detector there either — only `resist_spike` feeds
`spike_detector.py`).

```python
from temp_vibration_detection.feature_extractor import extract_chip_features
feat = extract_chip_features(chip, ref.temp_ref)
# feat.temp_roughness
```

## Reproducibility checklist

- Always pass an explicit `rng`/`seed` to simulator functions.
- `ReferenceConfig`/`FeatureConfig` live in `config.py` (lines 12, 18) — override them
  explicitly rather than hardcoding values; `tests/test_pipeline.py` shows working
  examples of tuned configs for fast test runs.
- A group's reference is unusable until its first wafer is processed
  (`ref.is_initialized()`); the pipeline skips detection on that first wafer by design
  (see `temp-inference-pipeline`).
- Validate preprocessing changes: `python -m pytest
  temp_anomaly/temp_vibration_detection/tests/test_pipeline.py -v` works from the repo
  root (pytest adds `temp_anomaly/` to `sys.path` automatically). If you're importing
  `temp_vibration_detection` from a standalone script instead of via pytest, put
  `temp_anomaly/` on `sys.path` first (or run with `cwd=temp_anomaly/`) — the package
  is `temp_vibration_detection`, not `temp_anomaly.temp_vibration_detection`.
