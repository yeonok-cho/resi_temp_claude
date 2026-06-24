"""
Standalone visual demo of the two temp-drift detectors used in
temp_anomaly/temp_vibration_detection:
  1. Intra-wafer drift  -> CUSUM on z-scored roughness within one wafer
  2. Inter-wafer drift  -> EWMA control chart on wafer-mean roughness across many wafers

This mirrors the real formulas/config defaults in:
  temp_anomaly/temp_vibration_detection/detector/intra_wafer.py
  temp_anomaly/temp_vibration_detection/detector/inter_wafer.py
  temp_anomaly/temp_vibration_detection/config.py
but uses synthetic data so it's self-contained and easy to read.

Run: python temp_anomaly/examples/drift_demo.py
Outputs intra_wafer_demo.png / inter_wafer_demo.png next to this file.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(42)

# ── Config defaults (same as config.py) ──
CUSUM_K = 0.5
CUSUM_H = 4.0
WARMUP_CHIPS = 10
EWMA_LAMBDA = 0.1
ALERT_SIGMA = 3.0
BASELINE_WAFERS = 10
MIN_CV = 0.15


# ════════════════════════════════════════════════════════
# 1. INTRA-WAFER: CUSUM on one wafer's chip sequence
# ════════════════════════════════════════════════════════
n_chips = 30
roughness = np.concatenate([
    rng.normal(1.0, 0.15, WARMUP_CHIPS),          # chips 0-9: normal warmup
    rng.normal(1.0, 0.15, 10) + np.linspace(0, 0.5, 10),  # chips 10-19: drift starts
    rng.normal(1.0, 0.15, 10) + 0.5,              # chips 20-29: stays elevated
])

warmup_mean = roughness[:WARMUP_CHIPS].mean()
warmup_std = roughness[:WARMUP_CHIPS].std(ddof=1)
effective_std = max(warmup_std, warmup_mean * MIN_CV, 1e-9)

z = (roughness - warmup_mean) / effective_std
detection_z = z[WARMUP_CHIPS:]

S = np.empty(len(detection_z))
s = 0.0
alarm_idx = -1
for i, x in enumerate(detection_z):
    s = max(0.0, s + x - CUSUM_K)
    S[i] = s
    if alarm_idx < 0 and s > CUSUM_H:
        alarm_idx = i

abs_alarm_idx = WARMUP_CHIPS + alarm_idx if alarm_idx >= 0 else None
severity_intra = S.max()

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax = axes[0]
ax.axvspan(-0.5, WARMUP_CHIPS - 0.5, color="grey", alpha=0.15, label="warmup (baseline)")
ax.plot(range(n_chips), roughness, "o-", color="tab:blue", label="roughness per chip")
ax.axhline(warmup_mean, color="grey", ls="--", lw=1, label=f"warmup mean={warmup_mean:.2f}")
if abs_alarm_idx is not None:
    ax.axvline(abs_alarm_idx, color="tab:red", ls=":", lw=2)
    ax.scatter([abs_alarm_idx], [roughness[abs_alarm_idx]], color="tab:red", zorder=5, s=80,
               label=f"alarm @ chip {abs_alarm_idx}")
ax.set_ylabel("temp roughness")
ax.set_title("Intra-wafer: roughness per chip (bond order)")
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
x_axis = range(WARMUP_CHIPS, n_chips)
ax.plot(x_axis, S, "o-", color="tab:orange", label="CUSUM S")
ax.axhline(CUSUM_H, color="tab:red", ls="--", label=f"threshold h={CUSUM_H}")
if abs_alarm_idx is not None:
    ax.axvline(abs_alarm_idx, color="tab:red", ls=":", lw=2)
    ax.scatter([abs_alarm_idx], [S[alarm_idx]], color="tab:red", zorder=5, s=80,
               label=f"severity={severity_intra:.2f}")
ax.set_xlabel("chip index (bond order)")
ax.set_ylabel("CUSUM value")
ax.set_title("Intra-wafer: CUSUM chart -> alarm when S crosses h")
ax.legend(loc="upper left", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "intra_wafer_demo.png"), dpi=130)
print(f"[intra] alarm_idx={abs_alarm_idx}, severity(CUSUM peak)={severity_intra:.3f}")


# ════════════════════════════════════════════════════════
# 2. INTER-WAFER: EWMA control chart across many wafers
# ════════════════════════════════════════════════════════
n_wafers = 40
wafer_mean_roughness = np.concatenate([
    rng.normal(2.0, 0.3, BASELINE_WAFERS),                       # wafers 0-9: baseline
    rng.normal(2.0, 0.3, n_wafers - BASELINE_WAFERS) + np.linspace(0, 1.5, n_wafers - BASELINE_WAFERS),  # slow drift
])

ewma = np.empty(n_wafers)
baseline_mean = 0.0
m2 = 0.0
n_samples = 0
baseline_std = 1.0
frozen = False
control_limit = np.full(n_wafers, np.nan)
alarm_wafer = None
severity_inter = None

for i, x in enumerate(wafer_mean_roughness):
    ewma[i] = x if i == 0 else (1 - EWMA_LAMBDA) * ewma[i - 1] + EWMA_LAMBDA * x

    if not frozen:
        n_samples += 1
        delta = x - baseline_mean
        baseline_mean += delta / n_samples
        m2 += delta * (x - baseline_mean)
        if n_samples >= 2:
            baseline_std = max(np.sqrt(m2 / (n_samples - 1)), 1e-9)
        if n_samples >= BASELINE_WAFERS:
            frozen = True

    if frozen:
        cl = baseline_mean + ALERT_SIGMA * baseline_std
        control_limit[i] = cl
        if alarm_wafer is None and ewma[i] > cl:
            alarm_wafer = i
            severity_inter = (ewma[i] - baseline_mean) / baseline_std

plt.figure(figsize=(9, 5))
plt.axvspan(-0.5, BASELINE_WAFERS - 0.5, color="grey", alpha=0.15, label="baseline window (frozen after this)")
plt.scatter(range(n_wafers), wafer_mean_roughness, color="tab:blue", alpha=0.5, label="wafer mean roughness (raw)")
plt.plot(range(n_wafers), ewma, color="tab:orange", lw=2, label=f"EWMA (λ={EWMA_LAMBDA})")
plt.axhline(baseline_mean, color="grey", ls="--", lw=1, label=f"baseline mean={baseline_mean:.2f}")
plt.plot(range(n_wafers), control_limit, color="tab:red", ls="--", lw=1.5,
         label=f"control limit = mean+{ALERT_SIGMA}σ")
if alarm_wafer is not None:
    plt.axvline(alarm_wafer, color="tab:red", ls=":", lw=2)
    plt.scatter([alarm_wafer], [ewma[alarm_wafer]], color="tab:red", zorder=5, s=100,
                label=f"alarm @ wafer {alarm_wafer}, severity={severity_inter:.2f}σ")
plt.xlabel("wafer index (track-out order)")
plt.ylabel("mean temp roughness")
plt.title("Inter-wafer: EWMA control chart across wafers")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "inter_wafer_demo.png"), dpi=130)
print(f"[inter] alarm_wafer={alarm_wafer}, severity(sigma)={severity_inter:.3f}")
