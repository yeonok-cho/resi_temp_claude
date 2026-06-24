"""
Visual walkthrough of how "temp roughness" is computed
(temp_vibration_detection/feature_extractor.py:compute_roughness),
step by step, comparing a normal chip vs. a vibrating chip:

  1. raw temp_series vs. reference profile
  2. residual = temp_series - reference
  3. diff = first-order difference of the residual
  4. roughness = std(diff)   <- final scalar feature

Run: python temp_anomaly/examples/roughness_demo.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# ── Shared "ground truth" thermal profile (smooth bonding curve) ──
t = np.arange(100)
reference = 25 + 40 / (1 + np.exp(-(t - 30) / 5)) - 10 / (1 + np.exp(-(t - 80) / 5))

# ── Chip A: normal — follows the reference with only slow, smooth noise ──
slow_noise = 0.6 * np.sin(t / 30) + rng.normal(0, 0.15, len(t))
temp_a = reference + slow_noise

# ── Chip B: vibrating — same slow noise + added high-frequency oscillation ──
vibration = 1.2 * np.sin(t / 2.0)
temp_b = reference + slow_noise + vibration


def compute_roughness(residual: np.ndarray) -> float:
    if len(residual) < 2:
        return 0.0
    diff = np.diff(residual)
    return float(np.std(diff)), diff


fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)

for col, (label, temp_series, color) in enumerate([
    ("Chip A (normal)", temp_a, "tab:blue"),
    ("Chip B (vibrating)", temp_b, "tab:red"),
]):
    residual = temp_series - reference
    roughness, diff = compute_roughness(residual)

    ax = axes[0, col]
    ax.plot(t, reference, "--", color="grey", label="reference profile")
    ax.plot(t, temp_series, color=color, label="temp_series (raw)")
    ax.set_title(f"{label}\n1) raw signal vs. reference")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1, col]
    ax.plot(t, residual, color=color)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("2) residual = temp_series - reference")

    ax = axes[2, col]
    ax.plot(t[1:], diff, color=color)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("3) diff = residual[i+1] - residual[i]")

    ax = axes[3, col]
    ax.bar(["roughness = std(diff)"], [roughness], color=color)
    ax.set_ylim(0, 1.5)
    ax.text(0, roughness + 0.05, f"{roughness:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("4) final feature")

axes[3, 0].set_ylabel("roughness")

fig.tight_layout()
out_path = os.path.join(OUT_DIR, "roughness_demo.png")
fig.savefig(out_path, dpi=130)
print(f"saved {out_path}")
