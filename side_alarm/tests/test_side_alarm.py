"""
Validation cases A-G from the RESI/TEMP side-alarm specification (§28).

The synthetic frames here mimic the shape of the production table: one row per
chip, carrying ``mp_score`` (RESI anomaly score) and ``temp_roughness``, keyed
by eqp_id / product / module_id / wafer_id / x / y / event_tmstp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..config import SideAlarmConfig, BaselineConfig, LEFT, RIGHT
from ..data_models import (
    ALARM, WARNING, CHECK, NORMAL,
    RESI_EVENT_MODULE_FAILURE, RESI_EVENT_PROCESS_SHIFT,
    TEMP_EVENT_WAFER, TEMP_EVENT_TREND,
    SEV_CRITICAL, SEV_SEVERE_CHECK,
    ResiResult, TempResult,
)
from ..baseline import build_resi_baseline, build_temp_baseline
from ..detect_resi import calculate_resi_stats, run_resi_detector
from ..detect_temp import calculate_temp_stats, run_temp_detector
from ..fuse_alarm import fuse_alarm
from ..loader import prepare_chips

EQP = "APW19"
PRODUCT = "P001"
N_CHIPS = 50
BASE_MEAN = 1.0
BASE_STD = 0.08
T0 = pd.Timestamp("2026-03-01 00:00:00")


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def exact_std(n: int, mean: float, std: float, seed: int) -> np.ndarray:
    """Normal-shaped sample rescaled to exactly the requested mean and std."""
    v = np.random.default_rng(seed).normal(size=n)
    v = (v - v.mean()) / v.std(ddof=1)
    return mean + std * v


def chip_rows(wafer: str, side: str, scores, roughness, when: pd.Timestamp) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    roughness = np.asarray(roughness, dtype=float)
    n = len(scores)
    return pd.DataFrame({
        "eqp_id": EQP,
        "product": PRODUCT,
        "module_id": side.lower(),
        "wafer_id": wafer,
        "x": np.arange(n) % 10,
        "y": np.arange(n) // 10,
        "event_tmstp": when,
        "mp_score": scores,
        "temp_roughness": roughness,
    })


def quantized_roughness(n: int, seed: int, levels=(0.03, 0.04, 0.05)) -> np.ndarray:
    """2026-style TEMP data: roughness lands on a handful of discrete levels."""
    return np.random.default_rng(seed).choice(levels, size=n)


def healthy_history(n_wafers: int = 12) -> pd.DataFrame:
    """Healthy wafers used to build the equipment-side baselines."""
    frames = []
    for i in range(n_wafers):
        when = T0 + pd.Timedelta(hours=i)
        for j, side in enumerate((LEFT, RIGHT)):
            frames.append(chip_rows(
                f"H{i:03d}", side,
                exact_std(N_CHIPS, BASE_MEAN, BASE_STD, seed=100 + i * 2 + j),
                quantized_roughness(N_CHIPS, seed=200 + i * 2 + j),
                when,
            ))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def config() -> SideAlarmConfig:
    cfg = SideAlarmConfig()
    cfg.baseline = BaselineConfig(min_baseline_wafers=10, min_baseline_chips=200)
    return cfg


@pytest.fixture(scope="module")
def baselines(config):
    chips = prepare_chips(healthy_history(), config)
    resi_stats = calculate_resi_stats(chips, None, config)
    temp_stats = calculate_temp_stats(chips, None, config)
    resi_bl = build_resi_baseline(chips, resi_stats, config)
    temp_bl = build_temp_baseline(chips, temp_stats, config)
    assert resi_bl["BASELINE_OK"].all()
    assert temp_bl["BASELINE_OK"].all()
    return resi_bl, temp_bl


# ---------------------------------------------------------------------------
# Case A — confirmed RESI heater failure
# ---------------------------------------------------------------------------

def test_case_a_resi_heater_failure(config, baselines):
    resi_bl, _ = baselines
    when = T0 + pd.Timedelta(days=1)
    wafer = pd.concat([
        chip_rows("GYK012-22", LEFT,
                  exact_std(N_CHIPS, BASE_MEAN, 0.078, seed=11),
                  quantized_roughness(N_CHIPS, seed=21), when),
        chip_rows("GYK012-22", RIGHT,
                  exact_std(N_CHIPS, BASE_MEAN, 0.445, seed=12),
                  quantized_roughness(N_CHIPS, seed=22), when),
    ], ignore_index=True)

    result = run_resi_detector(prepare_chips(wafer, config), resi_bl, config).iloc[0]

    assert result["RESI_SUSPECT_SIDE"] == RIGHT
    assert result["RESI_STD_RATIO"] == pytest.approx(0.445 / 0.078, rel=1e-3)
    assert result["SUSPECT_RESI_9SD"] >= 1
    assert result["RESI_LEVEL"] == ALARM
    assert result["RESI_EVENT"] == RESI_EVENT_MODULE_FAILURE
    assert "std ratio" in result["RESI_REASON"]


# ---------------------------------------------------------------------------
# Case B — process shift, NOT module failure
# ---------------------------------------------------------------------------

def test_case_b_process_shift(config, baselines):
    resi_bl, _ = baselines
    when = T0 + pd.Timedelta(days=1)
    wafer = pd.concat([
        chip_rows("PS-001", LEFT,
                  exact_std(N_CHIPS, 1.5, 0.06, seed=31),
                  quantized_roughness(N_CHIPS, seed=41), when),
        chip_rows("PS-001", RIGHT,
                  exact_std(N_CHIPS, 1.5, 0.05, seed=32),
                  quantized_roughness(N_CHIPS, seed=42), when),
    ], ignore_index=True)

    result = run_resi_detector(prepare_chips(wafer, config), resi_bl, config).iloc[0]

    assert result["RESI_STD_RATIO"] == pytest.approx(0.06 / 0.05, rel=1e-3)
    assert result["SUSPECT_RESI_9SD"] == 0
    assert result["RESI_EVENT"] == RESI_EVENT_PROCESS_SHIFT
    assert result["RESI_EVENT"] != RESI_EVENT_MODULE_FAILURE
    assert result["RESI_LEVEL"] != ALARM


# ---------------------------------------------------------------------------
# TEMP sequence helper
# ---------------------------------------------------------------------------

def temp_wafer(wafer: str, when: pd.Timestamp, abnormal_side: str | None, seed: int) -> pd.DataFrame:
    """One wafer; ``abnormal_side`` gets a population of high-roughness chips."""
    frames = []
    for j, side in enumerate((LEFT, RIGHT)):
        if side == abnormal_side:
            # Clear of the 1.5x mean-ratio threshold: a marginal population
            # would make the trend run depend on the healthy side's noise.
            rough = np.concatenate([
                np.full(30, 0.10),
                np.full(N_CHIPS - 30, 0.04),
            ])
        else:
            rough = quantized_roughness(N_CHIPS, seed=seed + j)
        frames.append(chip_rows(
            wafer, side,
            exact_std(N_CHIPS, BASE_MEAN, BASE_STD, seed=seed + 50 + j),
            rough, when,
        ))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Case C — single abnormal wafer stays a wafer event
# ---------------------------------------------------------------------------

def test_case_c_temp_single_wafer_event(config, baselines):
    _, temp_bl = baselines
    base = T0 + pd.Timedelta(days=2)
    frames = [
        temp_wafer("W0", base, RIGHT, seed=300),
        temp_wafer("W1", base + pd.Timedelta(hours=1), None, seed=310),
        temp_wafer("W2", base + pd.Timedelta(hours=2), None, seed=320),
    ]
    chips = prepare_chips(pd.concat(frames, ignore_index=True), config)
    out = run_temp_detector(chips, temp_bl, config).set_index("WAFER_ID")

    assert out.loc["W0", "TEMP_SUSPECT"]
    assert out.loc["W0", "TEMP_SUSPECT_SIDE"] == RIGHT
    assert out.loc["W0", "TEMP_EVENT"] == TEMP_EVENT_WAFER
    assert out.loc["W0", "TEMP_LEVEL"] == CHECK
    assert out.loc["W0", "TEMP_EVENT"] != TEMP_EVENT_TREND
    assert not out.loc["W1", "TEMP_SUSPECT"]
    assert out.loc["W2", "TEMP_LEVEL"] == NORMAL


# ---------------------------------------------------------------------------
# Case D — K consecutive same-side wafers become an equipment trend
# ---------------------------------------------------------------------------

def test_case_d_temp_equipment_trend(config, baselines):
    _, temp_bl = baselines
    base = T0 + pd.Timedelta(days=3)
    k = config.temp.trend_k
    frames = [
        temp_wafer(f"T{i}", base + pd.Timedelta(hours=i), RIGHT, seed=400 + i * 10)
        for i in range(k)
    ]
    chips = prepare_chips(pd.concat(frames, ignore_index=True), config)
    out = run_temp_detector(chips, temp_bl, config).set_index("WAFER_ID")

    last = out.loc[f"T{k - 1}"]
    assert last["TEMP_TREND_COUNT"] == k
    assert last["TEMP_EVENT"] == TEMP_EVENT_TREND
    assert last["TEMP_LEVEL"] == ALARM
    assert last["TEMP_SUSPECT_SIDE"] == RIGHT
    # The first wafer of the run must still read as a single-wafer event.
    assert out.loc["T0", "TEMP_EVENT"] == TEMP_EVENT_WAFER


# ---------------------------------------------------------------------------
# Case E / F — fusion of same-side and opposite-side abnormality
# ---------------------------------------------------------------------------

def _resi(level, side):
    return ResiResult(eqp_id=EQP, product=PRODUCT, wafer_id="F1",
                      level=level, event=RESI_EVENT_MODULE_FAILURE, suspect_side=side)


def _temp(level, side, event):
    return TempResult(eqp_id=EQP, product=PRODUCT, wafer_id="F1",
                      level=level, event=event, suspect_side=side, trend_count=3)


def test_case_e_same_side_sustained_is_critical():
    fused = fuse_alarm(_resi(ALARM, RIGHT), _temp(ALARM, RIGHT, TEMP_EVENT_TREND))
    assert fused.final_severity == SEV_CRITICAL
    assert fused.temp_supports_resi is True
    assert fused.cross_signal_direction_match is True


def test_case_f_opposite_side_is_not_critical():
    fused = fuse_alarm(_resi(ALARM, RIGHT), _temp(ALARM, LEFT, TEMP_EVENT_TREND))
    assert fused.final_severity != SEV_CRITICAL
    assert fused.final_severity == SEV_SEVERE_CHECK
    assert fused.temp_supports_resi is False
    assert fused.cross_signal_direction_match is False
    assert "disagree" in fused.final_reason


def test_temp_normal_never_downgrades_resi_alarm():
    """Spec §16, §17.2 — the fusion layer must not cancel a RESI alarm."""
    fused = fuse_alarm(_resi(ALARM, RIGHT), _temp(NORMAL, None, NORMAL))
    assert fused.final_severity == "RESI_ALARM"


def test_temp_alarm_visible_without_resi():
    """Spec §17.3 — TEMP stays independently visible to operators."""
    fused = fuse_alarm(
        ResiResult(eqp_id=EQP, product=PRODUCT, wafer_id="F1", level=NORMAL, suspect_side=None),
        _temp(ALARM, LEFT, TEMP_EVENT_TREND),
    )
    assert fused.final_severity == "TEMP_ALARM"


# ---------------------------------------------------------------------------
# Case G — quantized TEMP must not divide by zero or alarm on an infinite ratio
# ---------------------------------------------------------------------------

def test_case_g_quantized_temp_no_infinite_ratio(config, baselines):
    _, temp_bl = baselines
    when = T0 + pd.Timedelta(days=4)
    flat = np.full(N_CHIPS, 0.02)
    wafer = pd.concat([
        chip_rows("Q-001", LEFT, exact_std(N_CHIPS, BASE_MEAN, BASE_STD, seed=61), flat, when),
        chip_rows("Q-001", RIGHT, exact_std(N_CHIPS, BASE_MEAN, BASE_STD, seed=62), flat, when),
    ], ignore_index=True)

    out = run_temp_detector(prepare_chips(wafer, config), temp_bl, config).iloc[0]

    assert out["SUSPECT_TEMP_STD"] == 0.0
    assert np.isfinite(out["TEMP_STD_RATIO"])
    assert out["TEMP_HIGH_SHARE_RATIO"] == 0.0
    assert not out["TEMP_SUSPECT"]
    assert out["TEMP_LEVEL"] == NORMAL
    assert bool(temp_bl["TEMP_QUANTIZED"].all())


def test_quantization_diagnostic_flags_the_equipment(baselines):
    """The baseline must mark this quantized equipment so the legacy
    std-ratio path stays disabled (spec §13, §14)."""
    _, temp_bl = baselines
    assert temp_bl["TEMP_UNIQUE_COUNT"].max() <= 12
    assert bool(temp_bl["TEMP_QUANTIZED"].all())


# ---------------------------------------------------------------------------
# Guards from §26 / §27
# ---------------------------------------------------------------------------

def test_missing_side_marks_comparison_unavailable(config, baselines):
    resi_bl, _ = baselines
    when = T0 + pd.Timedelta(days=5)
    wafer = chip_rows("ONE-SIDE", LEFT,
                      exact_std(N_CHIPS, BASE_MEAN, BASE_STD, seed=71),
                      quantized_roughness(N_CHIPS, seed=72), when)

    out = run_resi_detector(prepare_chips(wafer, config), resi_bl, config).iloc[0]

    assert not out["SIDE_COMPARISON_AVAILABLE"]
    assert out["RESI_SUSPECT_SIDE"] == LEFT
    assert pd.isna(out["OTHER_RESI_STD"])
    assert out["RESI_EVENT"] != RESI_EVENT_PROCESS_SHIFT


def test_small_chip_population_is_not_silently_normal(config, baselines):
    resi_bl, _ = baselines
    when = T0 + pd.Timedelta(days=6)
    wafer = pd.concat([
        chip_rows("TINY", LEFT, exact_std(4, BASE_MEAN, BASE_STD, seed=81),
                  quantized_roughness(4, seed=82), when),
        chip_rows("TINY", RIGHT, exact_std(4, BASE_MEAN, 0.5, seed=83),
                  quantized_roughness(4, seed=84), when),
    ], ignore_index=True)

    out = run_resi_detector(prepare_chips(wafer, config), resi_bl, config).iloc[0]

    assert out["RESI_LEVEL"] == "INSUFFICIENT_DATA"
    assert out["RESI_LEVEL"] != NORMAL
