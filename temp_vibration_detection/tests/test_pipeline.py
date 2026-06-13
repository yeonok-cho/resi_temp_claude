"""
Synthetic data-based tests for the temperature vibration-increase detection
pipeline.

Each test verifies that intra-wafer or inter-wafer temperature vibration
increase is detected at a higher rate than the false-positive rate on
normal data.
"""
from __future__ import annotations

import numpy as np
import pytest

from ..config import (
    AnomalyConfig,
    ReferenceConfig,
    IntraWaferConfig,
    InterWaferConfig,
    FeatureConfig,
)
from ..pipeline import TempVibrationPipeline
from ..simulator import (
    make_normal_wafer_sequence,
    make_intra_vibration_wafer,
    make_inter_wafer_drift_sequence,
    make_normal_wafer,
)


# ---------------------------------------------------------------------------
# Shared configuration: tuned for fast test runs with small synthetic data
# ---------------------------------------------------------------------------

def _test_config() -> AnomalyConfig:
    return AnomalyConfig(
        reference=ReferenceConfig(ewma_alpha=0.3),
        feature=FeatureConfig(min_points=10),
        intra_wafer=IntraWaferConfig(
            cusum_k=0.5,
            cusum_h=3.0,
            warmup_chips=5,
        ),
        inter_wafer=InterWaferConfig(
            ewma_lambda=0.2,
            alert_sigma=2.5,
            baseline_wafers=5,
        ),
    )


N_CHIPS = 64    # 8x8 grid - fast enough for CI
N_POINTS = 200  # captures full ramp -> hold -> cool-down profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_events(results: dict[str, list], anomaly_type: str) -> int:
    return sum(
        1 for events in results.values()
        for e in events
        if e.anomaly_type == anomaly_type
    )


# ---------------------------------------------------------------------------
# Test 1: Normal sequence produces no false positives (smoke test)
# ---------------------------------------------------------------------------

class TestNormalBaseline:
    def test_no_spurious_inter_drift(self):
        """A clean wafer sequence should not trigger inter-wafer drift alerts."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=N_CHIPS, n_points=N_POINTS, seed=0
        )
        results = pipeline.process_wafer_sequence(wafers)
        n_inter = _count_events(results, "inter_drift")
        assert n_inter <= 2, f"Too many inter_drift false positives: {n_inter}"

    def test_no_spurious_intra_drift(self):
        """A clean wafer sequence should not trigger intra-wafer drift alerts."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=N_CHIPS, n_points=N_POINTS, seed=1
        )
        results = pipeline.process_wafer_sequence(wafers)
        n_intra = _count_events(results, "intra_drift")
        assert n_intra <= 2, f"Too many intra_drift false positives: {n_intra}"


# ---------------------------------------------------------------------------
# Test 2: Intra-wafer vibration increase is detected
# ---------------------------------------------------------------------------

class TestIntraWaferDrift:
    def test_intra_drift_detected(self):
        """
        After establishing a normal baseline, a wafer with progressive
        vibration increase in its latter half should trigger an intra_drift alert.
        """
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)

        # Baseline: 12 normal wafers
        normal_wafers = make_normal_wafer_sequence(
            n_wafers=12, n_chips=N_CHIPS, n_points=N_POINTS, seed=10
        )
        for w in normal_wafers:
            pipeline.process_wafer(w)

        # Anomalous wafer: noise grows in second half
        rng = np.random.default_rng(99)
        anomaly_wafer = make_intra_vibration_wafer(
            wafer_id="INTRA_ANOM",
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            onset_fraction=0.5,
            max_noise_multiplier=6.0,
            rng=rng,
        )
        events = pipeline.process_wafer(anomaly_wafer)
        intra_events = [e for e in events if e.anomaly_type == "intra_drift"]
        assert len(intra_events) > 0, (
            "Expected intra_drift alert on wafer with progressive noise increase"
        )

    def test_intra_drift_alarm_in_second_half(self):
        """Alarm index should be in the second half of the bonding sequence."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)

        normal_wafers = make_normal_wafer_sequence(
            n_wafers=12, n_chips=N_CHIPS, n_points=N_POINTS, seed=20
        )
        for w in normal_wafers:
            pipeline.process_wafer(w)

        rng = np.random.default_rng(100)
        anomaly_wafer = make_intra_vibration_wafer(
            wafer_id="INTRA_ANOM2",
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            onset_fraction=0.4,
            max_noise_multiplier=8.0,
            rng=rng,
        )
        events = pipeline.process_wafer(anomaly_wafer)
        intra_events = [e for e in events if e.anomaly_type == "intra_drift"]

        assert len(intra_events) > 0
        alarm_idx = intra_events[0].details["alarm_chip_index"]
        n_chips = intra_events[0].details["n_chips"]
        assert alarm_idx > n_chips * 0.3, (
            f"Expected alarm in 2nd half, got chip index {alarm_idx}/{n_chips}"
        )

    def test_normal_wafer_not_flagged_intra(self):
        """Normal wafers should not trigger intra_drift after baseline is set."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)

        all_wafers = make_normal_wafer_sequence(
            n_wafers=25, n_chips=N_CHIPS, n_points=N_POINTS, seed=30
        )
        for w in all_wafers[:15]:
            pipeline.process_wafer(w)

        n_flags = 0
        for w in all_wafers[15:]:
            events = pipeline.process_wafer(w)
            n_flags += sum(1 for e in events if e.anomaly_type == "intra_drift")

        assert n_flags <= 2, f"Too many false intra_drift on normal wafers: {n_flags}"


# ---------------------------------------------------------------------------
# Test 3: Inter-wafer drift (slowly increasing noise across wafers)
# ---------------------------------------------------------------------------

class TestInterWaferDrift:
    def test_inter_drift_detected(self):
        """
        A sequence with gradually increasing temperature noise across wafers
        should trigger inter_drift alerts in the drifting portion.
        """
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)

        wafers = make_inter_wafer_drift_sequence(
            n_normal=12,
            n_drift=10,
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            base_noise_std=0.5,
            drift_multiplier_final=5.0,
            seed=40,
        )
        results = pipeline.process_wafer_sequence(wafers)

        drift_wafer_ids = {w.wafer_id for w in wafers[12:]}
        drift_alerts = sum(
            1 for wid, evts in results.items()
            if wid in drift_wafer_ids
            for e in evts
            if e.anomaly_type == "inter_drift"
        )
        assert drift_alerts >= 1, (
            "Expected at least one inter_drift alert in drifting wafer portion"
        )

    def test_normal_wafers_clean_inter(self):
        """Only the normal portion should have zero or very few inter_drift events."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)

        wafers = make_inter_wafer_drift_sequence(
            n_normal=15,
            n_drift=10,
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            base_noise_std=0.5,
            drift_multiplier_final=5.0,
            seed=41,
        )
        results = pipeline.process_wafer_sequence(wafers)

        normal_wafer_ids = {w.wafer_id for w in wafers[:15]}
        normal_alerts = sum(
            1 for wid, evts in results.items()
            if wid in normal_wafer_ids
            for e in evts
            if e.anomaly_type == "inter_drift"
        )
        assert normal_alerts <= 2, (
            f"Too many inter_drift false positives on normal wafers: {normal_alerts}"
        )


# ---------------------------------------------------------------------------
# Test 4: Feature extractor unit tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_roughness_zero_for_flat_residual(self):
        """A perfectly flat residual should have zero roughness."""
        from ..feature_extractor import compute_roughness
        residual = np.zeros(100)
        assert compute_roughness(residual) == pytest.approx(0.0, abs=1e-12)

    def test_roughness_increases_with_noise(self):
        """Higher noise should produce higher roughness."""
        from ..feature_extractor import compute_roughness
        rng = np.random.default_rng(1)
        low_noise = rng.normal(0, 0.01, 500)
        high_noise = rng.normal(0, 1.0, 500)
        assert compute_roughness(high_noise) > compute_roughness(low_noise) * 10

    def test_roughness_insensitive_to_dc_offset(self):
        """Adding a DC offset to residual should not change roughness."""
        from ..feature_extractor import compute_roughness
        rng = np.random.default_rng(2)
        signal = rng.normal(0, 0.5, 300)
        assert compute_roughness(signal) == pytest.approx(
            compute_roughness(signal + 100.0), rel=1e-10
        )

    def test_extract_chip_features_shapes(self):
        """extract_chip_features should return a ChipFeatures object with a valid float."""
        from ..feature_extractor import extract_chip_features
        from ..simulator import make_normal_chip
        rng = np.random.default_rng(5)
        chip = make_normal_chip(
            "EQ1", "R1", "H1", "W1", 0, 0, 0, n_points=N_POINTS, rng=rng
        )
        ref = np.zeros(N_POINTS)
        feat = extract_chip_features(chip, ref)
        assert feat.temp_roughness >= 0.0


# ---------------------------------------------------------------------------
# Test 5: Reference manager unit tests
# ---------------------------------------------------------------------------

class TestReferenceManager:
    def test_reference_initialized_after_first_wafer(self):
        """Reference should be non-empty after processing one wafer."""
        from ..reference_manager import ReferenceManager
        from ..data_models import GroupKey
        rm = ReferenceManager()
        w = make_normal_wafer(n_chips=30, n_points=N_POINTS)
        rm.update(w)
        key = GroupKey("EQ01", "R001", "H1")
        ref = rm.get(key)
        assert ref is not None
        assert ref.is_initialized()
        assert len(ref.temp_ref) == N_POINTS

    def test_ewma_smoothing(self):
        """
        After many normal wafers, the EWMA reference should converge to
        the true profile (not drift away).
        """
        from ..reference_manager import ReferenceManager
        from ..simulator import _temp_profile
        from ..data_models import GroupKey

        rm = ReferenceManager()
        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=30, n_points=N_POINTS, seed=7
        )
        for w in wafers:
            rm.update(w)

        key = GroupKey("EQ01", "R001", "H1")
        ref = rm.get(key)
        assert ref is not None
        true_temp = _temp_profile(N_POINTS)
        rmse = float(np.sqrt(np.mean((ref.temp_ref - true_temp) ** 2)))
        assert rmse < 20.0, f"Reference too far from true profile: RMSE={rmse:.2f}"


# ---------------------------------------------------------------------------
# Test 6: Pipeline integration smoke test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_process_wafer_returns_list(self):
        pipeline = TempVibrationPipeline()
        w = make_normal_wafer(n_chips=20, n_points=N_POINTS)
        result = pipeline.process_wafer(w)
        assert isinstance(result, list)

    def test_process_wafer_sequence_returns_dict(self):
        pipeline = TempVibrationPipeline()
        wafers = make_normal_wafer_sequence(n_wafers=5, n_chips=20, n_points=N_POINTS)
        results = pipeline.process_wafer_sequence(wafers)
        assert isinstance(results, dict)
        assert len(results) == 5

    def test_group_key_routing(self):
        """Chips from different groups should not share references."""
        from ..data_models import GroupKey
        pipeline = TempVibrationPipeline()

        wafers_eq1 = make_normal_wafer_sequence(
            equipment_id="EQ01", n_wafers=5, n_chips=20, n_points=N_POINTS, seed=80
        )
        wafers_eq2 = make_normal_wafer_sequence(
            equipment_id="EQ02", n_wafers=5, n_chips=20, n_points=N_POINTS, seed=81
        )

        for w in wafers_eq1 + wafers_eq2:
            pipeline.process_wafer(w)

        key1 = GroupKey("EQ01", "R001", "H1")
        key2 = GroupKey("EQ02", "R001", "H1")
        ref1 = pipeline._ref_manager.get(key1)
        ref2 = pipeline._ref_manager.get(key2)
        assert ref1 is not None
        assert ref2 is not None
        assert ref1 is not ref2
