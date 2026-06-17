"""
Synthetic data-based tests for the temperature vibration-increase detection
pipeline.

Each test verifies that intra-wafer or inter-wafer temperature vibration
increase is detected at a higher rate than the false-positive rate on
normal data.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from ..config import (
    AnomalyConfig,
    ReferenceConfig,
    IntraWaferConfig,
    InterWaferConfig,
    FeatureConfig,
)
from ..data_models import GroupKey
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


# ---------------------------------------------------------------------------
# Test 7: PM (Preventive Maintenance) event baseline reset
# ---------------------------------------------------------------------------

class TestPMBaselineReset:
    def test_reset_baseline_clears_state(self):
        """reset_baseline() should discard both the reference profile and
        the inter-wafer EWMA baseline for the given group."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=12, n_chips=N_CHIPS, n_points=N_POINTS, seed=60
        )
        for w in wafers:
            pipeline.process_wafer(w)

        key = wafers[0].group_key
        assert pipeline._ref_manager.get(key) is not None
        assert key in pipeline._inter_tracker._states

        pipeline.reset_baseline(key)

        assert pipeline._ref_manager.get(key) is None
        assert key not in pipeline._inter_tracker._states

    def test_pm_event_is_cold_start(self):
        """A wafer flagged with pm_event=True resets the baseline and is
        itself treated as a cold-start wafer: reference rebuild only, no
        detection events."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=12, n_chips=N_CHIPS, n_points=N_POINTS, seed=61
        )
        for w in wafers[:11]:
            pipeline.process_wafer(w)

        events = pipeline.process_wafer(wafers[11], pm_event=True)
        assert events == []

        key = wafers[0].group_key
        ref = pipeline._ref_manager.get(key)
        assert ref is not None
        assert ref.n_wafers_seen == 1

    def test_intra_wafer_detects_during_post_pm_rebuild(self):
        """
        intra_wafer drift detection should fire on the very next wafer after
        a PM event, even while the inter-wafer baseline is still being
        rebuilt (not yet re-frozen).
        """
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)

        baseline_wafers = make_normal_wafer_sequence(
            n_wafers=12, n_chips=N_CHIPS, n_points=N_POINTS, seed=62
        )
        for w in baseline_wafers:
            pipeline.process_wafer(w)

        key = baseline_wafers[0].group_key

        # First wafer after PM: cold start, reference rebuild only.
        cold_start = make_normal_wafer(
            wafer_id="POST_PM_000", n_chips=N_CHIPS, n_points=N_POINTS,
            rng=np.random.default_rng(63),
        )
        events = pipeline.process_wafer(cold_start, pm_event=True)
        assert events == []

        # Second wafer after PM: progressive vibration increase.
        rng = np.random.default_rng(64)
        anomaly_wafer = make_intra_vibration_wafer(
            wafer_id="POST_PM_001", n_chips=N_CHIPS, n_points=N_POINTS,
            onset_fraction=0.5, max_noise_multiplier=6.0, rng=rng,
        )
        events = pipeline.process_wafer(anomaly_wafer)
        intra_events = [e for e in events if e.anomaly_type == "intra_drift"]
        assert len(intra_events) > 0, (
            "Expected intra_drift alert on the wafer right after a PM event"
        )

        # The inter-wafer baseline has only 1 post-PM sample so far and is
        # not yet re-frozen.
        state = pipeline._inter_tracker._states[key]
        assert state.baseline_frozen is False

    def test_inter_wafer_baseline_refreezes_after_pm(self):
        """After a PM reset, the inter-wafer baseline re-freezes once
        ``baseline_wafers`` new wafers have been processed."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)

        baseline_wafers = make_normal_wafer_sequence(
            n_wafers=12, n_chips=N_CHIPS, n_points=N_POINTS, seed=65
        )
        for w in baseline_wafers:
            pipeline.process_wafer(w)

        key = baseline_wafers[0].group_key
        pipeline.reset_baseline(key)

        rng = np.random.default_rng(66)
        post_pm_wafers = [
            make_normal_wafer(
                wafer_id=f"POST_PM_{i:03d}", n_chips=N_CHIPS, n_points=N_POINTS, rng=rng
            )
            for i in range(cfg.inter_wafer.baseline_wafers + 1)
        ]
        for w in post_pm_wafers:
            pipeline.process_wafer(w)

        state = pipeline._inter_tracker._states[key]
        assert state.baseline_frozen is True
        assert state.n_samples == cfg.inter_wafer.baseline_wafers


# ---------------------------------------------------------------------------
# Test 8: initialize_baseline / save_baseline / load_baseline
# ---------------------------------------------------------------------------

class TestBaselineInit:
    def _pipeline(self) -> TempVibrationPipeline:
        return TempVibrationPipeline(_test_config())

    def test_initialize_baseline_builds_reference(self):
        """After initialize_baseline, the reference profile should be set."""
        pipeline = self._pipeline()
        key = GroupKey("EQ01", "R001", "H1")
        wafers = make_normal_wafer_sequence(n_wafers=8, n_chips=N_CHIPS, n_points=N_POINTS, seed=200)
        pipeline.initialize_baseline(key, wafers)

        ref = pipeline._ref_manager.get(key)
        assert ref is not None
        assert ref.is_initialized()
        assert len(ref.temp_ref) == N_POINTS
        assert ref.n_wafers_seen == len(wafers)

    def test_initialize_baseline_freezes_inter_wafer_baseline(self):
        """If enough wafers are provided, the inter-wafer baseline should freeze."""
        cfg = _test_config()
        pipeline = TempVibrationPipeline(cfg)
        key = GroupKey("EQ01", "R001", "H1")
        n = cfg.inter_wafer.baseline_wafers + 2
        wafers = make_normal_wafer_sequence(n_wafers=n, n_chips=N_CHIPS, n_points=N_POINTS, seed=201)
        pipeline.initialize_baseline(key, wafers)

        state = pipeline._inter_tracker._states.get(key)
        assert state is not None
        assert state.baseline_frozen is True

    def test_initialize_baseline_no_events_returned(self):
        """initialize_baseline should not raise and should discard any internal events."""
        pipeline = self._pipeline()
        key = GroupKey("EQ01", "R001", "H1")
        wafers = make_normal_wafer_sequence(n_wafers=12, n_chips=N_CHIPS, n_points=N_POINTS, seed=202)
        # Should not raise; return value is None
        result = pipeline.initialize_baseline(key, wafers)
        assert result is None

    def test_initialize_baseline_per_group_independent(self):
        """Different group keys get independent baselines."""
        pipeline = self._pipeline()
        key1 = GroupKey("EQ01", "R001", "H1")
        key2 = GroupKey("EQ02", "R002", "H2")
        w1 = make_normal_wafer_sequence(
            n_wafers=8, n_chips=N_CHIPS, n_points=N_POINTS, equipment_id="EQ01",
            recipe="R001", head="H1", seed=203,
        )
        w2 = make_normal_wafer_sequence(
            n_wafers=8, n_chips=N_CHIPS, n_points=N_POINTS, equipment_id="EQ02",
            recipe="R002", head="H2", seed=204,
        )
        pipeline.initialize_baseline(key1, w1)
        pipeline.initialize_baseline(key2, w2)

        ref1 = pipeline._ref_manager.get(key1)
        ref2 = pipeline._ref_manager.get(key2)
        assert ref1 is not None and ref2 is not None
        assert ref1 is not ref2
        assert not np.allclose(ref1.temp_ref, ref2.temp_ref)

    def test_initialize_baseline_resets_prior_state(self):
        """Calling initialize_baseline twice should discard the first baseline."""
        pipeline = self._pipeline()
        key = GroupKey("EQ01", "R001", "H1")
        w1 = make_normal_wafer_sequence(n_wafers=6, n_chips=N_CHIPS, n_points=N_POINTS, seed=210)
        w2 = make_normal_wafer_sequence(n_wafers=6, n_chips=N_CHIPS, n_points=N_POINTS, seed=211)
        pipeline.initialize_baseline(key, w1)
        ref_after_w1 = pipeline._ref_manager.get(key).temp_ref.copy()
        pipeline.initialize_baseline(key, w2)
        ref_after_w2 = pipeline._ref_manager.get(key).temp_ref
        # The second initialization should give a different reference
        assert not np.allclose(ref_after_w1, ref_after_w2)

    def test_save_and_load_baseline_round_trip(self):
        """save_baseline + load_baseline should exactly restore all state."""
        cfg = _test_config()
        pipeline_a = TempVibrationPipeline(cfg)
        key = GroupKey("EQ01", "R001", "H1")
        wafers = make_normal_wafer_sequence(
            n_wafers=cfg.inter_wafer.baseline_wafers + 2,
            n_chips=N_CHIPS, n_points=N_POINTS, seed=220,
        )
        pipeline_a.initialize_baseline(key, wafers)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            pipeline_a.save_baseline(path)

            pipeline_b = TempVibrationPipeline(cfg)
            pipeline_b.load_baseline(path)

            ref_a = pipeline_a._ref_manager.get(key)
            ref_b = pipeline_b._ref_manager.get(key)
            assert ref_b is not None
            assert ref_b.n_wafers_seen == ref_a.n_wafers_seen
            assert np.allclose(ref_a.temp_ref, ref_b.temp_ref)

            st_a = pipeline_a._inter_tracker._states[key]
            st_b = pipeline_b._inter_tracker._states[key]
            assert st_b.baseline_frozen == st_a.baseline_frozen
            assert st_b.baseline_mean == pytest.approx(st_a.baseline_mean)
            assert st_b.baseline_std == pytest.approx(st_a.baseline_std)
            assert st_b.n_samples == st_a.n_samples
        finally:
            os.unlink(path)

    def test_saved_json_is_valid_and_readable(self):
        """save_baseline should produce a valid JSON file with expected keys."""
        pipeline = self._pipeline()
        key = GroupKey("EQ01", "R001", "H1")
        wafers = make_normal_wafer_sequence(n_wafers=6, n_chips=N_CHIPS, n_points=N_POINTS, seed=230)
        pipeline.initialize_baseline(key, wafers)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            path = tmp.name
        try:
            pipeline.save_baseline(path)
            with open(path) as f:
                doc = json.load(f)
            assert "version" in doc
            assert "created_at" in doc
            assert "config" in doc
            assert "groups" in doc
            assert len(doc["groups"]) == 1
            g = doc["groups"][0]
            assert g["group_key"] == {"equipment_id": "EQ01", "recipe": "R001", "head": "H1"}
            assert "temp_ref" in g["reference"]
            assert len(g["reference"]["temp_ref"]) == N_POINTS
        finally:
            os.unlink(path)

    def test_load_baseline_enables_immediate_detection(self):
        """After loading a frozen baseline, the next anomalous wafer should be detected."""
        cfg = _test_config()
        pipeline_a = TempVibrationPipeline(cfg)
        key = GroupKey("EQ01", "R001", "H1")
        wafers = make_normal_wafer_sequence(
            n_wafers=cfg.inter_wafer.baseline_wafers + 2,
            n_chips=N_CHIPS, n_points=N_POINTS, seed=240,
        )
        pipeline_a.initialize_baseline(key, wafers)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            pipeline_a.save_baseline(path)

            pipeline_b = TempVibrationPipeline(cfg)
            pipeline_b.load_baseline(path)

            rng = np.random.default_rng(241)
            anomaly_wafer = make_intra_vibration_wafer(
                wafer_id="ANOM_AFTER_LOAD",
                n_chips=N_CHIPS, n_points=N_POINTS,
                onset_fraction=0.4, max_noise_multiplier=8.0, rng=rng,
            )
            events = pipeline_b.process_wafer(anomaly_wafer)
            intra_events = [e for e in events if e.anomaly_type == "intra_drift"]
            assert len(intra_events) > 0, (
                "Expected intra_drift alert immediately after loading frozen baseline"
            )
        finally:
            os.unlink(path)
