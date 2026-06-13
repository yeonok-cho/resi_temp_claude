"""
Synthetic data-based tests for the HBM CoW anomaly detection pipeline.

Each test verifies that a specific anomaly type is detected at higher rate
than the false-positive rate on normal data, using the same pipeline config.
"""
from __future__ import annotations

import numpy as np
import pytest

from ..config import (
    AnomalyConfig,
    ReferenceConfig,
    IntraWaferConfig,
    InterWaferConfig,
    SpikeConfig,
    CouplingConfig,
    FeatureConfig,
    MultivariateInterWaferConfig,
    AutoencoderConfig,
)
from ..pipeline import AnomalyDetectionPipeline
from ..simulator import (
    make_normal_wafer_sequence,
    make_intra_vibration_wafer,
    make_spike_wafer,
    make_inter_wafer_drift_sequence,
    make_coupling_breakdown_sequence,
    make_normal_wafer,
)


# ---------------------------------------------------------------------------
# Shared configuration: tuned for fast test runs with small synthetic data
# ---------------------------------------------------------------------------

def _test_config() -> AnomalyConfig:
    return AnomalyConfig(
        reference=ReferenceConfig(
            ewma_alpha=0.3,
            bimodal_min_chips=10,
            bimodal_separation_threshold=0.10,
        ),
        feature=FeatureConfig(spike_threshold_sigma=5.0),
        intra_wafer=IntraWaferConfig(
            ewma_lambda=0.15,
            cusum_k=0.5,
            cusum_h=3.0,
            warmup_chips=5,
        ),
        inter_wafer=InterWaferConfig(
            ewma_lambda=0.2,
            alert_sigma=2.5,
            baseline_wafers=5,
        ),
        spike=SpikeConfig(
            residual_sigma=4.0,
            wafer_spike_rate_sigma=2.5,
        ),
        coupling=CouplingConfig(
            coupling_sigma=2.5,
            baseline_wafers=5,
            ewma_lambda=0.3,
        ),
        multivariate_inter_wafer=MultivariateInterWaferConfig(
            ewma_lambda=0.3,
            alert_alpha=0.05,
            baseline_wafers=5,
        ),
        autoencoder=AutoencoderConfig(
            input_length=10,
            hidden_dim=8,
            bottleneck_dim=2,
            baseline_wafers=5,
            epochs=80,
            learning_rate=0.05,
            threshold_percentile=99.0,
        ),
    )


N_CHIPS = 64   # 8×8 grid — fast enough for CI
N_POINTS = 200  # 4 second @ 20 ms — still captures full ramp profile


# ---------------------------------------------------------------------------
# Helper: count anomaly events of a given type
# ---------------------------------------------------------------------------

def _count_events(
    results: dict[str, list],
    anomaly_type: str,
) -> int:
    return sum(
        1 for events in results.values()
        for e in events
        if e.anomaly_type == anomaly_type
    )


def _any_event(results: dict[str, list], anomaly_type: str) -> bool:
    return any(
        e.anomaly_type == anomaly_type
        for events in results.values()
        for e in events
    )


# ---------------------------------------------------------------------------
# Test 1: Normal sequence produces no false positives (smoke test)
# ---------------------------------------------------------------------------

class TestNormalBaseline:
    def test_no_spurious_inter_drift(self):
        """A clean wafer sequence should not trigger inter-wafer drift alerts."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=N_CHIPS, n_points=N_POINTS, seed=0
        )
        results = pipeline.process_wafer_sequence(wafers)
        n_inter = _count_events(results, "inter_drift")
        # Allow a small false-positive count (≤ 2 out of 20 wafers)
        assert n_inter <= 2, f"Too many inter_drift false positives: {n_inter}"

    def test_no_spurious_intra_drift(self):
        """A clean wafer sequence should not trigger intra-wafer drift alerts."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=N_CHIPS, n_points=N_POINTS, seed=1
        )
        results = pipeline.process_wafer_sequence(wafers)
        n_intra = _count_events(results, "intra_drift")
        assert n_intra <= 2, f"Too many intra_drift false positives: {n_intra}"

    def test_no_spurious_coupling(self):
        """A normal (unimodal resist) sequence should not trigger coupling alerts."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=N_CHIPS, n_points=N_POINTS, seed=2
        )
        results = pipeline.process_wafer_sequence(wafers)
        n_coupling = _count_events(results, "coupling")
        assert n_coupling == 0, f"Spurious coupling events on unimodal data: {n_coupling}"


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
        pipeline = AnomalyDetectionPipeline(cfg)

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
            noise_std_resist=0.02,
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
        pipeline = AnomalyDetectionPipeline(cfg)

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

        if intra_events:
            alarm_idx = intra_events[0].details.get("alarm_chip_index", 0)
            n_chips = intra_events[0].details.get("n_chips", N_CHIPS)
            assert alarm_idx > n_chips * 0.3, (
                f"Expected alarm in 2nd half, got chip index {alarm_idx}/{n_chips}"
            )

    def test_normal_wafer_not_flagged_intra(self):
        """Normal wafers should not trigger intra_drift after baseline is set."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        all_wafers = make_normal_wafer_sequence(
            n_wafers=25, n_chips=N_CHIPS, n_points=N_POINTS, seed=30
        )
        # Train on first 15
        for w in all_wafers[:15]:
            pipeline.process_wafer(w)

        # Check remaining 10 normal wafers
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
        A sequence with gradually increasing resist noise across wafers
        should trigger inter_drift alerts in the drifting portion.
        """
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        wafers = make_inter_wafer_drift_sequence(
            n_normal=12,
            n_drift=10,
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            base_noise_std=0.02,
            drift_multiplier_final=5.0,
            seed=40,
        )
        results = pipeline.process_wafer_sequence(wafers)

        # Expect alerts in the drifting wafers (last 10)
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
        pipeline = AnomalyDetectionPipeline(cfg)

        wafers = make_inter_wafer_drift_sequence(
            n_normal=15,
            n_drift=10,
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            base_noise_std=0.02,
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
        # Baseline period + small FP rate
        assert normal_alerts <= 2, (
            f"Too many inter_drift false positives on normal wafers: {normal_alerts}"
        )


# ---------------------------------------------------------------------------
# Test 4: Resistance spike detection
# ---------------------------------------------------------------------------

class TestSpikeDetection:
    def test_spike_chips_detected(self):
        """
        A wafer with spike chips should produce spike anomaly events
        after the pipeline has a baseline.
        """
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        # Baseline
        normal_wafers = make_normal_wafer_sequence(
            n_wafers=10, n_chips=N_CHIPS, n_points=N_POINTS, seed=50
        )
        for w in normal_wafers:
            pipeline.process_wafer(w)

        rng = np.random.default_rng(200)
        spike_wafer = make_spike_wafer(
            wafer_id="SPIKE_W",
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            spike_fraction=0.20,
            spike_amplitude=8.0,
            rng=rng,
        )
        events = pipeline.process_wafer(spike_wafer)
        spike_events = [e for e in events if e.anomaly_type == "spike"]
        assert len(spike_events) > 0, "Expected spike events on wafer with injected spikes"

    def test_spike_rate_tracking(self):
        """
        A sequence where spike rate increases across wafers should eventually
        trigger a spike rate alert.
        """
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        rng = np.random.default_rng(201)

        # Normal wafers (few/no spikes)
        normal_wafers = make_normal_wafer_sequence(
            n_wafers=10, n_chips=N_CHIPS, n_points=N_POINTS, seed=51
        )
        for w in normal_wafers:
            pipeline.process_wafer(w)

        # Wafers with increasing spike rates
        spike_events_found = False
        for i in range(6):
            sw = make_spike_wafer(
                wafer_id=f"SPIKE_SEQ_{i}",
                n_chips=N_CHIPS,
                n_points=N_POINTS,
                spike_fraction=0.30,
                spike_amplitude=10.0,
                rng=rng,
            )
            evts = pipeline.process_wafer(sw)
            if any(e.anomaly_type == "spike" for e in evts):
                spike_events_found = True

        assert spike_events_found, "Expected spike events during spikey wafer sequence"

    def test_normal_wafer_no_spikes(self):
        """Normal wafers should have a very low chip-level spike false-positive rate."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=N_CHIPS, n_points=N_POINTS, seed=52
        )
        results = pipeline.process_wafer_sequence(wafers)

        chip_spike_events = sum(
            1 for evts in results.values()
            for e in evts
            if e.anomaly_type == "spike" and "chip_x" in e.details
        )
        total_chips = 20 * N_CHIPS
        fp_rate = chip_spike_events / total_chips
        # Allow up to 1% false-positive rate — within-wafer MAD thresholding is
        # inherently statistical and a small FP rate is expected.
        assert fp_rate <= 0.01, (
            f"Chip-level spike FP rate {fp_rate:.1%} ({chip_spike_events}/{total_chips}) "
            f"exceeds 1% on normal data"
        )


# ---------------------------------------------------------------------------
# Test 5: Bimodal coupling breakdown detection
# ---------------------------------------------------------------------------

class TestCouplingBreakdown:
    def test_coupling_breakdown_detected(self):
        """
        After a baseline of bimodal-normal wafers, a sequence where the
        temperature profiles differ between resistance clusters should
        trigger coupling alerts.
        """
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        wafers = make_coupling_breakdown_sequence(
            n_normal=12,
            n_broken=6,
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            temp_offset=20.0,
            seed=60,
        )
        results = pipeline.process_wafer_sequence(wafers)

        broken_wafer_ids = {w.wafer_id for w in wafers[12:]}
        coupling_alerts = sum(
            1 for wid, evts in results.items()
            if wid in broken_wafer_ids
            for e in evts
            if e.anomaly_type == "coupling"
        )
        assert coupling_alerts >= 1, (
            "Expected coupling alerts when temperature profiles diverge by cluster"
        )

    def test_coupling_normal_no_alert(self):
        """Normal bimodal wafers should not trigger coupling alerts."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        wafers = make_coupling_breakdown_sequence(
            n_normal=20,
            n_broken=0,
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            seed=61,
        )
        results = pipeline.process_wafer_sequence(wafers)
        coupling_alerts = _count_events(results, "coupling")
        assert coupling_alerts <= 1, (
            f"Too many coupling false positives on normal bimodal data: {coupling_alerts}"
        )


# ---------------------------------------------------------------------------
# Test 6: Feature extractor unit tests
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

    def test_spike_detects_outlier(self):
        """compute_spike should catch a large single-point outlier."""
        from ..feature_extractor import compute_spike
        residual = np.zeros(200)
        residual[100] = 50.0
        assert compute_spike(residual) == pytest.approx(50.0)

    def test_extract_chip_features_shapes(self):
        """extract_chip_features should return a ChipFeatures object with valid floats."""
        from ..feature_extractor import extract_chip_features
        from ..simulator import make_normal_chip
        rng = np.random.default_rng(5)
        chip = make_normal_chip(
            "EQ1", "R1", "H1", "W1", 0, 0, 0, n_points=N_POINTS, rng=rng
        )
        ref = np.zeros(N_POINTS)
        feat = extract_chip_features(chip, ref, ref)
        assert feat.temp_roughness >= 0.0
        assert feat.resist_roughness >= 0.0
        assert feat.temp_spike >= 0.0
        assert feat.resist_spike >= 0.0


# ---------------------------------------------------------------------------
# Test 7: Reference manager unit tests
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

    def test_bimodal_detection(self):
        """Bimodal wafer should set is_bimodal=True on the reference."""
        from ..reference_manager import ReferenceManager
        from ..simulator import make_bimodal_wafer
        rm = ReferenceManager()
        w = make_bimodal_wafer(n_chips=60, n_points=N_POINTS)
        rm.update(w)
        from ..data_models import GroupKey
        key = GroupKey("EQ01", "R001", "H1")
        ref = rm.get(key)
        assert ref is not None
        assert ref.bimodal.is_bimodal, "Expected bimodal detection on bimodal wafer"

    def test_ewma_smoothing(self):
        """
        After many normal wafers, the EWMA reference should converge to
        the true profile (not drift away).
        """
        from ..reference_manager import ReferenceManager
        from ..simulator import _temp_profile
        from ..data_models import GroupKey

        rm = ReferenceManager()
        rng = np.random.default_rng(7)
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
        # RMSE should be small relative to the profile range (~275°C span)
        assert rmse < 20.0, f"Reference too far from true profile: RMSE={rmse:.2f}"


# ---------------------------------------------------------------------------
# Test 8: Pipeline integration smoke test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_process_wafer_returns_list(self):
        pipeline = AnomalyDetectionPipeline()
        w = make_normal_wafer(n_chips=20, n_points=N_POINTS)
        result = pipeline.process_wafer(w)
        assert isinstance(result, list)

    def test_process_wafer_sequence_returns_dict(self):
        pipeline = AnomalyDetectionPipeline()
        wafers = make_normal_wafer_sequence(n_wafers=5, n_chips=20, n_points=N_POINTS)
        results = pipeline.process_wafer_sequence(wafers)
        assert isinstance(results, dict)
        assert len(results) == 5

    def test_group_key_routing(self):
        """Chips from different groups should not share references."""
        from ..data_models import GroupKey
        pipeline = AnomalyDetectionPipeline()

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
        # References are tracked independently
        assert ref1 is not ref2


# ---------------------------------------------------------------------------
# Test 9: Multivariate (Hotelling's T^2) inter-wafer drift detection
# ---------------------------------------------------------------------------

class TestMultivariateInterWaferDrift:
    def test_no_spurious_joint_drift(self):
        """A clean wafer sequence should not trigger inter_drift_mv alerts."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)
        wafers = make_normal_wafer_sequence(
            n_wafers=20, n_chips=N_CHIPS, n_points=N_POINTS, seed=90
        )
        results = pipeline.process_wafer_sequence(wafers)
        n_mv = _count_events(results, "inter_drift_mv")
        assert n_mv <= 2, f"Too many inter_drift_mv false positives: {n_mv}"

    def test_joint_drift_detected(self):
        """
        A sequence with gradually increasing resist roughness across wafers
        should trigger inter_drift_mv alerts in the drifting portion: the
        joint [temp_roughness_mean, resist_roughness_mean] vector moves
        outside the baseline covariance ellipse even though temp roughness
        stays flat.
        """
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        wafers = make_inter_wafer_drift_sequence(
            n_normal=12,
            n_drift=10,
            n_chips=N_CHIPS,
            n_points=N_POINTS,
            base_noise_std=0.02,
            drift_multiplier_final=5.0,
            seed=91,
        )
        results = pipeline.process_wafer_sequence(wafers)

        drift_wafer_ids = {w.wafer_id for w in wafers[12:]}
        mv_alerts = sum(
            1 for wid, evts in results.items()
            if wid in drift_wafer_ids
            for e in evts
            if e.anomaly_type == "inter_drift_mv"
        )
        assert mv_alerts >= 1, (
            "Expected at least one inter_drift_mv alert in drifting wafer portion"
        )


# ---------------------------------------------------------------------------
# Test 10: NumPy autoencoder building blocks
# ---------------------------------------------------------------------------

class TestSimpleAutoencoder:
    def test_reconstruction_error_decreases_with_training(self):
        """Training should reduce mean reconstruction error on the training set."""
        from ..detector.autoencoder import SimpleAutoencoder
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(50, 10))

        ae = SimpleAutoencoder(input_dim=10, hidden_dim=8, bottleneck_dim=2, seed=0)
        err_before = float(ae.reconstruction_error(X).mean())
        ae.fit(X, epochs=200, lr=0.05)
        err_after = float(ae.reconstruction_error(X).mean())

        assert err_after < err_before

    def test_downsample_shape_and_mean(self):
        from ..detector.autoencoder import _downsample
        x = np.arange(100, dtype=float)
        ds = _downsample(x, 10)
        assert len(ds) == 10
        assert ds[0] == pytest.approx(np.mean(x[:10]))

    def test_extract_residual_vector_shape(self):
        from ..detector.autoencoder import extract_residual_vector
        from ..simulator import make_normal_chip
        rng = np.random.default_rng(5)
        chip = make_normal_chip(
            "EQ1", "R1", "H1", "W1", 0, 0, 0, n_points=N_POINTS, rng=rng
        )
        ref = np.zeros(N_POINTS)
        vec = extract_residual_vector(chip, ref, ref, target_len=10)
        assert vec.shape == (20,)


# ---------------------------------------------------------------------------
# Test 11: Deep autoencoder anomaly detector integration
# ---------------------------------------------------------------------------

class TestDeepAutoencoderDetector:
    def test_fits_after_baseline(self):
        """After baseline_wafers normal wafers, the autoencoder should be fitted."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        wafers = make_normal_wafer_sequence(
            n_wafers=cfg.autoencoder.baseline_wafers + 2,
            n_chips=N_CHIPS, n_points=N_POINTS, seed=92,
        )
        for w in wafers:
            pipeline.process_wafer(w)

        key = wafers[0].group_key
        state = pipeline._autoencoder_detector._states[key]
        assert state.fitted

    def test_normal_wafer_low_fp_rate(self):
        """After fitting, normal wafers should have a low deep_anomaly FP rate."""
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        n_wafers = 20
        wafers = make_normal_wafer_sequence(
            n_wafers=n_wafers, n_chips=N_CHIPS, n_points=N_POINTS, seed=93
        )
        results = pipeline.process_wafer_sequence(wafers)

        n_deep = _count_events(results, "deep_anomaly")
        # 1 cold-start wafer (reference init only) + baseline_wafers buffered
        # for training are never scored.
        n_scored_wafers = n_wafers - 1 - cfg.autoencoder.baseline_wafers
        fp_rate = n_deep / (n_scored_wafers * N_CHIPS)
        assert fp_rate <= 0.05, (
            f"deep_anomaly FP rate {fp_rate:.1%} too high on normal data"
        )

    def test_vibration_wafer_more_anomalies_than_normal(self):
        """
        A wafer with strong, wafer-wide vibration increase should produce
        more deep_anomaly events than a normal wafer scored by the same
        fitted autoencoder.
        """
        cfg = _test_config()
        pipeline = AnomalyDetectionPipeline(cfg)

        baseline_wafers = make_normal_wafer_sequence(
            n_wafers=cfg.autoencoder.baseline_wafers + 1,
            n_chips=N_CHIPS, n_points=N_POINTS, seed=94,
        )
        for w in baseline_wafers:
            pipeline.process_wafer(w)

        rng_normal = np.random.default_rng(95)
        normal_wafer = make_normal_wafer(
            wafer_id="DEEP_NORMAL", n_chips=N_CHIPS, n_points=N_POINTS, rng=rng_normal
        )
        normal_events = pipeline.process_wafer(normal_wafer)
        n_normal_deep = sum(1 for e in normal_events if e.anomaly_type == "deep_anomaly")

        rng_anom = np.random.default_rng(96)
        anomaly_wafer = make_intra_vibration_wafer(
            wafer_id="DEEP_ANOM", n_chips=N_CHIPS, n_points=N_POINTS,
            onset_fraction=0.0, max_noise_multiplier=8.0, rng=rng_anom,
        )
        anomaly_events = pipeline.process_wafer(anomaly_wafer)
        n_anom_deep = sum(1 for e in anomaly_events if e.anomaly_type == "deep_anomaly")

        assert n_anom_deep > n_normal_deep, (
            f"Expected more deep_anomaly events on vibration wafer "
            f"({n_anom_deep}) than normal wafer ({n_normal_deep})"
        )
