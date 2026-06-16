"""
Tests for the RMT-based MP-score anomaly detector.

Synthetic data: normal residuals ~ N(0, sigma), anomalous residuals have
an added structured spike in one direction.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from ..config import MPConfig
from ..base_profile import BaseProfile, extract_channel_residual, _downsample
from ..mp_detector import MPProfileDetector
from hbm_anomaly_detection.data_models import GroupKey
from hbm_anomaly_detection.simulator import make_normal_wafer_sequence, make_normal_wafer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY = GroupKey("EQ01", "R001", "H1")
_P = 20      # input_length (feature dimension)
_N_CHIPS = 50
_N_POINTS = 200


def _normal_residuals(n_chips: int, p: int, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """Pure-noise residual matrix: (n_chips, p)."""
    return np.random.default_rng(seed).normal(0, sigma, size=(n_chips, p))


def _spike_residual(p: int, spike_dir: np.ndarray, amplitude: float, sigma: float = 1.0,
                    seed: int = 42) -> np.ndarray:
    """One residual vector = Gaussian noise + structured spike."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma, size=p) + amplitude * spike_dir / np.linalg.norm(spike_dir)


def _make_residuals_for_wafer(n_chips: int, p: int, sigma: float, seed: int) -> list[np.ndarray]:
    X = _normal_residuals(n_chips, p, sigma, seed)
    return list(X)


def _build_detector_after_baseline(cfg: MPConfig, n_baseline_wafers: int | None = None) -> tuple[MPProfileDetector, _KEY]:
    """Feed baseline wafers so both channels are fitted."""
    n = n_baseline_wafers or cfg.baseline_wafers
    det = MPProfileDetector(cfg)
    for i in range(n + 1):  # +1 for cold-start-style first wafer
        temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i)
        resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 0.1, seed=i + 100)
        dummy_chips = _dummy_chips(_N_CHIPS)
        det.process_wafer(_KEY, f"W{i:03d}", temp_res, resist_res, dummy_chips)
    return det, _KEY


def _dummy_chips(n: int):
    from hbm_anomaly_detection.data_models import ChipData
    return [
        ChipData(
            equipment_id="EQ01", recipe="R001", head="H1", wafer_id="W000",
            x=i % 8, y=i // 8, bond_order=i,
            temp_series=np.zeros(10), resist_series=np.zeros(10),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests: BaseProfile
# ---------------------------------------------------------------------------

class TestBaseProfile:
    def test_fit_sets_dimensions(self):
        """After fit, k_noise + k_signal should equal p."""
        X = _normal_residuals(200, _P, seed=0)
        bp = BaseProfile()
        bp.fit(X, min_noise_rank=2)
        assert bp.fitted
        assert bp.k_noise_ + bp.k_signal_ == _P
        assert bp.k_noise_ >= 2

    def test_normal_scores_near_k_noise(self):
        """Mean score on hold-out normal data should be close to k_noise (chi2 mean)."""
        X_train = _normal_residuals(500, _P, seed=1)
        X_test = _normal_residuals(200, _P, seed=2)
        bp = BaseProfile()
        bp.fit(X_train, min_noise_rank=2)
        scores = np.array([bp.score(x) for x in X_test])
        # chi2(k_noise) has mean k_noise
        assert abs(scores.mean() / bp.k_noise_ - 1.0) < 0.2

    def test_spike_residual_high_score(self):
        """A residual with a large structured spike that is not in the signal subspace
        should score much higher than the threshold."""
        X_train = _normal_residuals(500, _P, sigma=1.0, seed=3)
        bp = BaseProfile()
        bp.fit(X_train, min_noise_rank=2)

        # Spike in the first standard-basis direction (not learned as signal if data is isotropic)
        spike_dir = np.zeros(_P)
        spike_dir[0] = 1.0
        anomalous = _spike_residual(_P, spike_dir, amplitude=20.0, sigma=1.0, seed=99)
        s = bp.score(anomalous)
        thr = bp.threshold(alert_alpha=0.01)
        assert s > thr, f"spike score {s:.1f} should exceed threshold {thr:.1f}"

    def test_threshold_is_chi2_ppf(self):
        """Threshold should equal chi2.ppf(1-alpha, k_noise)."""
        X = _normal_residuals(300, _P, seed=4)
        bp = BaseProfile()
        bp.fit(X, min_noise_rank=2)
        thr = bp.threshold(0.01)
        expected = float(chi2.ppf(0.99, df=bp.k_noise_))
        assert abs(thr - expected) < 1e-10

    def test_mp_bulk_edge_separates_spike_eigenvalue(self):
        """A dataset with an injected structured direction should have k_signal >= 1."""
        rng = np.random.default_rng(5)
        n, p = 500, _P
        X_noise = rng.normal(0, 1.0, (n, p))
        # Inject a spike component along the first direction with amplitude 5
        spike = rng.normal(0, 5.0, (n, 1)) * np.eye(p)[[0]]
        X = X_noise + spike
        bp = BaseProfile()
        bp.fit(X, min_noise_rank=1)
        assert bp.k_signal_ >= 1, "Expected at least one eigenvalue above the MP upper edge"

    def test_false_positive_rate_below_alpha(self):
        """On pure-noise data, fraction of scores > threshold should be <= alpha + slack."""
        X_train = _normal_residuals(500, _P, seed=6)
        X_test = _normal_residuals(1000, _P, seed=7)
        bp = BaseProfile()
        bp.fit(X_train, min_noise_rank=2)
        thr = bp.threshold(alert_alpha=0.01)
        fp_rate = float(np.mean([bp.score(x) > thr for x in X_test]))
        assert fp_rate <= 0.05, f"FP rate {fp_rate:.1%} too high for alpha=0.01"


# ---------------------------------------------------------------------------
# Unit tests: extract_channel_residual / _downsample
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_downsample_output_length(self):
        x = np.arange(100, dtype=float)
        assert len(_downsample(x, 20)) == 20

    def test_downsample_preserves_mean(self):
        x = np.ones(100, dtype=float) * 3.5
        ds = _downsample(x, 10)
        assert np.allclose(ds, 3.5)

    def test_extract_channel_residual_length(self):
        raw = np.random.default_rng(0).normal(0, 1, 200)
        ref = np.random.default_rng(1).normal(0, 1, 200)
        res = extract_channel_residual(raw, ref, target_len=25)
        assert res.shape == (25,)

    def test_extract_channel_residual_zero_for_identical(self):
        ref = np.ones(200) * 2.0
        res = extract_channel_residual(ref.copy(), ref, target_len=10)
        assert np.allclose(res, 0.0)


# ---------------------------------------------------------------------------
# Integration tests: MPProfileDetector
# ---------------------------------------------------------------------------

class TestMPProfileDetector:
    def _cfg(self, **kwargs) -> MPConfig:
        defaults = dict(input_length=_P, baseline_wafers=4, alert_alpha=0.01, min_noise_rank=2)
        defaults.update(kwargs)
        return MPConfig(**defaults)

    def test_no_events_during_baseline(self):
        """During the baseline period, no events should be emitted."""
        cfg = self._cfg()
        det = MPProfileDetector(cfg)
        for i in range(cfg.baseline_wafers - 1):
            temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i)
            resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 0.1, seed=i + 50)
            events = det.process_wafer(_KEY, f"W{i}", temp_res, resist_res, _dummy_chips(_N_CHIPS))
            assert events == [], f"Unexpected events during baseline at wafer {i}"

    def test_fitted_after_baseline(self):
        """Both channel models should be fitted after baseline_wafers wafers."""
        cfg = self._cfg()
        det = MPProfileDetector(cfg)
        for i in range(cfg.baseline_wafers + 1):
            temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i)
            resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 0.1, seed=i + 50)
            det.process_wafer(_KEY, f"W{i}", temp_res, resist_res, _dummy_chips(_N_CHIPS))
        st = det._states[_KEY]
        assert st.temp.fitted
        assert st.resist.fitted

    def test_normal_wafers_low_fp_rate(self):
        """After fitting, normal wafers should produce few mp_anomaly events."""
        cfg = self._cfg(baseline_wafers=5)
        det = MPProfileDetector(cfg)
        n_total = 20
        n_events = 0
        for i in range(n_total):
            temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i * 2)
            resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i * 2 + 1)
            events = det.process_wafer(_KEY, f"W{i}", temp_res, resist_res, _dummy_chips(_N_CHIPS))
            if det._states[_KEY].temp.fitted:
                n_events += len(events)

        # Each channel contributes independently; across both channels, alpha=0.01
        n_scored_wafers = n_total - cfg.baseline_wafers
        fp_rate = n_events / (n_scored_wafers * _N_CHIPS * 2)
        assert fp_rate <= 0.05, f"FP rate {fp_rate:.1%} too high"

    def test_anomalous_chip_detected(self):
        """A chip with a large structured spike in the temp residual should be detected."""
        cfg = self._cfg(baseline_wafers=5, alert_alpha=0.001)
        det = MPProfileDetector(cfg)

        # Feed baseline
        for i in range(cfg.baseline_wafers + 1):
            temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i)
            resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i + 50)
            det.process_wafer(_KEY, f"W{i}", temp_res, resist_res, _dummy_chips(_N_CHIPS))

        # One anomalous wafer: first chip has a strong spike
        spike_dir = np.zeros(cfg.input_length)
        spike_dir[:5] = 1.0  # anomaly in the first 5 time points
        rng = np.random.default_rng(200)
        temp_res_anom = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=200)
        temp_res_anom[0] = (
            rng.normal(0, 1.0, cfg.input_length)
            + 30.0 * spike_dir / np.linalg.norm(spike_dir)
        )
        resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=250)
        events = det.process_wafer(_KEY, "W_ANOM", temp_res_anom, resist_res, _dummy_chips(_N_CHIPS))

        mp_temp_events = [e for e in events if e.anomaly_type == "mp_anomaly_temp"]
        assert len(mp_temp_events) > 0, "Anomalous chip should trigger mp_anomaly_temp"
        # The anomalous chip (index 0) should be among flagged chips
        flagged_bond_orders = {e.details["bond_order"] for e in mp_temp_events}
        assert 0 in flagged_bond_orders

    def test_both_channels_independent(self):
        """An anomaly in resist only should not generate temp events."""
        cfg = self._cfg(baseline_wafers=5, alert_alpha=0.001)
        det = MPProfileDetector(cfg)

        for i in range(cfg.baseline_wafers + 1):
            temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i)
            resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i + 50)
            det.process_wafer(_KEY, f"W{i}", temp_res, resist_res, _dummy_chips(_N_CHIPS))

        spike_dir = np.ones(cfg.input_length) / np.sqrt(cfg.input_length)
        temp_res_normal = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=300)
        resist_res_anom = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=350)
        resist_res_anom[0] = np.random.default_rng(301).normal(0, 1, cfg.input_length) + 30 * spike_dir

        events = det.process_wafer(_KEY, "W_RESI_ANOM", temp_res_normal, resist_res_anom, _dummy_chips(_N_CHIPS))

        temp_events = [e for e in events if e.anomaly_type == "mp_anomaly_temp"]
        resist_events = [e for e in events if e.anomaly_type == "mp_anomaly_resist"]
        assert len(resist_events) > 0, "Should detect anomaly in resist channel"
        assert len(temp_events) == 0 or len(temp_events) < len(resist_events), (
            "Resist-only anomaly should not generate more temp events than resist events"
        )

    def test_periodic_retrain(self):
        """With retrain_interval_wafers=3, model is re-fitted after 3 scored wafers."""
        cfg = self._cfg(baseline_wafers=4, retrain_interval_wafers=3)
        det = MPProfileDetector(cfg)

        # Process exactly baseline_wafers wafers: the last one triggers the fit
        # but returns early (no retrain increment), so n_wafers_since_fit stays 0.
        for i in range(cfg.baseline_wafers):
            temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i)
            resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i + 50)
            det.process_wafer(_KEY, f"W{i}", temp_res, resist_res, _dummy_chips(_N_CHIPS))

        first_bp = det._states[_KEY].temp.base_profile
        for i in range(cfg.retrain_interval_wafers):
            temp_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i + 100)
            resist_res = _make_residuals_for_wafer(_N_CHIPS, cfg.input_length, 1.0, seed=i + 150)
            det.process_wafer(_KEY, f"WR{i}", temp_res, resist_res, _dummy_chips(_N_CHIPS))

        assert det._states[_KEY].temp.base_profile is not first_bp
        assert det._states[_KEY].temp.n_wafers_since_fit == 0
