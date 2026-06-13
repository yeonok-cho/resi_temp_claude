"""
Minimal NumPy-only multilayer-perceptron autoencoder used as a simple,
unsupervised "joint signature" anomaly detector for paired temperature /
resistance residual time series.

No external ML dependencies (no torch, no sklearn) — just numpy, trained
with a hand-rolled Adam optimizer on a tanh MLP. Reconstruction error is the
anomaly score: a chip whose (temp_residual, resist_residual) shape looks
unlike anything seen during the baseline period reconstructs poorly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AutoencoderConfig, DEFAULT_CONFIG
from ..data_models import ChipData, GroupKey, AnomalyEvent


# ---------------------------------------------------------------------------
# Input vector construction
# ---------------------------------------------------------------------------

def _downsample(series: np.ndarray, target_len: int) -> np.ndarray:
    """Block-average ``series`` down to ``target_len`` points."""
    n = len(series)
    if n == target_len:
        return series.astype(float)
    edges = np.linspace(0, n, target_len + 1).astype(int)
    return np.array([
        series[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()
        for i in range(target_len)
    ])


def extract_residual_vector(
    chip: ChipData,
    temp_ref: np.ndarray,
    resist_ref: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """
    Build the autoencoder input vector for one chip: downsampled
    ``[temp_residual, resist_residual]`` concatenated to length
    ``2 * target_len``.
    """
    ref_len = min(len(temp_ref), len(resist_ref), chip.n_points)
    temp_res = chip.temp_series[:ref_len] - temp_ref[:ref_len]
    resist_res = chip.resist_series[:ref_len] - resist_ref[:ref_len]
    return np.concatenate([
        _downsample(temp_res, target_len),
        _downsample(resist_res, target_len),
    ])


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class SimpleAutoencoder:
    """
    Small tanh MLP autoencoder: input -> hidden -> bottleneck -> hidden -> input.

    Trained with full-batch gradient descent (Adam) on a reconstruction
    (MSE) loss. ``reconstruction_error`` is the per-sample anomaly score.
    """

    _PARAMS = ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")

    def __init__(self, input_dim: int, hidden_dim: int = 16, bottleneck_dim: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)

        def _init(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))

        self.W1 = _init(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = _init(hidden_dim, bottleneck_dim)
        self.b2 = np.zeros(bottleneck_dim)
        self.W3 = _init(bottleneck_dim, hidden_dim)
        self.b3 = np.zeros(hidden_dim)
        self.W4 = _init(hidden_dim, input_dim)
        self.b4 = np.zeros(input_dim)

        self._m = {p: np.zeros_like(getattr(self, p)) for p in self._PARAMS}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self._PARAMS}
        self._t = 0

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, tuple]:
        a1 = np.tanh(X @ self.W1 + self.b1)
        a2 = np.tanh(a1 @ self.W2 + self.b2)
        a3 = np.tanh(a2 @ self.W3 + self.b3)
        out = a3 @ self.W4 + self.b4
        return out, (a1, a2, a3)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample mean squared reconstruction error, shape (N,)."""
        out, _ = self._forward(X)
        return np.mean((out - X) ** 2, axis=1)

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 150,
        lr: float = 0.02,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Full-batch Adam training to minimize reconstruction MSE on X."""
        n, d = X.shape
        for _ in range(epochs):
            out, (a1, a2, a3) = self._forward(X)
            d_out = 2.0 * (out - X) / (n * d)

            dW4 = a3.T @ d_out
            db4 = d_out.sum(axis=0)
            dz3 = (d_out @ self.W4.T) * (1 - a3 ** 2)

            dW3 = a2.T @ dz3
            db3 = dz3.sum(axis=0)
            dz2 = (dz3 @ self.W3.T) * (1 - a2 ** 2)

            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            dz1 = (dz2 @ self.W2.T) * (1 - a1 ** 2)

            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)

            grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3, W4=dW4, b4=db4)
            self._t += 1
            for p in self._PARAMS:
                g = grads[p]
                self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
                self._v[p] = beta2 * self._v[p] + (1 - beta2) * (g ** 2)
                m_hat = self._m[p] / (1 - beta1 ** self._t)
                v_hat = self._v[p] / (1 - beta2 ** self._t)
                setattr(self, p, getattr(self, p) - lr * m_hat / (np.sqrt(v_hat) + eps))


# ---------------------------------------------------------------------------
# Per-group anomaly detector wrapper
# ---------------------------------------------------------------------------

@dataclass
class AutoencoderState:
    """
    Per-group autoencoder state.

    During the baseline period (first ``baseline_wafers`` wafers), chip
    residual vectors are buffered. Once enough wafers are collected, the
    autoencoder is fit once on the buffered data and the buffer is dropped.
    """
    autoencoder: SimpleAutoencoder | None = None
    training_data: list[np.ndarray] = field(default_factory=list)
    scale: np.ndarray = field(default_factory=lambda: np.ones(2))
    threshold: float = 0.0
    n_wafers_seen: int = 0
    fitted: bool = False


@dataclass
class AutoencoderAnomalyDetector:
    """
    Tracks per-group autoencoder state for joint temp/resist residual
    anomaly detection.

    One detector instance covers all groups; state is keyed by group_key.
    """
    config: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    _states: dict[GroupKey, AutoencoderState] = field(default_factory=dict, init=False)

    def _get_state(self, group_key: GroupKey) -> AutoencoderState:
        if group_key not in self._states:
            self._states[group_key] = AutoencoderState()
        return self._states[group_key]

    def process_wafer(
        self,
        group_key: GroupKey,
        wafer_id: str,
        chips: list[ChipData],
        residual_vectors: list[np.ndarray],
    ) -> list[AnomalyEvent]:
        """
        During the baseline period, buffer ``residual_vectors`` for training
        and return no events. Once fitted, score each chip's residual vector
        by reconstruction error and emit a ``deep_anomaly`` event for chips
        that exceed the training-derived threshold.

        ``chips`` and ``residual_vectors`` must be aligned (same order,
        same length).
        """
        state = self._get_state(group_key)
        cfg = self.config
        events: list[AnomalyEvent] = []

        if not state.fitted:
            state.training_data.extend(residual_vectors)
            state.n_wafers_seen += 1
            if state.n_wafers_seen >= cfg.baseline_wafers:
                self._fit(state, cfg)
            return events

        X = self._normalize(state, np.stack(residual_vectors))
        errors = state.autoencoder.reconstruction_error(X)

        for chip, err in zip(chips, errors):
            if err > state.threshold:
                events.append(AnomalyEvent(
                    group_key=group_key,
                    wafer_id=wafer_id,
                    anomaly_type="deep_anomaly",
                    channel="both",
                    severity=float(err / state.threshold),
                    details={
                        "chip_x": chip.x,
                        "chip_y": chip.y,
                        "bond_order": chip.bond_order,
                        "reconstruction_error": float(err),
                        "threshold": state.threshold,
                    },
                ))

        return events

    def _fit(self, state: AutoencoderState, cfg: AutoencoderConfig) -> None:
        X_raw = np.stack(state.training_data)
        L = cfg.input_length
        scale_temp = max(float(np.std(X_raw[:, :L])), 1e-9)
        scale_resist = max(float(np.std(X_raw[:, L:])), 1e-9)
        state.scale = np.array([scale_temp, scale_resist])

        X = self._normalize(state, X_raw)
        ae = SimpleAutoencoder(
            input_dim=X.shape[1],
            hidden_dim=cfg.hidden_dim,
            bottleneck_dim=cfg.bottleneck_dim,
            seed=cfg.seed,
        )
        ae.fit(X, epochs=cfg.epochs, lr=cfg.learning_rate)

        state.autoencoder = ae
        state.threshold = float(np.percentile(ae.reconstruction_error(X), cfg.threshold_percentile))
        state.fitted = True
        state.training_data = []

    def _normalize(self, state: AutoencoderState, X: np.ndarray) -> np.ndarray:
        L = self.config.input_length
        out = X.astype(float).copy()
        out[:, :L] /= state.scale[0]
        out[:, L:] /= state.scale[1]
        return out
