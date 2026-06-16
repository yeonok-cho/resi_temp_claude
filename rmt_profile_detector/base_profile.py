"""
PCA-based baseline profile model with Marchenko-Pastur bulk-edge
partitioning of the eigenspace.

Theory recap
------------
Given a baseline residual matrix X ∈ R^{n×p} (n chips, p features):

  Σ = X_c^T X_c / (n-1),  X_c = X - mean(X, axis=0)

Eigendecompose Σ = V Λ V^T.  Under a pure-noise null (X_c has i.i.d.
entries with variance σ²), the empirical eigenvalues follow the
Marchenko-Pastur distribution on [λ_, λ+] with

  λ± = σ²(1 ± √γ)²,  γ = p/n  (aspect ratio).

Eigenvalues above λ+ are "spikes" that represent structured common-mode
variation across chips.  The remaining k_noise eigenvalues form the
"noise subspace".

For a new chip residual x, the MP score is the chi-square statistic
in the noise subspace:

  score(x) = Σᵢ∈noise  (vᵢ·r)² / λᵢ   where r = x − mean_

Under the null this is chi²(k_noise), so the threshold at significance
level α is chi2.ppf(1-α, df=k_noise) — a purely theoretical value
that requires no data-driven calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2


def _downsample(series: np.ndarray, target_len: int) -> np.ndarray:
    """Block-average ``series`` to exactly ``target_len`` points."""
    n = len(series)
    if n == target_len:
        return series.astype(float)
    edges = np.linspace(0, n, target_len + 1).astype(int)
    return np.array([
        series[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()
        for i in range(target_len)
    ])


def extract_channel_residual(
    raw_series: np.ndarray,
    ref_series: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """
    Compute the downsampled residual for one channel.

    Returns a vector of length ``target_len``.
    """
    n = min(len(raw_series), len(ref_series))
    residual = raw_series[:n] - ref_series[:n]
    return _downsample(residual, target_len)


@dataclass
class BaseProfile:
    """
    Learned PCA model for one channel's residual distribution.

    After ``fit``, the model partitions the p-dimensional residual space
    into:
      - a k_signal-dimensional "signal" subspace  (eigenvalues > λ+)
      - a k_noise-dimensional  "noise"  subspace  (eigenvalues ≤ λ+)

    ``score(x)`` returns the chi-square statistic restricted to the
    noise subspace.  A value >> k_noise (or > threshold(alpha)) indicates
    that the chip's residual contains structure NOT explained by the
    learned common-mode patterns.
    """
    mean_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    noise_vecs_: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    noise_eigenvalues_: np.ndarray = field(default_factory=lambda: np.ones(0))
    k_noise_: int = 0
    k_signal_: int = 0
    sigma_sq_: float = 1.0
    lambda_plus_: float = 0.0
    fitted: bool = False

    def fit(self, X: np.ndarray, min_noise_rank: int = 2) -> None:
        """
        Fit the PCA model on baseline residual matrix X ∈ R^{n×p}.

        Parameters
        ----------
        X :
            (n_chips_total, input_length) array of per-chip residual vectors
            collected during the baseline period.
        min_noise_rank :
            Minimum number of noise-subspace dimensions to retain.
        """
        n, p = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        gamma = p / n
        cov = Xc.T @ Xc / max(n - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order

        sigma_sq = float(np.mean(eigenvalues))
        lambda_plus = sigma_sq * (1.0 + np.sqrt(gamma)) ** 2

        noise_mask = eigenvalues <= lambda_plus

        # Enforce minimum noise rank: keep at least min_noise_rank smallest eigs as noise
        if noise_mask.sum() < min_noise_rank:
            noise_mask = np.zeros(p, dtype=bool)
            noise_mask[np.argsort(eigenvalues)[:min_noise_rank]] = True

        # Refine sigma_sq from actual noise-subspace eigenvalues
        noise_eigs = eigenvalues[noise_mask]
        sigma_sq = float(np.mean(noise_eigs))

        self.mean_ = self.mean_
        self.noise_vecs_ = eigenvectors[:, noise_mask]           # (p, k_noise)
        self.noise_eigenvalues_ = np.maximum(noise_eigs, 1e-12)
        self.k_noise_ = int(noise_mask.sum())
        self.k_signal_ = p - self.k_noise_
        self.sigma_sq_ = max(sigma_sq, 1e-12)
        self.lambda_plus_ = lambda_plus
        self.fitted = True

    def score(self, x: np.ndarray) -> float:
        """
        Chi-square statistic in the noise subspace.

        score = Σᵢ (cᵢ²/λᵢ)  where cᵢ = noise_vecs_[:,i] · (x - mean_)

        Under the null this is chi²(k_noise).
        """
        if not self.fitted or self.k_noise_ == 0:
            return 0.0
        r = x - self.mean_
        c = self.noise_vecs_.T @ r        # (k_noise,)
        return float(np.sum(c ** 2 / self.noise_eigenvalues_))

    def threshold(self, alert_alpha: float) -> float:
        """
        Theoretical detection threshold: chi2.ppf(1 - alert_alpha, df=k_noise).

        No training-data calibration required — purely a function of the
        noise-subspace dimension and the desired false-alarm rate.
        """
        if self.k_noise_ == 0:
            return float("inf")
        return float(chi2.ppf(1.0 - alert_alpha, df=self.k_noise_))
