"""
Configuration parameters for temperature vibration-increase detection.

This module is the temperature-only subset of the full HBM CoW anomaly
detection system: resistance, bimodal clustering, spike, and coupling
detection are intentionally excluded.
"""
from dataclasses import dataclass, field


@dataclass
class ReferenceConfig:
    """Reference profile management parameters."""
    ewma_alpha: float = 0.1   # smoothing factor for reference update (0 = no update, 1 = replace)


@dataclass
class FeatureConfig:
    """Feature extraction parameters."""
    min_points: int = 10      # min time points required to compute features


@dataclass
class IntraWaferConfig:
    """Intra-wafer drift detection (CUSUM on chip sequence)."""
    cusum_k: float = 0.5      # CUSUM allowance (half of detectable shift in sigma units)
    cusum_h: float = 4.0      # CUSUM threshold (decision interval in sigma units)
    warmup_chips: int = 10    # chips used to establish within-wafer baseline


@dataclass
class InterWaferConfig:
    """Inter-wafer drift detection (EWMA on wafer-level statistics)."""
    ewma_lambda: float = 0.1   # EWMA smoothing for wafer-level roughness (small = more sensitive to slow drift)
    alert_sigma: float = 3.0   # alert if EWMA deviates > N sigma from baseline
    baseline_wafers: int = 10  # wafers used to establish (and then freeze) baseline mean/std


@dataclass
class AnomalyConfig:
    """Top-level configuration aggregating all sub-configs."""
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    intra_wafer: IntraWaferConfig = field(default_factory=IntraWaferConfig)
    inter_wafer: InterWaferConfig = field(default_factory=InterWaferConfig)


DEFAULT_CONFIG = AnomalyConfig()
