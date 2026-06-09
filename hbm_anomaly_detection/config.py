"""
Configuration parameters for HBM CoW bonding anomaly detection.
"""
from dataclasses import dataclass, field


@dataclass
class ReferenceConfig:
    """Reference profile management parameters."""
    ewma_alpha: float = 0.1          # smoothing factor for reference update (0 = no update, 1 = replace)
    n_history_wafers: int = 20       # max wafers kept for initialization
    bimodal_min_chips: int = 10      # min chips to attempt bimodal clustering
    bimodal_separation_threshold: float = 0.15  # min normalized separation to declare bimodal


@dataclass
class FeatureConfig:
    """Feature extraction parameters."""
    spike_threshold_sigma: float = 5.0   # sigma multiplier for spike detection per chip
    min_points: int = 10                 # min time points required to compute features


@dataclass
class IntraWaferConfig:
    """Intra-wafer drift detection (CUSUM/EWMA on chip sequence)."""
    ewma_lambda: float = 0.1        # EWMA smoothing (small = more sensitive to slow drift)
    cusum_k: float = 0.5            # CUSUM allowance (half of detectable shift in sigma units)
    cusum_h: float = 4.0            # CUSUM threshold (decision interval in sigma units)
    warmup_chips: int = 10          # chips to consume before signaling


@dataclass
class InterWaferConfig:
    """Inter-wafer drift detection (EWMA on wafer-level statistics)."""
    ewma_lambda: float = 0.1        # EWMA smoothing for wafer-level stats
    alert_sigma: float = 3.0        # alert if EWMA deviates > N sigma from baseline
    baseline_wafers: int = 10       # wafers used to establish initial baseline mean/std


@dataclass
class SpikeConfig:
    """Resistance spike detection parameters."""
    residual_sigma: float = 5.0         # per-chip spike threshold
    wafer_spike_rate_sigma: float = 3.0 # threshold for wafer-level spike rate trend


@dataclass
class CouplingConfig:
    """Bimodal resistance × temperature coupling detection."""
    coupling_sigma: float = 3.0         # threshold for coupling distance vs baseline
    baseline_wafers: int = 10           # wafers used to establish coupling distance baseline
    ewma_lambda: float = 0.2            # EWMA for coupling distance baseline


@dataclass
class AnomalyConfig:
    """Top-level configuration aggregating all sub-configs."""
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    intra_wafer: IntraWaferConfig = field(default_factory=IntraWaferConfig)
    inter_wafer: InterWaferConfig = field(default_factory=InterWaferConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)


DEFAULT_CONFIG = AnomalyConfig()
