"""
Main pipeline for HBM CoW bonding anomaly detection.

On each wafer track-out event, the pipeline:
  1. Updates reference profiles from the new wafer
  2. Extracts per-chip roughness and spike features
  3. Runs all detectors (intra-wafer, inter-wafer, spike, coupling)
  4. Returns a list of AnomalyEvent objects

Call pipeline.process_wafer() for each wafer in track-out order.
The pipeline maintains state across calls (reference profiles, EWMA
trackers) so results improve as more wafers are processed.
"""
from __future__ import annotations

import numpy as np

from .config import AnomalyConfig, DEFAULT_CONFIG
from .data_models import (
    WaferData,
    WaferFeatures,
    ChipFeatures,
    AnomalyEvent,
    GroupKey,
)
from .reference_manager import ReferenceManager
from .feature_extractor import extract_chip_features
from .detector.intra_wafer import detect_intra_wafer_drift
from .detector.inter_wafer import InterWaferTracker
from .detector.spike_detector import detect_chip_spikes, SpikeRateTracker
from .detector.coupling_detector import CouplingDetector
from .detector.multivariate_inter_wafer import MultivariateInterWaferTracker
from .detector.autoencoder import AutoencoderAnomalyDetector, extract_residual_vector


class AnomalyDetectionPipeline:
    """
    Stateful pipeline that processes wafers sequentially.

    Create one instance per equipment/recipe/head group,
    or share a single instance across multiple groups (the pipeline
    routes each wafer's chips to the correct group state).
    """

    def __init__(self, config: AnomalyConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._ref_manager = ReferenceManager(self._config.reference)
        self._inter_tracker = InterWaferTracker(self._config.inter_wafer)
        self._spike_rate_tracker = SpikeRateTracker(self._config.spike)
        self._coupling_detector = CouplingDetector(self._config.coupling)
        self._mv_inter_tracker = MultivariateInterWaferTracker(self._config.multivariate_inter_wafer)
        self._autoencoder_detector = AutoencoderAnomalyDetector(self._config.autoencoder)

    def process_wafer(
        self,
        wafer: WaferData,
        update_reference: bool = True,
    ) -> list[AnomalyEvent]:
        """
        Process one wafer and return all detected anomaly events.

        Parameters
        ----------
        wafer :
            Wafer data received at track-out.
        update_reference :
            If True (default), the reference profiles and roughness baselines
            are updated with this wafer's data after detection. Set to False
            for "burn-in" wafers that should not trigger alerts but should
            still contribute to the baseline.
        """
        if not wafer.chips:
            return []

        key = wafer.group_key
        if key is None:
            return []

        # --- Step 1: Ensure reference exists (initialize from first wafer) ---
        ref = self._ref_manager.get_or_create(key)
        if not ref.is_initialized():
            # First wafer: build reference only, skip detection to avoid false alarms
            # from an uninitialized roughness baseline.
            if update_reference:
                self._ref_manager.update(wafer)
            return []

        # --- Step 2: Extract per-chip features ---
        chip_features: list[ChipFeatures] = []
        residual_vectors: list[np.ndarray] = []
        for chip in wafer.chips:
            temp_ref, resist_ref, cluster = self._ref_manager.get_references(chip, ref)
            if len(temp_ref) == 0 or len(resist_ref) == 0:
                continue
            feat = extract_chip_features(
                chip, temp_ref, resist_ref, cluster, self._config.feature
            )
            chip_features.append(feat)
            residual_vectors.append(extract_residual_vector(
                chip, temp_ref, resist_ref, self._config.autoencoder.input_length
            ))

        if not chip_features:
            if update_reference:
                self._ref_manager.update(wafer)
            return []

        wafer_feats = WaferFeatures(wafer_id=wafer.wafer_id, chip_features=chip_features)

        # --- Step 3: Collect events from all detectors ---
        events: list[AnomalyEvent] = []

        # 3a. Intra-wafer drift (CUSUM on bond-order sequence)
        for channel in ("temp", "resist"):
            if channel == "temp":
                b_mean = ref.roughness_mean_temp
                b_std = ref.roughness_std_temp
            else:
                b_mean = ref.roughness_mean_resist
                b_std = ref.roughness_std_resist

            events.extend(detect_intra_wafer_drift(
                wafer_feats,
                key,
                baseline_mean=b_mean,
                baseline_std=b_std,
                channel=channel,
                config=self._config.intra_wafer,
            ))

        # 3b. Spike detection (per-chip and wafer-level rate)
        spike_events, spike_rate = detect_chip_spikes(
            wafer_feats, key, self._config.spike
        )
        events.extend(spike_events)
        events.extend(self._spike_rate_tracker.update_and_detect(
            key, wafer.wafer_id, spike_rate
        ))

        # 3c. Inter-wafer drift (EWMA on wafer-level roughness mean)
        summary = wafer_feats.summary_stats("resist")
        events.extend(self._inter_tracker.update_and_detect(
            key, wafer.wafer_id, summary["mean"], channel="resist"
        ))
        summary_temp = wafer_feats.summary_stats("temp")
        events.extend(self._inter_tracker.update_and_detect(
            key, wafer.wafer_id, summary_temp["mean"], channel="temp"
        ))

        # 3c-bis. Joint (temp, resist) inter-wafer drift via Hotelling's T^2 (MEWMA)
        events.extend(self._mv_inter_tracker.update_and_detect(
            key, wafer.wafer_id, np.array([summary_temp["mean"], summary["mean"]])
        ))

        # 3d. Bimodal coupling breakdown
        events.extend(self._coupling_detector.process_wafer(
            wafer,
            key,
            bimodal_separation_threshold=self._config.reference.bimodal_separation_threshold,
        ))

        # 3e. Deep autoencoder anomaly detection (joint temp+resist residual signature)
        events.extend(self._autoencoder_detector.process_wafer(
            key, wafer.wafer_id, [f.chip for f in chip_features], residual_vectors
        ))

        # --- Step 4: Update reference with this wafer (after detection) ---
        if update_reference:
            self._ref_manager.update(wafer)
            # Update roughness baseline with this wafer's stats
            self._ref_manager.update_roughness_baseline(
                ref,
                wafer_mean_roughness_temp=float(
                    np.mean([f.temp_roughness for f in chip_features])
                ),
                wafer_mean_roughness_resist=float(
                    np.mean([f.resist_roughness for f in chip_features])
                ),
            )

        return events

    def process_wafer_sequence(
        self,
        wafers: list[WaferData],
        n_warmup: int = 5,
    ) -> dict[str, list[AnomalyEvent]]:
        """
        Process a sequence of wafers and return events keyed by wafer_id.

        The first n_warmup wafers are processed without raising alerts
        (update_reference=True but detector states are still warming up).
        """
        results: dict[str, list[AnomalyEvent]] = {}
        for wafer in wafers:
            events = self.process_wafer(wafer, update_reference=True)
            results[wafer.wafer_id] = events
        return results
