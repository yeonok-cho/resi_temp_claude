"""
Main pipeline for temperature vibration-increase detection.

On each wafer track-out event, the pipeline:
  1. Updates the temperature reference profile from the new wafer
  2. Extracts per-chip temperature roughness features
  3. Runs intra-wafer (CUSUM) and inter-wafer (EWMA) drift detectors
  4. Returns a list of AnomalyEvent objects

Call pipeline.process_wafer() for each wafer in track-out order.
The pipeline maintains state across calls (reference profile, EWMA
trackers) so results improve as more wafers are processed.
"""
from __future__ import annotations

import numpy as np

from .config import AnomalyConfig, DEFAULT_CONFIG
from .data_models import WaferData, WaferFeatures, ChipFeatures, AnomalyEvent, GroupKey
from .reference_manager import ReferenceManager
from .feature_extractor import extract_chip_features
from .detector.intra_wafer import detect_intra_wafer_drift
from .detector.inter_wafer import InterWaferTracker


class TempVibrationPipeline:
    """
    Stateful pipeline that processes wafers sequentially.

    A single instance can handle multiple (equipment, recipe, head) groups —
    the pipeline routes each wafer's chips to the correct group state via
    its group_key.
    """

    def __init__(self, config: AnomalyConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG
        self._ref_manager = ReferenceManager(self._config.reference)
        self._inter_tracker = InterWaferTracker(self._config.inter_wafer)

    def reset_baseline(self, group_key: GroupKey) -> None:
        """
        Reset all baseline state for a group after a PM (Preventive
        Maintenance) event.

        The temperature reference profile and the inter-wafer EWMA baseline
        are both discarded, so they are rebuilt from scratch starting with
        the next wafer for this group (the inter-wafer baseline re-freezes
        after ``inter_wafer.baseline_wafers`` wafers).

        Intra-wafer drift detection is self-calibrating and unaffected — it
        keeps running on every wafer, including during the post-PM baseline
        rebuild period.
        """
        self._ref_manager.reset(group_key)
        self._inter_tracker.reset(group_key)

    def process_wafer(
        self,
        wafer: WaferData,
        update_reference: bool = True,
        pm_event: bool = False,
    ) -> list[AnomalyEvent]:
        """
        Process one wafer and return all detected anomaly events.

        Parameters
        ----------
        wafer :
            Wafer data received at track-out.
        update_reference :
            If True (default), the reference profile is updated with this
            wafer's data after detection.
        pm_event :
            If True, this wafer is the first one after a Preventive
            Maintenance event: the group's baseline state is reset (see
            ``reset_baseline``) before processing, so this wafer is treated
            as a cold-start wafer (reference rebuild only, no detection).
        """
        if not wafer.chips:
            return []

        key = wafer.group_key
        if key is None:
            return []

        if pm_event:
            self.reset_baseline(key)

        # --- Step 1: Ensure reference exists (initialize from first wafer) ---
        ref = self._ref_manager.get_or_create(key)
        if not ref.is_initialized():
            # First wafer: build reference only, skip detection.
            if update_reference:
                self._ref_manager.update(wafer)
            return []

        # --- Step 2: Extract per-chip temperature roughness features ---
        chip_features: list[ChipFeatures] = []
        for chip in wafer.chips:
            feat = extract_chip_features(chip, ref.temp_ref, self._config.feature)
            chip_features.append(feat)

        if not chip_features:
            if update_reference:
                self._ref_manager.update(wafer)
            return []

        wafer_feats = WaferFeatures(wafer_id=wafer.wafer_id, chip_features=chip_features)

        # --- Step 3: Run detectors ---
        events: list[AnomalyEvent] = []

        # 3a. Intra-wafer drift (CUSUM on bond-order sequence)
        events.extend(detect_intra_wafer_drift(
            wafer_feats, key, config=self._config.intra_wafer
        ))

        # 3b. Inter-wafer drift (EWMA on wafer-level roughness mean)
        summary = wafer_feats.summary_stats()
        events.extend(self._inter_tracker.update_and_detect(
            key, wafer.wafer_id, summary["mean"]
        ))

        # --- Step 4: Update reference with this wafer (after detection) ---
        if update_reference:
            self._ref_manager.update(wafer)

        return events

    def process_wafer_sequence(
        self,
        wafers: list[WaferData],
    ) -> dict[str, list[AnomalyEvent]]:
        """Process a sequence of wafers and return events keyed by wafer_id."""
        results: dict[str, list[AnomalyEvent]] = {}
        for wafer in wafers:
            results[wafer.wafer_id] = self.process_wafer(wafer, update_reference=True)
        return results
