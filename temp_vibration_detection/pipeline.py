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

import dataclasses
import json
import os
from datetime import datetime, timezone

import numpy as np

from .config import AnomalyConfig, DEFAULT_CONFIG, ReferenceConfig, FeatureConfig, IntraWaferConfig, InterWaferConfig
from .data_models import WaferData, WaferFeatures, ChipFeatures, AnomalyEvent, GroupKey
from .reference_manager import ReferenceManager, GroupReference
from .feature_extractor import extract_chip_features
from .detector.intra_wafer import detect_intra_wafer_drift
from .detector.inter_wafer import InterWaferTracker, InterWaferState


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

    def initialize_baseline(self, group_key: GroupKey, wafers: list[WaferData]) -> None:
        """
        Pre-warm the reference profile and inter-wafer baseline for one group
        using a set of known-good wafers from a historical period.

        The group's state is first reset (clean start), then each wafer is
        processed in order: the reference profile and EWMA tracker accumulate
        normally, but detected anomaly events are discarded.  After this call
        the pipeline is ready to detect anomalies on live production wafers.

        Parameters
        ----------
        group_key :
            The (equipment_id, recipe, head) group to initialize.
        wafers :
            Known-good wafers in chronological order.  Should contain at
            least ``inter_wafer.baseline_wafers`` wafers for the inter-wafer
            baseline to freeze; fewer wafers also work but the inter-wafer
            detector will continue accumulating from production data.
        """
        self.reset_baseline(group_key)
        for wafer in wafers:
            self.process_wafer(wafer, update_reference=True)

    def save_baseline(self, path: str) -> None:
        """
        Serialize all per-group baseline state to a JSON file.

        The file captures the temperature reference profiles and inter-wafer
        EWMA state for every group seen so far.  Pass the path to
        ``load_baseline`` to restore this state in a future run.

        Parameters
        ----------
        path :
            Destination file path (e.g. ``"baseline.json"``).  Parent
            directories must already exist.
        """
        cfg = self._config
        groups_out = []

        all_keys = set(self._ref_manager._groups) | set(self._inter_tracker._states)
        for key in sorted(all_keys):
            entry: dict = {
                "group_key": {
                    "equipment_id": key.equipment_id,
                    "recipe": key.recipe,
                    "head": key.head,
                },
                "reference": None,
                "inter_wafer_state": None,
            }

            ref = self._ref_manager.get(key)
            if ref is not None:
                entry["reference"] = {
                    "n_points": ref.n_points,
                    "temp_ref": ref.temp_ref.tolist() if len(ref.temp_ref) else [],
                    "n_wafers_seen": ref.n_wafers_seen,
                }

            st = self._inter_tracker._states.get(key)
            if st is not None:
                entry["inter_wafer_state"] = {
                    "ewma_value": st.ewma_value,
                    "baseline_mean": st.baseline_mean,
                    "baseline_std": st.baseline_std,
                    "n_samples": st.n_samples,
                    "m2": st.m2,
                    "initialized": st.initialized,
                    "baseline_frozen": st.baseline_frozen,
                }

            groups_out.append(entry)

        doc = {
            "version": "1.0",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "config": {
                "reference": dataclasses.asdict(cfg.reference),
                "feature": dataclasses.asdict(cfg.feature),
                "intra_wafer": dataclasses.asdict(cfg.intra_wafer),
                "inter_wafer": dataclasses.asdict(cfg.inter_wafer),
            },
            "groups": groups_out,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    def load_baseline(self, path: str) -> None:
        """
        Restore per-group baseline state from a JSON file created by
        ``save_baseline``.

        Existing in-memory state for loaded groups is replaced; groups that
        appear in memory but not in the file are left untouched.

        Parameters
        ----------
        path :
            JSON file written by a previous call to ``save_baseline``.
        """
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        for entry in doc.get("groups", []):
            gk_dict = entry["group_key"]
            key = GroupKey(
                equipment_id=gk_dict["equipment_id"],
                recipe=gk_dict["recipe"],
                head=gk_dict["head"],
            )

            ref_dict = entry.get("reference")
            if ref_dict is not None:
                ref = self._ref_manager.get_or_create(key)
                ref.n_points = ref_dict["n_points"]
                ref.temp_ref = np.array(ref_dict["temp_ref"], dtype=float)
                ref.n_wafers_seen = ref_dict["n_wafers_seen"]

            st_dict = entry.get("inter_wafer_state")
            if st_dict is not None:
                st = self._inter_tracker._get_state(key)
                st.ewma_value = st_dict["ewma_value"]
                st.baseline_mean = st_dict["baseline_mean"]
                st.baseline_std = st_dict["baseline_std"]
                st.n_samples = st_dict["n_samples"]
                st.m2 = st_dict["m2"]
                st.initialized = st_dict["initialized"]
                st.baseline_frozen = st_dict["baseline_frozen"]

    def process_wafer_sequence(
        self,
        wafers: list[WaferData],
    ) -> dict[str, list[AnomalyEvent]]:
        """Process a sequence of wafers and return events keyed by wafer_id."""
        results: dict[str, list[AnomalyEvent]] = {}
        for wafer in wafers:
            results[wafer.wafer_id] = self.process_wafer(wafer, update_reference=True)
        return results
