"""
Synthetic data generator for temperature vibration-increase detection testing.

Generates realistic temperature time-series for normal and anomalous
(intra-wafer and inter-wafer vibration increase) scenarios. Resistance is
intentionally excluded from this package.
"""
from __future__ import annotations

import numpy as np

from .data_models import ChipData, WaferData


# ---------------------------------------------------------------------------
# Base profile generator
# ---------------------------------------------------------------------------

def _temp_profile(
    n_points: int,
    ramp_fraction: float = 0.25,
    hold_fraction: float = 0.50,
    t_start: float = 25.0,
    t_peak: float = 300.0,
) -> np.ndarray:
    """
    Synthetic temperature profile: ramp-up → hold → cool-down.
    Returns array of shape (n_points,).
    """
    ramp_end = int(n_points * ramp_fraction)
    hold_end = int(n_points * (ramp_fraction + hold_fraction))

    profile = np.empty(n_points)
    profile[:ramp_end] = np.linspace(t_start, t_peak, ramp_end)
    profile[ramp_end:hold_end] = t_peak
    profile[hold_end:] = np.linspace(t_peak, t_start + 20, n_points - hold_end)
    return profile


def _add_noise(profile: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    return profile + rng.normal(0, noise_std, size=profile.shape)


# ---------------------------------------------------------------------------
# Chip-level generators
# ---------------------------------------------------------------------------

def make_normal_chip(
    equipment_id: str,
    recipe: str,
    head: str,
    wafer_id: str,
    x: int,
    y: int,
    bond_order: int,
    n_points: int = 500,
    noise_std_temp: float = 0.5,
    rng: np.random.Generator | None = None,
) -> ChipData:
    """Generate a normal (no anomaly) chip."""
    if rng is None:
        rng = np.random.default_rng()

    temp = _add_noise(_temp_profile(n_points), noise_std_temp, rng)

    return ChipData(
        equipment_id=equipment_id,
        recipe=recipe,
        head=head,
        wafer_id=wafer_id,
        x=x,
        y=y,
        bond_order=bond_order,
        temp_series=temp,
    )


def make_vibration_chip(
    equipment_id: str,
    recipe: str,
    head: str,
    wafer_id: str,
    x: int,
    y: int,
    bond_order: int,
    noise_multiplier: float = 3.0,
    n_points: int = 500,
    noise_std_temp: float = 0.5,
    rng: np.random.Generator | None = None,
) -> ChipData:
    """Generate a chip with elevated temperature vibration (noise_multiplier > 1)."""
    if rng is None:
        rng = np.random.default_rng()

    temp = _add_noise(_temp_profile(n_points), noise_std_temp * noise_multiplier, rng)

    return ChipData(
        equipment_id=equipment_id,
        recipe=recipe,
        head=head,
        wafer_id=wafer_id,
        x=x,
        y=y,
        bond_order=bond_order,
        temp_series=temp,
    )


# ---------------------------------------------------------------------------
# Wafer-level generators
# ---------------------------------------------------------------------------

def _make_chip_grid(n_chips: int) -> list[tuple[int, int]]:
    """Generate a simple square-ish chip grid for a wafer."""
    side = int(np.ceil(np.sqrt(n_chips)))
    coords = [(x, y) for y in range(side) for x in range(side)]
    return coords[:n_chips]


def make_normal_wafer(
    equipment_id: str = "EQ01",
    recipe: str = "R001",
    head: str = "H1",
    wafer_id: str = "W001",
    n_chips: int = 100,
    n_points: int = 500,
    noise_std_temp: float = 0.5,
    rng: np.random.Generator | None = None,
) -> WaferData:
    """Generate a normal wafer with no anomalies."""
    if rng is None:
        rng = np.random.default_rng()

    coords = _make_chip_grid(n_chips)
    chips = []
    for order, (x, y) in enumerate(coords):
        chip = make_normal_chip(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=wafer_id,
            x=x,
            y=y,
            bond_order=order,
            n_points=n_points,
            noise_std_temp=noise_std_temp,
            rng=rng,
        )
        chips.append(chip)

    return WaferData(wafer_id=wafer_id, chips=chips)


def make_intra_vibration_wafer(
    equipment_id: str = "EQ01",
    recipe: str = "R001",
    head: str = "H1",
    wafer_id: str = "W_INTRA",
    n_chips: int = 100,
    n_points: int = 500,
    onset_fraction: float = 0.5,
    max_noise_multiplier: float = 4.0,
    noise_std_temp: float = 0.5,
    rng: np.random.Generator | None = None,
) -> WaferData:
    """
    Wafer where temperature noise std progressively increases in the latter
    half of the bonding sequence (intra-wafer vibration drift).
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = _make_chip_grid(n_chips)
    onset_idx = int(n_chips * onset_fraction)
    chips = []

    for order, (x, y) in enumerate(coords):
        if order < onset_idx:
            multiplier = 1.0
        else:
            progress = (order - onset_idx) / max(1, n_chips - onset_idx)
            multiplier = 1.0 + (max_noise_multiplier - 1.0) * progress

        chip = make_vibration_chip(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=wafer_id,
            x=x,
            y=y,
            bond_order=order,
            noise_multiplier=multiplier,
            n_points=n_points,
            noise_std_temp=noise_std_temp,
            rng=rng,
        )
        chips.append(chip)

    return WaferData(wafer_id=wafer_id, chips=chips)


# ---------------------------------------------------------------------------
# Multi-wafer scenario generators
# ---------------------------------------------------------------------------

def make_normal_wafer_sequence(
    n_wafers: int = 20,
    equipment_id: str = "EQ01",
    recipe: str = "R001",
    head: str = "H1",
    n_chips: int = 100,
    n_points: int = 500,
    noise_std_temp: float = 0.5,
    seed: int = 42,
) -> list[WaferData]:
    """Sequence of normal wafers for baseline building."""
    rng = np.random.default_rng(seed)
    return [
        make_normal_wafer(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=f"W{i:03d}",
            n_chips=n_chips,
            n_points=n_points,
            noise_std_temp=noise_std_temp,
            rng=rng,
        )
        for i in range(n_wafers)
    ]


def make_inter_wafer_drift_sequence(
    n_normal: int = 15,
    n_drift: int = 10,
    equipment_id: str = "EQ01",
    recipe: str = "R001",
    head: str = "H1",
    n_chips: int = 100,
    n_points: int = 500,
    base_noise_std: float = 0.5,
    drift_multiplier_final: float = 3.0,
    seed: int = 42,
) -> list[WaferData]:
    """
    Sequence where the first n_normal wafers are normal and the following
    n_drift wafers show gradually increasing temperature noise
    (inter-wafer drift).
    """
    rng = np.random.default_rng(seed)
    wafers: list[WaferData] = []

    for i in range(n_normal):
        wafers.append(make_normal_wafer(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=f"W{i:03d}",
            n_chips=n_chips,
            n_points=n_points,
            noise_std_temp=base_noise_std,
            rng=rng,
        ))

    for j in range(n_drift):
        progress = (j + 1) / n_drift
        noise_std = base_noise_std * (1.0 + (drift_multiplier_final - 1.0) * progress)
        wafers.append(make_normal_wafer(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=f"W{n_normal + j:03d}",
            n_chips=n_chips,
            n_points=n_points,
            noise_std_temp=noise_std,
            rng=rng,
        ))

    return wafers
