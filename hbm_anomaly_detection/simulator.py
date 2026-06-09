"""
Synthetic data generator for HBM CoW bonding anomaly detection testing.

Generates realistic temperature and resistance time-series for various
normal and anomalous scenarios.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from .data_models import ChipData, WaferData


# ---------------------------------------------------------------------------
# Base profile generators
# ---------------------------------------------------------------------------

def _temp_profile(
    n_points: int,
    ramp_fraction: float = 0.25,
    hold_fraction: float = 0.50,
    t_start: float = 25.0,
    t_peak: float = 300.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Synthetic temperature profile: ramp-up → hold → cool-down.
    Returns array of shape (n_points,).
    """
    if rng is None:
        rng = np.random.default_rng()

    ramp_end = int(n_points * ramp_fraction)
    hold_end = int(n_points * (ramp_fraction + hold_fraction))

    profile = np.empty(n_points)
    # ramp-up
    profile[:ramp_end] = np.linspace(t_start, t_peak, ramp_end)
    # hold
    profile[ramp_end:hold_end] = t_peak
    # cool-down
    profile[hold_end:] = np.linspace(t_peak, t_start + 20, n_points - hold_end)
    return profile


def _resist_profile_peak_jumps(
    n_points: int,
    base_level: float = 0.5,
    peak_positions: tuple[float, float] = (0.35, 0.60),
    peak_height: float = 2.0,
    peak_width_frac: float = 0.04,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Resistance profile with two sharp peak jumps (rise-and-fall).
    Returns array of shape (n_points,).
    """
    if rng is None:
        rng = np.random.default_rng()

    profile = np.full(n_points, base_level)
    w = int(n_points * peak_width_frac)
    for pos in peak_positions:
        center = int(n_points * pos)
        start = max(0, center - w)
        end = min(n_points, center + w)
        x = np.linspace(-1, 1, end - start)
        profile[start:end] += peak_height * np.exp(-4 * x**2)
    return profile


def _resist_profile_ramp_hold(
    n_points: int,
    base_level: float = 0.2,
    jump_frac: float = 0.55,
    hold_level: float = 2.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Resistance profile: gradual ramp then sharp jump to hold level.
    Returns array of shape (n_points,).
    """
    if rng is None:
        rng = np.random.default_rng()

    profile = np.empty(n_points)
    jump_idx = int(n_points * jump_frac)
    profile[:jump_idx] = np.linspace(base_level, base_level * 1.5, jump_idx)
    profile[jump_idx:] = hold_level
    return profile


# ---------------------------------------------------------------------------
# Noise / anomaly injectors
# ---------------------------------------------------------------------------

def _add_noise(profile: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    return profile + rng.normal(0, noise_std, size=profile.shape)


def _inject_spike(
    series: np.ndarray,
    spike_pos_frac: float,
    spike_width_frac: float,
    spike_amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Insert a positive spike at spike_pos_frac of the series length."""
    n = len(series)
    out = series.copy()
    center = int(n * spike_pos_frac)
    w = max(1, int(n * spike_width_frac))
    start = max(0, center - w)
    end = min(n, center + w)
    out[start:end] += spike_amplitude
    return out


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
    noise_std_resist: float = 0.02,
    resist_mode: Literal["peaks", "ramp_hold"] = "peaks",
    rng: np.random.Generator | None = None,
) -> ChipData:
    """Generate a normal (no anomaly) chip."""
    if rng is None:
        rng = np.random.default_rng()

    temp = _add_noise(_temp_profile(n_points, rng=rng), noise_std_temp, rng)
    if resist_mode == "peaks":
        resist = _add_noise(_resist_profile_peak_jumps(n_points, rng=rng), noise_std_resist, rng)
    else:
        resist = _add_noise(_resist_profile_ramp_hold(n_points, rng=rng), noise_std_resist, rng)

    return ChipData(
        equipment_id=equipment_id,
        recipe=recipe,
        head=head,
        wafer_id=wafer_id,
        x=x,
        y=y,
        bond_order=bond_order,
        temp_series=temp,
        resist_series=resist,
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
    noise_std_resist: float = 0.02,
    rng: np.random.Generator | None = None,
) -> ChipData:
    """Generate a chip with elevated vibration (noise_multiplier > 1)."""
    if rng is None:
        rng = np.random.default_rng()

    temp = _add_noise(
        _temp_profile(n_points, rng=rng),
        noise_std_temp * noise_multiplier,
        rng,
    )
    resist = _add_noise(
        _resist_profile_peak_jumps(n_points, rng=rng),
        noise_std_resist * noise_multiplier,
        rng,
    )
    return ChipData(
        equipment_id=equipment_id,
        recipe=recipe,
        head=head,
        wafer_id=wafer_id,
        x=x,
        y=y,
        bond_order=bond_order,
        temp_series=temp,
        resist_series=resist,
    )


def make_spike_chip(
    equipment_id: str,
    recipe: str,
    head: str,
    wafer_id: str,
    x: int,
    y: int,
    bond_order: int,
    spike_amplitude: float = 5.0,
    n_points: int = 500,
    noise_std_resist: float = 0.02,
    rng: np.random.Generator | None = None,
) -> ChipData:
    """Generate a chip with a resistance spike anomaly."""
    if rng is None:
        rng = np.random.default_rng()

    temp = _add_noise(_temp_profile(n_points, rng=rng), 0.5, rng)
    resist_base = _resist_profile_peak_jumps(n_points, rng=rng)
    resist = _inject_spike(
        _add_noise(resist_base, noise_std_resist, rng),
        spike_pos_frac=rng.uniform(0.2, 0.8),
        spike_width_frac=0.03,
        spike_amplitude=spike_amplitude,
        rng=rng,
    )
    return ChipData(
        equipment_id=equipment_id,
        recipe=recipe,
        head=head,
        wafer_id=wafer_id,
        x=x,
        y=y,
        bond_order=bond_order,
        temp_series=temp,
        resist_series=resist,
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
    noise_std_resist: float = 0.02,
    resist_mode: Literal["peaks", "ramp_hold"] = "peaks",
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
            noise_std_resist=noise_std_resist,
            resist_mode=resist_mode,
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
    noise_std_resist: float = 0.02,
    rng: np.random.Generator | None = None,
) -> WaferData:
    """
    Wafer where noise std progressively increases in the latter half of
    the bonding sequence (intra-wafer vibration drift).
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
            noise_std_resist=noise_std_resist,
            rng=rng,
        )
        chips.append(chip)

    return WaferData(wafer_id=wafer_id, chips=chips)


def make_spike_wafer(
    equipment_id: str = "EQ01",
    recipe: str = "R001",
    head: str = "H1",
    wafer_id: str = "W_SPIKE",
    n_chips: int = 100,
    n_points: int = 500,
    spike_fraction: float = 0.15,
    spike_amplitude: float = 5.0,
    rng: np.random.Generator | None = None,
) -> WaferData:
    """Wafer where a fraction of chips have resistance spikes."""
    if rng is None:
        rng = np.random.default_rng()

    coords = _make_chip_grid(n_chips)
    n_spike = int(n_chips * spike_fraction)
    spike_indices = set(rng.choice(n_chips, size=n_spike, replace=False).tolist())
    chips = []

    for order, (x, y) in enumerate(coords):
        if order in spike_indices:
            chip = make_spike_chip(
                equipment_id=equipment_id,
                recipe=recipe,
                head=head,
                wafer_id=wafer_id,
                x=x,
                y=y,
                bond_order=order,
                spike_amplitude=spike_amplitude,
                n_points=n_points,
                rng=rng,
            )
        else:
            chip = make_normal_chip(
                equipment_id=equipment_id,
                recipe=recipe,
                head=head,
                wafer_id=wafer_id,
                x=x,
                y=y,
                bond_order=order,
                n_points=n_points,
                rng=rng,
            )
        chips.append(chip)

    return WaferData(wafer_id=wafer_id, chips=chips)


def make_bimodal_wafer(
    equipment_id: str = "EQ01",
    recipe: str = "R001",
    head: str = "H1",
    wafer_id: str = "W_BIMODAL",
    n_chips: int = 100,
    n_points: int = 500,
    coupling_broken: bool = False,
    temp_offset_group1: float = 10.0,
    noise_std_temp: float = 0.5,
    noise_std_resist: float = 0.02,
    rng: np.random.Generator | None = None,
) -> WaferData:
    """
    Wafer with bimodal resistance profiles (two types of chips).
    When coupling_broken=True, group 1 chips also get a temperature offset,
    simulating the coupling breakdown anomaly.
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = _make_chip_grid(n_chips)
    chips = []

    for order, (x, y) in enumerate(coords):
        # Alternate between two resistance modes
        if order % 2 == 0:
            # Group 0: peak-jump resistance
            resist_base = _resist_profile_peak_jumps(n_points, base_level=0.5, rng=rng)
            temp_base = _temp_profile(n_points, rng=rng)
        else:
            # Group 1: ramp-hold resistance (bimodal counterpart)
            resist_base = _resist_profile_ramp_hold(n_points, rng=rng)
            temp_base = _temp_profile(n_points, rng=rng)
            if coupling_broken:
                temp_base = temp_base + temp_offset_group1

        temp = _add_noise(temp_base, noise_std_temp, rng)
        resist = _add_noise(resist_base, noise_std_resist, rng)

        chip = ChipData(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=wafer_id,
            x=x,
            y=y,
            bond_order=order,
            temp_series=temp,
            resist_series=resist,
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
    noise_std_resist: float = 0.02,
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
            noise_std_resist=noise_std_resist,
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
    base_noise_std: float = 0.02,
    drift_multiplier_final: float = 3.0,
    seed: int = 42,
) -> list[WaferData]:
    """
    Sequence where the first n_normal wafers are normal and the following
    n_drift wafers show gradually increasing resistance noise (inter-wafer drift).
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
            noise_std_resist=base_noise_std,
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
            noise_std_resist=noise_std,
            rng=rng,
        ))

    return wafers


def make_coupling_breakdown_sequence(
    n_normal: int = 15,
    n_broken: int = 5,
    equipment_id: str = "EQ01",
    recipe: str = "R001",
    head: str = "H1",
    n_chips: int = 100,
    n_points: int = 500,
    temp_offset: float = 15.0,
    seed: int = 42,
) -> list[WaferData]:
    """
    Sequence of bimodal wafers where after n_normal wafers the
    temperature coupling breaks (group 1 gets a temp offset).
    """
    rng = np.random.default_rng(seed)
    wafers: list[WaferData] = []

    for i in range(n_normal):
        wafers.append(make_bimodal_wafer(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=f"W{i:03d}",
            n_chips=n_chips,
            n_points=n_points,
            coupling_broken=False,
            rng=rng,
        ))

    for j in range(n_broken):
        wafers.append(make_bimodal_wafer(
            equipment_id=equipment_id,
            recipe=recipe,
            head=head,
            wafer_id=f"W{n_normal + j:03d}",
            n_chips=n_chips,
            n_points=n_points,
            coupling_broken=True,
            temp_offset_group1=temp_offset,
            rng=rng,
        ))

    return wafers
