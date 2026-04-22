from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from typing import Mapping, Sequence

EARTH_RADIUS_KM = 6371.0

Grid = Sequence[Sequence[float]]


@dataclass(frozen=True)
class CyclonePoint:
    lat: float
    lon: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1.0 - a))


def _grid_shape(grid: Grid) -> tuple[int, int]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return rows, cols


def _ensure_same_shape(*grids: Grid) -> None:
    base_shape = _grid_shape(grids[0])
    if base_shape[0] == 0 or base_shape[1] == 0:
        raise ValueError("Grid fields must be non-empty")

    for row in grids[0]:
        if len(row) != base_shape[1]:
            raise ValueError("Grid fields must be rectangular")

    for grid in grids[1:]:
        if _grid_shape(grid) != base_shape:
            raise ValueError("All input grids must have the same shape")
        for row in grid:
            if len(row) != base_shape[1]:
                raise ValueError("Grid fields must be rectangular")


def locate_level_center(
    geopotential_height: Grid,
    sea_level_pressure: Grid,
    lat: Grid,
    lon: Grid,
    guess: CyclonePoint,
    search_radius_km: float,
) -> CyclonePoint:
    """Find a level-specific center using equally weighted normalized geopotential and SLP minima."""
    _ensure_same_shape(geopotential_height, sea_level_pressure, lat, lon)

    candidates: list[tuple[float, float, float, float]] = []
    for i, lat_row in enumerate(lat):
        for j, lat_value in enumerate(lat_row):
            lon_value = lon[i][j]
            if _haversine_km(guess.lat, guess.lon, lat_value, lon_value) <= search_radius_km:
                candidates.append((geopotential_height[i][j], sea_level_pressure[i][j], lat_value, lon_value))

    if not candidates:
        raise ValueError("No grid points found inside the search radius")

    g_values = [c[0] for c in candidates]
    p_values = [c[1] for c in candidates]
    g_min, g_max = min(g_values), max(g_values)
    p_min, p_max = min(p_values), max(p_values)

    g_scale = g_max - g_min if g_max != g_min else 1.0
    p_scale = p_max - p_min if p_max != p_min else 1.0

    best = min(candidates, key=lambda c: ((c[0] - g_min) / g_scale) + ((c[1] - p_min) / p_scale))
    return CyclonePoint(lat=float(best[2]), lon=float(best[3]))


def weighted_mean_track(level_centers: Mapping[str, CyclonePoint], level_weights: Mapping[str, float]) -> CyclonePoint:
    total_weight = 0.0
    lat_acc = 0.0
    lon_acc = 0.0

    for level, center in level_centers.items():
        weight = float(level_weights.get(level, 1.0))
        total_weight += weight
        lat_acc += center.lat * weight
        lon_acc += center.lon * weight

    if total_weight == 0.0:
        raise ValueError("Total weight must be greater than zero")

    return CyclonePoint(lat=lat_acc / total_weight, lon=lon_acc / total_weight)


def track_cyclone(
    geopotential_by_level: Mapping[str, Grid],
    slp_by_level: Mapping[str, Grid],
    lat: Grid,
    lon: Grid,
    initial_guess: CyclonePoint,
    search_radius_km: float = 250.0,
    level_weights: Mapping[str, float] | None = None,
) -> CyclonePoint:
    if set(geopotential_by_level) != set(slp_by_level):
        mismatched_keys = sorted(set(geopotential_by_level).symmetric_difference(set(slp_by_level)))
        raise ValueError(f"Level keys for geopotential and pressure fields must match: {mismatched_keys}")

    level_centers: dict[str, CyclonePoint] = {}
    for level in geopotential_by_level:
        level_centers[level] = locate_level_center(
            geopotential_by_level[level],
            slp_by_level[level],
            lat,
            lon,
            initial_guess,
            search_radius_km,
        )

    return weighted_mean_track(level_centers, level_weights or {})
