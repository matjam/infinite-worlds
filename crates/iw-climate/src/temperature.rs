//! Annual-mean temperature, seasonality, distance-to-ocean and glacial forcing.

use iw_core::{Phase, Planet, PlanetConfig};
use iw_mesh::Mesh;
use rayon::prelude::*;

/// Annual-mean temperature at the equator before any modifiers, degrees C.
pub const T_EQUATOR_C: f32 = 28.0;
/// Annual-mean temperature at the poles before any modifiers, degrees C.
pub const T_POLE_C: f32 = -25.0;
/// Environmental lapse rate applied above sea level, degrees C per metre.
pub const LAPSE_RATE_C_PER_M: f32 = 0.0065;
/// Distance-to-ocean saturation, in cells: beyond this a cell is fully
/// continental and the maritime terms stop changing.
pub const MAX_OCEAN_DISTANCE_CELLS: u8 = 8;
/// Reference temperature the ocean drags coastal cells toward, degrees C.
const MARITIME_REF_C: f32 = 14.0;
/// Fraction of the way to [`MARITIME_REF_C`] a fully maritime cell is dragged.
const MARITIME_GAIN: f32 = 0.10;
/// Seasonal amplitude scale, degrees C (see [`seasonal_amplitude_at`]).
pub const SEASONAL_BASE_C: f32 = 8.0;
/// Axial tilt the seasonal amplitude is calibrated against, degrees.
pub const REFERENCE_TILT_DEG: f32 = 23.4;
/// Length of one glacial cycle in the RecentPast phase, Myr (~100 kyr).
pub const GLACIAL_PERIOD_MYR: f64 = 0.1;
/// Temperature drop at a full glacial maximum at `glacial_intensity == 1`.
pub const GLACIAL_DEPTH_C: f32 = 6.0;

/// Continentality in 0..=1: 0 on the coast, 1 at or beyond
/// [`MAX_OCEAN_DISTANCE_CELLS`] cells from open ocean.
#[inline]
pub fn continentality(dist_cells: u8) -> f32 {
    dist_cells.min(MAX_OCEAN_DISTANCE_CELLS) as f32 / MAX_OCEAN_DISTANCE_CELLS as f32
}

/// Latitudinal insolation curve: `T_eq - (T_eq - T_pole) * sin^2(lat)`.
#[inline]
pub fn zonal_mean_c(lat_rad: f32) -> f32 {
    let s = lat_rad.sin();
    T_EQUATOR_C - (T_EQUATOR_C - T_POLE_C) * s * s
}

/// Global temperature anomaly from Milankovitch-style glacial cycles.
///
/// Zero outside the RecentPast phase. Inside it,
/// `dT = -6 C * glacial_intensity * (0.5 - 0.5*cos(2*pi*t_recent / 0.1 Myr))`,
/// where `t_recent` is time elapsed since the start of RecentPast — so the
/// phase begins at an interglacial (0 C) and reaches a glacial maximum of
/// `-6 C * intensity` every 100 kyr, with ~10 cycles over a default 2 Myr
/// phase. See the crate docs; `iw-surface` depends on this shape.
pub fn glacial_forcing_c(config: &PlanetConfig, time_myr: f64) -> f32 {
    let start: f64 = config.phase_durations_myr[..3].iter().sum();
    let end = start + config.phase_durations_myr[Phase::RecentPast.index()];
    if time_myr < start || time_myr > end {
        return 0.0;
    }
    let elapsed = time_myr - start;
    let theta = std::f64::consts::TAU * elapsed / GLACIAL_PERIOD_MYR;
    let cold = 0.5 - 0.5 * theta.cos();
    -GLACIAL_DEPTH_C * config.glacial_intensity * cold as f32
}

/// Annual-mean temperature of one cell, degrees C.
///
/// `surface_above_sea_m` is the height of the actual radiating surface (rock
/// plus ice) above sea level; negative values are clamped to 0 so ocean and
/// below-sea-level land share the sea-level baseline.
pub fn annual_mean_temperature_c(
    lat_rad: f32,
    surface_above_sea_m: f32,
    dist_cells: u8,
    config: &PlanetConfig,
    time_myr: f64,
) -> f32 {
    let base = zonal_mean_c(lat_rad);
    // Ocean heat capacity pulls maritime cells toward a mild reference: warmer
    // winters/annual means at high latitude, slightly cooler in the tropics.
    let maritime = MARITIME_GAIN * (1.0 - continentality(dist_cells)) * (MARITIME_REF_C - base);
    base + maritime - LAPSE_RATE_C_PER_M * surface_above_sea_m.max(0.0)
        + config.temperature_offset_c
        + glacial_forcing_c(config, time_myr)
}

/// Peak-to-mean seasonal swing of one cell, degrees C (always >= 0).
///
/// `amp = 8 C * sin^2(lat) * (tilt / 23.4) * (1 + min(dist, 8) / 4)`: driven by
/// axial tilt, zero at the equator, amplified up to 3x in continental
/// interiors. Summer = annual mean + amp, winter = annual mean - amp.
#[inline]
pub fn seasonal_amplitude_at(lat_rad: f32, axial_tilt_deg: f32, dist_cells: u8) -> f32 {
    let s = lat_rad.sin();
    let cont = 1.0 + dist_cells.min(MAX_OCEAN_DISTANCE_CELLS) as f32 / 4.0;
    SEASONAL_BASE_C * s * s * (axial_tilt_deg / REFERENCE_TILT_DEG).max(0.0) * cont
}

/// Cells of graph distance from each cell to the nearest open-ocean cell,
/// saturating at [`MAX_OCEAN_DISTANCE_CELLS`].
///
/// Multi-source BFS seeded from every ocean cell in ascending cell order, so
/// the result depends only on `(planet, mesh)`. A planet with no ocean at all
/// (before geology has solved a sea level) yields the saturated value
/// everywhere.
pub fn distance_to_ocean_cells(planet: &Planet, mesh: &Mesh) -> Vec<u8> {
    let n = planet.n_cells();
    let max = MAX_OCEAN_DISTANCE_CELLS;
    let mut dist = vec![max; n];
    let mut queue: Vec<u32> = Vec::with_capacity(n / 4);
    for c in 0..n as u32 {
        if planet.is_ocean(c) {
            dist[c as usize] = 0;
            queue.push(c);
        }
    }
    let mut head = 0usize;
    while head < queue.len() {
        let c = queue[head];
        head += 1;
        let nd = dist[c as usize] + 1;
        for &m in mesh.neighbors_of(c) {
            if dist[m as usize] > nd {
                dist[m as usize] = nd;
                if nd < max {
                    queue.push(m);
                }
            }
        }
    }
    dist
}

/// Seasonal amplitude for every cell, degrees C.
///
/// Recomputed from `(planet, mesh)` in O(n) — this is the helper `iw-biomes`
/// and `iw-surface` call; `Planet::temperature_c` carries only the annual mean.
pub fn seasonal_amplitude_c(planet: &Planet, mesh: &Mesh) -> Vec<f32> {
    let dist = distance_to_ocean_cells(planet, mesh);
    let tilt = planet.config.axial_tilt_deg;
    (0..planet.n_cells())
        .into_par_iter()
        .map(|i| seasonal_amplitude_at(mesh.latlon[i][0], tilt, dist[i]))
        .collect()
}

/// Summer and winter temperature endpoints for every cell, degrees C, as
/// `(summer, winter)` = annual mean +- [`seasonal_amplitude_c`].
pub fn seasonal_extremes_c(planet: &Planet, mesh: &Mesh) -> (Vec<f32>, Vec<f32>) {
    let amp = seasonal_amplitude_c(planet, mesh);
    let summer = planet
        .temperature_c
        .par_iter()
        .zip(amp.par_iter())
        .map(|(t, a)| t + a)
        .collect();
    let winter = planet
        .temperature_c
        .par_iter()
        .zip(amp.par_iter())
        .map(|(t, a)| t - a)
        .collect();
    (summer, winter)
}

/// Write the annual-mean temperature field into `planet.temperature_c`.
pub(crate) fn update(planet: &mut Planet, mesh: &Mesh, dist: &[u8]) {
    let config = planet.config.clone();
    let time_myr = planet.time_myr;
    let sea = planet.sea_level_m;
    let elev = &planet.elevation_m;
    let ice = &planet.ice_thickness_m;
    let latlon = &mesh.latlon;
    planet
        .temperature_c
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, t)| {
            let surface = elev[i] + ice[i] - sea;
            *t = annual_mean_temperature_c(latlon[i][0], surface, dist[i], &config, time_myr);
        });
}

/// FNV-1a hash of the land/sea mask. The distance-to-ocean BFS is only redone
/// when this changes, which keeps the cache a pure function of planet state
/// (a cached instance and a freshly constructed one always agree).
pub(crate) fn ocean_mask_hash(planet: &Planet) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in 0..planet.n_cells() as u32 {
        h ^= planet.is_ocean(c) as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
