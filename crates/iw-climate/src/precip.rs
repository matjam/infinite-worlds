//! Moisture evaporation, downwind advection and rainout.

use iw_core::Planet;
use iw_mesh::Mesh;
use rayon::prelude::*;

use crate::geometry::MeshCache;

/// Evaporation from an open water surface at the reference temperature
/// ([`T_EQUATOR_C`](crate::T_EQUATOR_C)), mm/yr.
pub const EVAPORATION_MAX_MM_YR: f32 = 2000.0;
/// Fraction of a lake surface's evaporation relative to open ocean.
const LAKE_EVAPORATION_FACTOR: f32 = 0.8;
/// Ice thickness at which a water surface stops evaporating, metres.
const ICE_SEAL_M: f32 = 10.0;
/// Rainout fraction per reference cell traversed, before modifiers.
///
/// Calibration: at 0.10 an air parcel shed most of its load inside ~1000 km of
/// where it evaporated. Ocean basins here are several times that wide, so the
/// overwhelming majority of ocean evaporation rained straight back into the
/// ocean and land ran arid — deserts took 40%+ of the land surface against
/// Earth's ~19%. 0.07 stretches the rainout length to ~1600 km, which both
/// lands more of the ocean's moisture on the coast and carries it further
/// inland once it is there.
const BASE_RAINOUT: f32 = 0.07;
/// ITCZ convergence bonus at the equator.
const ITCZ_GAIN: f32 = 0.30;
/// Gaussian width of the ITCZ, degrees.
const ITCZ_WIDTH_DEG: f32 = 9.0;
/// Polar-front convergence bonus, centred on [`POLAR_FRONT_DEG`].
/// Calibration: moved equatorward from 55 deg and widened. The storm-track
/// rainfall maximum sits nearer 50 deg, and at 55/12 the wet belt cleared the
/// 34-50 deg temperate band almost entirely — that band was left between the
/// subtropical dry belt and the polar front with no convergence term at all, so
/// temperate forest never got the 800 mm/yr it needs.
const POLAR_FRONT_GAIN: f32 = 0.15;
const POLAR_FRONT_DEG: f32 = 50.0;
const POLAR_FRONT_WIDTH_DEG: f32 = 14.0;
/// Subtropical subsidence penalty, centred on [`SUBTROPICAL_DEG`]: descending
/// air in the horse latitudes converts almost nothing to rain.
/// Scaled down with [`BASE_RAINOUT`]: the penalty is meant to leave the horse
/// latitudes at the [`RAINOUT_MIN`] floor, not to hold a much wider band there.
const SUBTROPICAL_PENALTY: f32 = 0.055;
const SUBTROPICAL_DEG: f32 = 30.0;
const SUBTROPICAL_WIDTH_DEG: f32 = 11.0;
/// Uphill rise, in metres, that adds a full [`ORO_MAX`] of rainout.
const ORO_SCALE_M: f32 = 2500.0;
/// Cap on the orographic rainout bonus. Kept below 0.5 so a range sheds its
/// moisture over several cells instead of emptying the air in one.
const ORO_MAX: f32 = 0.40;
/// Rainout fraction is clamped into this range.
const RAINOUT_MIN: f32 = 0.02;
const RAINOUT_MAX: f32 = 0.90;
/// Advection sweeps at the reference subdivision level (6).
///
/// Calibration: 14 sweeps of a ~110 km cell is a 1550 km fetch, which is short
/// even against a single step's worth of atmosphere — water vapour's residence
/// time is about nine days and mid-latitude winds run 10 m/s, so a parcel
/// really does travel several thousand kilometres between evaporating and
/// raining. 24 sweeps (~2650 km) is still conservative and is what stops
/// continental interiors starving: they were coming out below the 250 mm/yr
/// desert threshold almost everywhere.
const SWEEPS_REF: f32 = 24.0;
/// Fraction of the previous step's rainfall a land cell returns to the air.
/// Continental interiors are fed by this recycling, not by ocean fetch: one
/// step's advection only reaches `sweeps` cells inland, and real interiors sit
/// much further from the sea than that.
///
/// Calibration: this is evapotranspiration as a fraction of what fell, and
/// Earth's land-wide ET/P is ~0.6, not 0.4. At 0.4 the recycling chain lost
/// 60% per advection length and continental interiors an advection length or
/// two from the coast fell below the 250 mm/yr desert threshold almost
/// everywhere.
///
/// **0.50 is a stability ceiling, not a free choice.** Land recycling is a
/// closed loop — precipitation feeds evapotranspiration feeds precipitation —
/// and the global water-balance rescale below adds gain wherever rainout is
/// efficient. Above ~0.5 the loop gain in the ITCZ band exceeds one and a
/// planet with no ocean at all grows rain out of nothing instead of drying
/// out (`tolerates_a_planet_with_no_ocean` catches exactly this). Anything
/// wetter has to come from a real reservoir, not from raising this.
const ET_FRACTION: f32 = 0.50;
/// Fraction of the previous precipitation field retained each step.
const PRECIP_INERTIA: f32 = 0.2;

/// Evaporation from a water surface, mm/yr.
///
/// Clausius-Clapeyron-ish: doubles per 10 C, saturating at
/// [`EVAPORATION_MAX_MM_YR`] at 28 C. Sea/lake ice seals the surface.
pub fn evaporation_mm_yr(temperature_c: f32, ice_m: f32, is_lake: bool) -> f32 {
    let sat = ((temperature_c - crate::T_EQUATOR_C) / 10.0)
        .exp2()
        .clamp(0.0, 1.0);
    let open = 1.0 - (ice_m / ICE_SEAL_M).clamp(0.0, 1.0);
    let kind = if is_lake {
        LAKE_EVAPORATION_FACTOR
    } else {
        1.0
    };
    EVAPORATION_MAX_MM_YR * sat * open * kind
}

/// Latitudinal part of the rainout efficiency: ITCZ and polar-front
/// convergence add, subtropical subsidence subtracts.
pub fn convergence_rainout(lat_rad: f32) -> f32 {
    let a = lat_rad.to_degrees().abs();
    let g = |gain: f32, centre: f32, width: f32| {
        let x = (a - centre) / width;
        gain * (-x * x).exp()
    };
    g(ITCZ_GAIN, 0.0, ITCZ_WIDTH_DEG) + g(POLAR_FRONT_GAIN, POLAR_FRONT_DEG, POLAR_FRONT_WIDTH_DEG)
        - g(SUBTROPICAL_PENALTY, SUBTROPICAL_DEG, SUBTROPICAL_WIDTH_DEG)
}

/// Number of advection sweeps and the width of one cell in reference (level 6)
/// cells. Coarse meshes take fewer, longer sweeps so the distance moisture
/// travels in kilometres is resolution-independent.
fn sweep_schedule(level: u8) -> (usize, f32) {
    let width = 2f32.powi(6i32 - level as i32);
    let sweeps = (SWEEPS_REF / width).round().clamp(5.0, 40.0) as usize;
    (sweeps, width)
}

/// Recompute `planet.precip_mm_yr`. Requires `planet.temperature_c` to be
/// current (this runs after the temperature pass).
pub(crate) fn update(planet: &mut Planet, mesh: &Mesh, cache: &MeshCache) {
    let n = planet.n_cells();
    let (sweeps, width) = sweep_schedule(mesh.level);
    let sea = planet.sea_level_m;

    // Land-surface height above sea level, including ice: the orographic term
    // measures the climb the air makes on its way downwind.
    let height: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| (planet.elevation_m[i] - sea).max(0.0) + planet.ice_thickness_m[i])
        .collect();

    // Moisture sources: open water evaporates, land transpires a fraction of
    // what fell on it last step (whichever of water supply and available
    // energy is the smaller). The recycling term divides out
    // `precip_multiplier` so that multiplier stays an exact linear gain on the
    // whole field instead of compounding through the feedback.
    let inv_mult = 1.0 / planet.config.precip_multiplier.max(1e-6);
    let evap: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (t, ice) = (planet.temperature_c[i], planet.ice_thickness_m[i]);
            let lake = planet.lake_depth_m[i] > 0.0;
            if planet.is_ocean(i as u32) || lake {
                evaporation_mm_yr(t, ice, lake)
            } else {
                let supply = ET_FRACTION * planet.precip_mm_yr[i] * inv_mult;
                supply.min(evaporation_mm_yr(t, ice, false))
            }
        })
        .collect();

    // Per-sweep rainout fraction. `r_ref` is per reference cell traversed;
    // rescaling by `width` keeps the km-scale rainout profile fixed.
    let rain_frac: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let a = mesh.neighbor_offsets[i] as usize;
            let b = mesh.neighbor_offsets[i + 1] as usize;
            let mut lift = 0.0f32;
            for (k, &j) in mesh.neighbors[a..b].iter().enumerate() {
                lift += cache.out_w[a + k] * (height[j as usize] - height[i]).max(0.0);
            }
            let oro = (lift / ORO_SCALE_M).min(ORO_MAX);
            let r_ref = (BASE_RAINOUT + oro + convergence_rainout(mesh.latlon[i][0]))
                .clamp(RAINOUT_MIN, RAINOUT_MAX);
            1.0 - (1.0 - r_ref).powf(width)
        })
        .collect();

    // Inject once, then advect: total rainout equals total evaporation minus
    // the moisture still airborne after the last sweep, which is discarded
    // (dry belts are exactly the places that never condense theirs) and
    // compensated by a global rescale below.
    let mut q = evap.clone();
    let mut carry = vec![0.0f32; n];
    let mut rain = vec![0.0f32; n];
    for _ in 0..sweeps {
        carry
            .par_iter_mut()
            .zip(rain.par_iter_mut())
            .zip(q.par_iter())
            .zip(rain_frac.par_iter())
            .for_each(|(((c, out), qi), r)| {
                *c = qi * (1.0 - r);
                *out += qi - *c;
            });
        q.par_iter_mut().enumerate().for_each(|(i, out)| {
            let a = mesh.neighbor_offsets[i] as usize;
            let b = mesh.neighbor_offsets[i + 1] as usize;
            let mut acc = carry[i] * cache.stay[i];
            for k in a..b {
                let j = mesh.neighbors[k] as usize;
                let rv = cache.rev[k] as usize;
                acc += carry[j] * cache.out_w[rv];
            }
            *out = acc;
        });
    }

    // Global water balance: precipitation integrates to evaporation. Serial
    // f64 accumulation - a parallel float reduction would not be deterministic.
    let mut evap_total = 0.0f64;
    let mut rain_total = 0.0f64;
    for i in 0..n {
        let area = mesh.areas_km2[i] as f64;
        evap_total += evap[i] as f64 * area;
        rain_total += rain[i] as f64 * area;
    }
    let scale = if rain_total > 0.0 {
        (evap_total / rain_total) as f32
    } else {
        1.0
    };

    let gain = scale * planet.config.precip_multiplier;
    planet
        .precip_mm_yr
        .par_iter_mut()
        .zip(rain.par_iter())
        .for_each(|(p, r)| {
            *p = ((1.0 - PRECIP_INERTIA) * (r * gain) + PRECIP_INERTIA * *p).max(0.0);
        });
}
