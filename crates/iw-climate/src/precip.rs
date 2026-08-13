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
/// Rainout multiplier over land (continental convection, roughness, orography
/// beyond the explicit lift term).
/// Calibration note: 1.7/0.72 were tuned while the advection operator was
/// silently broken (stale mesh cache after re-tessellation) — every knob was
/// compensating for dead transport. With transport fixed, the bias mostly
/// steers the global rain SHARE between land and sea (the water-balance
/// rescale pins the total), and the un-biased physics is the honest baseline
/// to calibrate from.
/// Re-tuned ON WORKING TRANSPORT (the 1.7/0.72 era was compensating for the
/// dead advection operator): at 1.0/1.0 a 28%-land world kept only a third
/// of the rain share Earth's land gets and two thirds of the land read
/// desert. 1.3/0.9 steers share without re-strangling the inland fetch.
const LAND_RAINOUT_BIAS: f32 = 1.3;
/// Rainout multiplier over open water: maritime air holds on to its moisture.
const OCEAN_RAINOUT_BIAS: f32 = 0.9;
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
const ORO_SCALE_M: f32 = 2000.0;
/// Cap on the orographic rainout bonus. Kept near 0.5 so a range sheds its
/// moisture over a few cells instead of emptying the air in one. (Applied
/// per EDGE, so fine meshes climb a range in many small steps and never see
/// the cap; only the coarse-cell case does.)
const ORO_MAX: f32 = 0.48;
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

/// Reference cell area: a level-6 Goldberg cell (~12,450 km^2, ~112 km wide).
/// All rainout constants are calibrated per one of these.
const REF_CELL_AREA_KM2: f32 = 4.0 * std::f32::consts::PI * 6371.0 * 6371.0 / 40_962.0;

/// Metric sweep schedule: how many hops moisture takes, and each CELL's width
/// in reference-cell units, derived from actual cell areas — never from a
/// subdivision level. On a terrain-sized Voronoi mesh (docs/voronoi-v2.md)
/// cell widths span an order of magnitude, so the width is per-cell: a hop
/// across a 15 km coastal cell must rain out ~1/7 of what a hop across a
/// 112 km reference cell does, or every interior turns to desert the moment
/// the mesh resolves fine coasts.
fn sweep_schedule(mesh: &Mesh) -> (usize, Vec<f32>) {
    let widths: Vec<f32> = mesh
        .areas_km2
        .iter()
        .map(|a| (a / REF_CELL_AREA_KM2).sqrt().clamp(0.05, 8.0))
        .collect();
    // Moisture advances one CELL per sweep, so the sweep count must serve the
    // FINE cells (coasts, mountains — exactly where the rain matters), not
    // the mean: sized by the mean, air makes landfall on a 20 km coastal cell
    // and dies of old sweeps a few hundred km inland, and the continent
    // interior reads 150 mm/yr on a planet whose land should average ~750.
    // Rainout is per-km (see below), so extra sweeps cannot over-carry
    // moisture across coarse regions — it decays on schedule regardless.
    let mut sorted = widths.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let p10 = sorted[sorted.len() / 10].max(0.05);
    let sweeps = (SWEEPS_REF / p10).round().clamp(5.0, 160.0) as usize;
    (sweeps, widths)
}

/// Recompute `planet.precip_mm_yr`. Requires `planet.temperature_c` to be
/// current (this runs after the temperature pass).
pub(crate) fn update(planet: &mut Planet, mesh: &Mesh, cache: &MeshCache) {
    let n = planet.n_cells();
    let (sweeps, widths) = sweep_schedule(mesh);
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
            // Land squeezes harder than open sea: continental convection and
            // surface roughness make moist air drop its load over land, and
            // the reduced ocean rainout lets maritime air survive the fetch.
            // Without the asymmetry most precipitation falls at sea and the
            // continents run 3x drier than Earth's land mean.
            let surface_bias = if planet.elevation_m[i] >= sea {
                LAND_RAINOUT_BIAS
            } else {
                OCEAN_RAINOUT_BIAS
            };
            // Two different physics, two different scalings: background
            // rainout is per-KILOMETRE travelled (exponent = cell width),
            // but the orographic squeeze is per-EDGE — climbing this cell's
            // rise sheds a fraction set by the CLIMB, however wide the cell
            // is. Width-scaling the climb term diluted mountains to nothing
            // on fine meshes and literally inverted the rain shadow.
            let r_base = ((BASE_RAINOUT + convergence_rainout(mesh.latlon[i][0])) * surface_bias)
                .clamp(RAINOUT_MIN, RAINOUT_MAX);
            let per_km = 1.0 - (1.0 - r_base).powf(widths[i]);
            let per_edge = (oro * surface_bias).min(ORO_MAX);
            (1.0 - (1.0 - per_km) * (1.0 - per_edge)).clamp(0.0, 0.95)
        })
        .collect();

    // Inject once, then advect — IN VOLUME (depth × area), not depth. Cells
    // differ in area by two orders of magnitude on a Voronoi mesh: passing
    // depths between them silently multiplies or destroys water at every size
    // boundary (a 200 km ocean cell handing its "1400 mm" to a 20 km coastal
    // cell would concentrate a hundredfold). Volumes conserve; the rained
    // depth divides the area back out at the end. Airborne remainder after
    // the last sweep is discarded (dry belts never condense theirs) and
    // compensated by the global rescale below.
    let mut q: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| evap[i] * mesh.areas_km2[i])
        .collect();
    let mut carry = vec![0.0f32; n];
    let mut rain_vol = vec![0.0f32; n];
    for _ in 0..sweeps {
        carry
            .par_iter_mut()
            .zip(rain_vol.par_iter_mut())
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
        evap_total += evap[i] as f64 * mesh.areas_km2[i] as f64;
        rain_total += rain_vol[i] as f64;
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
        .enumerate()
        .for_each(|(i, p)| {
            let depth = rain_vol[i] / mesh.areas_km2[i].max(1.0);
            *p = ((1.0 - PRECIP_INERTIA) * (depth * gain) + PRECIP_INERTIA * *p).max(0.0);
        });
}
