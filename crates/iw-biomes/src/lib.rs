//! Biome classification: the last process in the pipeline (DESIGN.md §8).
//!
//! Every land cell is classified from temperature (annual mean plus summer/
//! winter extremes), precipitation, elevation above sea level, coastal
//! distance and local water state, using a Whittaker temperature x
//! precipitation lookup with a handful of special-case overrides evaluated
//! first. Water and ice are classified before any of that: ice sheet, then
//! ocean, then lake.
//!
//! # Model
//!
//! Water/ice, in priority order:
//! 1. `ice_thickness_m > `[`ICE_SHEET_THICKNESS_M`] -> [`Biome::IceSheet`]
//!    (checked before ocean/lake, so a frozen sea or a frozen lake is still
//!    ice sheet).
//! 2. [`Planet::is_ocean`] -> [`Biome::Ocean`] (below sea level and not a
//!    lake; a below-sea-level closed basin with standing water is a lake,
//!    not an inlet of the sea, matching the convention `iw-climate` already
//!    uses for its distance-to-ocean field).
//! 3. `lake_depth_m > `[`LAKE_DEPTH_M`] -> [`Biome::Lake`].
//!
//! Land, overrides in priority order (first match wins), then the Whittaker
//! core for whatever is left:
//! 1. **Mangrove** — coastal (`dist_to_ocean <= `[`MANGROVE_MAX_DIST_CELLS`]),
//!    tropical (`winter_c >= `[`MANGROVE_MIN_WINTER_C`]), low-lying
//!    (`elevation_above_sea_m < `[`MANGROVE_MAX_ELEV_M`]), and muddy
//!    (`precip_mm_yr >= `[`MANGROVE_MIN_PRECIP_MM`] or a river-mouth proxy —
//!    see [`water_flux_high_threshold`]).
//! 2. **Flooded grassland** — lowland (`elevation_above_sea_m < `
//!    [`FLOODED_MAX_ELEV_M`]), a high-water-table proxy (lake-adjacent, or
//!    the same river-mouth flux proxy as mangrove), and warm
//!    (`temperature_c >= `[`FLOODED_MIN_TEMP_C`]).
//! 3. **Montane grassland** — high (`elevation_above_sea_m > `
//!    [`MONTANE_MIN_ELEV_M`]), low-latitude (`|lat| < `
//!    [`MONTANE_MAX_LAT_DEG`]), and not desert-dry
//!    (`precip_mm_yr >= `[`MONTANE_MIN_PRECIP_MM`]).
//! 4. **Mediterranean** — an honest approximation: real Mediterranean
//!    climates are winter-wet/summer-dry west coasts at 30-45 deg latitude,
//!    which needs prevailing-wind-relative coast orientation this model
//!    doesn't carry per cell. We substitute "coastal-ish"
//!    (`dist_to_ocean <= `[`MED_MAX_DIST_CELLS`]) in the right latitude band
//!    with a temperate, moderate-rainfall envelope
//!    (`temperature_c` in [[`MED_MIN_TEMP_C`], [`MED_MAX_TEMP_C`]],
//!    `precip_mm_yr` in [[`MED_MIN_PRECIP_MM`], [`MED_MAX_PRECIP_MM`]]) as a
//!    stand-in for the winter-wet signature. This overclassifies moderate
//!    east coasts too; acceptable per DESIGN.md §2 (realistic-looking, not
//!    scientifically exact).
//! 5. **Tundra** — `summer_c < `[`TUNDRA_MAX_SUMMER_C`] (the classic Koppen
//!    warmest-month-below-10C line); kept as a standalone override (not just
//!    the Whittaker `T < -5` backstop) because plenty of tundra has an
//!    annual mean above -5 C once maritime moderation is included.
//!
//! Whittaker core (land not caught by an override above), see [`whittaker`]
//! for the exact thresholds.
//!
//! # River-mouth / high-water-table proxy
//!
//! `iw-surface` (WP7) owns `water_flux_m3_yr`; this crate has no notion of
//! what counts as "a lot" of discharge in absolute terms since it scales
//! with upstream catchment area and mesh resolution. [`water_flux_high_threshold`]
//! computes the 90th percentile of positive flux across the planet once per
//! step and cells at or above it are treated as "high flux" — a
//! self-calibrating stand-in for "river mouth" wherever this planet's rivers
//! actually are.

#![warn(missing_docs)]

use iw_climate::{distance_to_ocean_cells, seasonal_extremes_c};
use iw_core::{Biome, Planet, Process, StepCtx};
use iw_mesh::Mesh;
use rayon::prelude::*;

// --- water / ice -----------------------------------------------------------

/// Ice thickness above which a cell is classified [`Biome::IceSheet`],
/// meters. Checked before ocean/lake, so this also covers sea ice and frozen
/// lakes.
pub const ICE_SHEET_THICKNESS_M: f32 = 10.0;
/// Standing water depth above which a land cell is classified
/// [`Biome::Lake`], meters. Matches the definition `iw-surface` uses for
/// `lake_depth_m`.
pub const LAKE_DEPTH_M: f32 = 0.5;

// --- mangrove ----------------------------------------------------------

/// Maximum graph distance to open ocean for the mangrove override, cells.
pub const MANGROVE_MAX_DIST_CELLS: u8 = 1;
/// Minimum winter temperature extreme for the mangrove override, degrees C.
pub const MANGROVE_MIN_WINTER_C: f32 = 20.0;
/// Maximum elevation above sea level for the mangrove override, meters.
pub const MANGROVE_MAX_ELEV_M: f32 = 50.0;
/// Minimum annual precipitation that counts as "muddy" for the mangrove
/// override on its own (independent of the river-mouth flux proxy), mm/yr.
pub const MANGROVE_MIN_PRECIP_MM: f32 = 800.0;

// --- flooded grassland ---------------------------------------------------

/// Maximum elevation above sea level for the flooded-grassland override,
/// meters.
pub const FLOODED_MAX_ELEV_M: f32 = 200.0;
/// Minimum annual-mean temperature for the flooded-grassland override,
/// degrees C.
pub const FLOODED_MIN_TEMP_C: f32 = 10.0;

// --- montane grassland ---------------------------------------------------

/// Minimum elevation above sea level for the montane override, meters.
pub const MONTANE_MIN_ELEV_M: f32 = 2500.0;
/// Maximum absolute latitude for the montane override, degrees.
pub const MONTANE_MAX_LAT_DEG: f32 = 35.0;
/// Minimum annual precipitation for the montane override (excludes cold/dry
/// desert peaks), mm/yr.
pub const MONTANE_MIN_PRECIP_MM: f32 = 250.0;

// --- mediterranean ---------------------------------------------------------

/// Minimum absolute latitude for the Mediterranean override, degrees.
pub const MED_MIN_LAT_DEG: f32 = 30.0;
/// Maximum absolute latitude for the Mediterranean override, degrees.
pub const MED_MAX_LAT_DEG: f32 = 45.0;
/// Maximum graph distance to open ocean for the Mediterranean override
/// ("coastal-ish" stand-in for "west coast"; see module docs), cells.
pub const MED_MAX_DIST_CELLS: u8 = 4;
/// Minimum annual-mean temperature for the Mediterranean override, degrees C.
pub const MED_MIN_TEMP_C: f32 = 8.0;
/// Maximum annual-mean temperature for the Mediterranean override, degrees C.
pub const MED_MAX_TEMP_C: f32 = 18.0;
/// Minimum annual precipitation for the Mediterranean override, mm/yr.
pub const MED_MIN_PRECIP_MM: f32 = 300.0;
/// Maximum annual precipitation for the Mediterranean override, mm/yr.
pub const MED_MAX_PRECIP_MM: f32 = 900.0;

// --- tundra ------------------------------------------------------------

/// Maximum summer temperature extreme for the tundra override, degrees C
/// (the classic Koppen warmest-month-below-10C line).
pub const TUNDRA_MAX_SUMMER_C: f32 = 10.0;

// --- Whittaker core: tropical (T >= TROPICAL_MIN_T_C) -----------------

/// Annual-mean temperature at/above which a cell is "tropical" for the
/// Whittaker core, degrees C.
pub const TROPICAL_MIN_T_C: f32 = 20.0;
/// Precipitation at/above which tropical land is
/// [`Biome::TropicalMoistBroadleaf`], mm/yr.
pub const TROPICAL_MOIST_MIN_PRECIP_MM: f32 = 1800.0;
/// Precipitation at/above which tropical land enters the dry-broadleaf /
/// moist-broadleaf split band, mm/yr.
pub const TROPICAL_DRY_MIN_PRECIP_MM: f32 = 900.0;
/// Seasonal swing (summer - winter) at/above which the dry-broadleaf /
/// moist-broadleaf split band is called dry (a pronounced dry season),
/// degrees C.
pub const TROPICAL_DRY_SEASON_AMPLITUDE_C: f32 = 6.0;
/// Precipitation below which the dry-broadleaf / moist-broadleaf split band
/// is called dry regardless of seasonality, mm/yr.
pub const TROPICAL_DRY_SEASON_PRECIP_MM: f32 = 1400.0;
/// Precipitation at/above which tropical land enters the
/// conifer/grassland split band, mm/yr.
pub const TROPICAL_CONIFER_MIN_PRECIP_MM: f32 = 500.0;
/// Elevation above which the conifer/grassland split band is called
/// cooler-highland conifer instead of grassland, meters.
pub const TROPICAL_HIGHLAND_ELEV_M: f32 = 800.0;
/// Precipitation at/above which (low-elevation) tropical land is
/// [`Biome::TropicalGrassland`] rather than [`Biome::Desert`], mm/yr.
pub const TROPICAL_GRASSLAND_MIN_PRECIP_MM: f32 = 250.0;

// --- Whittaker core: temperate (TEMPERATE_MIN_T_C <= T < TROPICAL_MIN_T_C) -

/// Annual-mean temperature at/above which a cell is "temperate" for the
/// Whittaker core, degrees C.
pub const TEMPERATE_MIN_T_C: f32 = 5.0;
/// Precipitation at/above which temperate land is
/// [`Biome::TemperateBroadleaf`], mm/yr.
pub const TEMPERATE_BROADLEAF_MIN_PRECIP_MM: f32 = 1400.0;
/// Precipitation at/above which temperate land enters the
/// broadleaf/conifer split band, mm/yr.
pub const TEMPERATE_MIXED_MIN_PRECIP_MM: f32 = 800.0;
/// Winter temperature extreme above which the broadleaf/conifer split band
/// is called broadleaf (mild winters), degrees C.
pub const TEMPERATE_BROADLEAF_MIN_WINTER_C: f32 = -2.0;
/// Precipitation at/above which temperate land enters the
/// conifer/grassland split band, mm/yr.
pub const TEMPERATE_CONIFER_MIN_PRECIP_MM: f32 = 400.0;
/// Summer temperature extreme below which the conifer/grassland split band
/// is called conifer (cool summers), degrees C.
pub const TEMPERATE_CONIFER_MAX_SUMMER_C: f32 = 22.0;
/// Precipitation at/above which (low-precipitation) temperate land is
/// [`Biome::TemperateGrassland`] rather than [`Biome::Desert`], mm/yr.
pub const TEMPERATE_GRASSLAND_MIN_PRECIP_MM: f32 = 250.0;

// --- Whittaker core: boreal (BOREAL_MIN_T_C <= T < TEMPERATE_MIN_T_C) ------

/// Annual-mean temperature at/above which a cell is "boreal" for the
/// Whittaker core, degrees C. Below this, land is [`Biome::Tundra`] as a
/// backstop (most such cells are already caught by the standalone tundra
/// override above).
pub const BOREAL_MIN_T_C: f32 = -5.0;
/// Precipitation at/above which boreal land is [`Biome::BorealTaiga`] rather
/// than [`Biome::Desert`] (WWF classifies cold, dry land as xeric), mm/yr.
pub const BOREAL_MIN_PRECIP_MM: f32 = 250.0;

// --- river-mouth / high-water-table proxy -----------------------------

/// Percentile of positive `water_flux_m3_yr` treated as "high flux" (see
/// module docs and [`water_flux_high_threshold`]).
pub const WATER_FLUX_HIGH_PERCENTILE: f32 = 0.9;

/// The 90th-percentile (see [`WATER_FLUX_HIGH_PERCENTILE`]) positive value of
/// `water_flux_m3_yr` across a planet, used as the "river mouth / high water
/// table" threshold for the mangrove and flooded-grassland overrides.
///
/// Self-calibrating: `water_flux_m3_yr` scales with catchment area and mesh
/// resolution, so an absolute threshold would misfire across subdivision
/// levels or before `iw-surface` has grown any real drainage network. A
/// planet with no positive flux anywhere (e.g. before `iw-surface` first
/// runs) returns `f32::INFINITY`, so the proxy simply never fires.
///
/// O(n) average (`slice::select_nth_unstable_by`), single-threaded and
/// order-preserving over `flux`, so the result is deterministic regardless
/// of thread count.
pub fn water_flux_high_threshold(flux: &[f32]) -> f32 {
    let mut positive: Vec<f32> = flux.iter().copied().filter(|f| *f > 0.0).collect();
    if positive.is_empty() {
        return f32::INFINITY;
    }
    let idx =
        (((positive.len() as f32) * WATER_FLUX_HIGH_PERCENTILE) as usize).min(positive.len() - 1);
    let (_, &mut val, _) = positive.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    val
}

// --- classification ------------------------------------------------------

/// Inputs to biome classification for a single cell, decoupled from
/// [`Planet`] so the classifier is a pure function exercisable in synthetic
/// tests without a full mesh/simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellInputs {
    /// Annual-mean temperature, degrees C (`Planet::temperature_c`).
    pub temperature_c: f32,
    /// Summer temperature extreme, degrees C
    /// (`iw_climate::seasonal_extremes_c`).
    pub summer_c: f32,
    /// Winter temperature extreme, degrees C
    /// (`iw_climate::seasonal_extremes_c`).
    pub winter_c: f32,
    /// Annual precipitation, mm/yr (`Planet::precip_mm_yr`).
    pub precip_mm_yr: f32,
    /// Ground height above sea level, meters (negative if submerged).
    pub elevation_above_sea_m: f32,
    /// Latitude, radians.
    pub lat_rad: f32,
    /// Graph distance to the nearest open-ocean cell, saturating a few cells
    /// inland (`iw_climate::distance_to_ocean_cells`).
    pub dist_to_ocean: u8,
    /// True if any neighbor cell is a lake (`lake_depth_m > `
    /// [`LAKE_DEPTH_M`]).
    pub lake_adjacent: bool,
    /// True if this cell's `water_flux_m3_yr` is at or above the per-step
    /// high-flux threshold; see [`water_flux_high_threshold`].
    pub water_flux_high: bool,
    /// Ice thickness, meters.
    pub ice_thickness_m: f32,
    /// True if this cell is open ocean (`Planet::is_ocean`).
    pub is_ocean: bool,
    /// Standing lake water depth, meters (0 if not a lake).
    pub lake_depth_m: f32,
}

/// Whittaker temperature x precipitation core for land not caught by any
/// override, per DESIGN.md §8. See the module docs for the constants each
/// branch uses.
fn whittaker(t: f32, summer_c: f32, winter_c: f32, p: f32, elev_above_sea_m: f32) -> Biome {
    if t >= TROPICAL_MIN_T_C {
        if p >= TROPICAL_MOIST_MIN_PRECIP_MM {
            Biome::TropicalMoistBroadleaf
        } else if p >= TROPICAL_DRY_MIN_PRECIP_MM {
            let dry_season = (summer_c - winter_c) >= TROPICAL_DRY_SEASON_AMPLITUDE_C
                || p < TROPICAL_DRY_SEASON_PRECIP_MM;
            if dry_season {
                Biome::TropicalDryBroadleaf
            } else {
                Biome::TropicalMoistBroadleaf
            }
        } else if p >= TROPICAL_CONIFER_MIN_PRECIP_MM {
            if elev_above_sea_m > TROPICAL_HIGHLAND_ELEV_M {
                Biome::TropicalConifer
            } else {
                Biome::TropicalGrassland
            }
        } else if p >= TROPICAL_GRASSLAND_MIN_PRECIP_MM {
            Biome::TropicalGrassland
        } else {
            Biome::Desert
        }
    } else if t >= TEMPERATE_MIN_T_C {
        if p >= TEMPERATE_BROADLEAF_MIN_PRECIP_MM {
            Biome::TemperateBroadleaf
        } else if p >= TEMPERATE_MIXED_MIN_PRECIP_MM {
            if winter_c > TEMPERATE_BROADLEAF_MIN_WINTER_C {
                Biome::TemperateBroadleaf
            } else {
                Biome::TemperateConifer
            }
        } else if p >= TEMPERATE_CONIFER_MIN_PRECIP_MM {
            if summer_c < TEMPERATE_CONIFER_MAX_SUMMER_C {
                Biome::TemperateConifer
            } else {
                Biome::TemperateGrassland
            }
        } else if p >= TEMPERATE_GRASSLAND_MIN_PRECIP_MM {
            Biome::TemperateGrassland
        } else {
            Biome::Desert
        }
    } else if t >= BOREAL_MIN_T_C {
        if p >= BOREAL_MIN_PRECIP_MM {
            Biome::BorealTaiga
        } else {
            Biome::Desert
        }
    } else {
        Biome::Tundra
    }
}

/// Classify a land cell (water/ice already ruled out): overrides in priority
/// order, then the Whittaker core. See the module docs for the full table.
fn classify_land(inputs: &CellInputs) -> Biome {
    let lat_deg = inputs.lat_rad.to_degrees().abs();

    // 1. Mangrove: coastal, tropical, low-lying, muddy.
    if inputs.dist_to_ocean <= MANGROVE_MAX_DIST_CELLS
        && inputs.winter_c >= MANGROVE_MIN_WINTER_C
        && inputs.elevation_above_sea_m < MANGROVE_MAX_ELEV_M
        && (inputs.precip_mm_yr >= MANGROVE_MIN_PRECIP_MM || inputs.water_flux_high)
    {
        return Biome::Mangrove;
    }

    // 2. Flooded grassland: lowland, high water table, warm.
    if inputs.elevation_above_sea_m < FLOODED_MAX_ELEV_M
        && (inputs.lake_adjacent || inputs.water_flux_high)
        && inputs.temperature_c >= FLOODED_MIN_TEMP_C
    {
        return Biome::FloodedGrassland;
    }

    // 3. Montane grassland: high, low-latitude, not desert-dry.
    if inputs.elevation_above_sea_m > MONTANE_MIN_ELEV_M
        && lat_deg < MONTANE_MAX_LAT_DEG
        && inputs.precip_mm_yr >= MONTANE_MIN_PRECIP_MM
    {
        return Biome::MontaneGrassland;
    }

    // 4. Mediterranean: 30-45 deg, coastal-ish, temperate + moderate rain
    //    (winter-wet/summer-dry proxy; see module docs).
    if (MED_MIN_LAT_DEG..MED_MAX_LAT_DEG).contains(&lat_deg)
        && inputs.dist_to_ocean <= MED_MAX_DIST_CELLS
        && (MED_MIN_TEMP_C..=MED_MAX_TEMP_C).contains(&inputs.temperature_c)
        && (MED_MIN_PRECIP_MM..=MED_MAX_PRECIP_MM).contains(&inputs.precip_mm_yr)
    {
        return Biome::Mediterranean;
    }

    // 5. Tundra: warmest month still below 10 C.
    if inputs.summer_c < TUNDRA_MAX_SUMMER_C {
        return Biome::Tundra;
    }

    whittaker(
        inputs.temperature_c,
        inputs.summer_c,
        inputs.winter_c,
        inputs.precip_mm_yr,
        inputs.elevation_above_sea_m,
    )
}

/// Classify one cell: water/ice first (ice sheet, ocean, lake), then land.
/// Pure function — see the module docs for the full priority order.
pub fn classify_cell(inputs: &CellInputs) -> Biome {
    if inputs.ice_thickness_m > ICE_SHEET_THICKNESS_M {
        return Biome::IceSheet;
    }
    if inputs.is_ocean {
        return Biome::Ocean;
    }
    if inputs.lake_depth_m > LAKE_DEPTH_M {
        return Biome::Lake;
    }
    classify_land(inputs)
}

/// Satellite-look palette for the beauty view and map export (WP11/WP9
/// consume this): saturated greens for wet vegetation, sand/earth tones for
/// arid land, grey-browns for tundra, deep blues for water, near-white for
/// ice.
pub fn biome_color(b: Biome) -> [u8; 3] {
    match b {
        Biome::Unclassified => [128, 128, 128],
        Biome::TropicalMoistBroadleaf => [15, 90, 31],
        Biome::TropicalDryBroadleaf => [107, 122, 44],
        Biome::TropicalConifer => [46, 94, 63],
        Biome::TemperateBroadleaf => [58, 122, 58],
        Biome::TemperateConifer => [36, 74, 66],
        Biome::BorealTaiga => [42, 66, 56],
        Biome::TropicalGrassland => [189, 176, 90],
        Biome::TemperateGrassland => [163, 172, 100],
        Biome::FloodedGrassland => [90, 122, 92],
        Biome::MontaneGrassland => [150, 140, 110],
        Biome::Tundra => [138, 133, 119],
        Biome::Mediterranean => [147, 158, 108],
        Biome::Desert => [201, 168, 106],
        Biome::Mangrove => [22, 58, 51],
        Biome::Ocean => [11, 42, 94],
        Biome::Lake => [29, 78, 137],
        Biome::IceSheet => [238, 244, 248],
    }
}

// --- process ---------------------------------------------------------------

/// The biome [`Process`]: writes `Planet::biome` every step from the
/// current temperature/precipitation/elevation/water fields.
///
/// Holds no state (not even a mesh-derived cache) — every field is
/// recomputed from `(planet, mesh)`, so classification is a pure function of
/// current planet state: determinism and cold-start/warm-instance equality
/// fall out for free. Full 14-biome classification runs in every phase
/// (including `CrustalFormation`/`Drift`); the measured cost at subdivision
/// level 6 is well under the 5 ms budget (see the `step_budget_level_6`
/// test), so there's no benefit to the cheaper "water/ice + Unclassified
/// only" early-phase mode the plan allows — one code path is simpler and the
/// early globe still renders sensibly since e.g. an all-Ocean proto-planet
/// classifies as such immediately.
#[derive(Debug, Default)]
pub struct BiomeProcess;

impl BiomeProcess {
    /// New process. Stateless; equivalent to `BiomeProcess::default()`.
    pub fn new() -> BiomeProcess {
        BiomeProcess
    }
}

impl Process for BiomeProcess {
    fn name(&self) -> &'static str {
        "biomes"
    }

    /// Diagnostic: the output depends only on the current `Planet` fields
    /// and mesh, not on `dt_myr`, and no randomness is consumed.
    fn step(&mut self, planet: &mut Planet, mesh: &Mesh, _dt_myr: f64, _ctx: &mut StepCtx) {
        debug_assert_eq!(planet.n_cells(), mesh.n_cells());
        let n = planet.n_cells();

        let dist = distance_to_ocean_cells(planet, mesh);
        let (summer, winter) = seasonal_extremes_c(planet, mesh);
        let flux_high = water_flux_high_threshold(&planet.water_flux_m3_yr);
        let lake_mask: Vec<bool> = planet
            .lake_depth_m
            .iter()
            .map(|d| *d > LAKE_DEPTH_M)
            .collect();

        let biome: Vec<Biome> = (0..n)
            .into_par_iter()
            .map(|i| {
                let cell = i as u32;
                let lake_adjacent = mesh
                    .neighbors_of(cell)
                    .iter()
                    .any(|&nb| lake_mask[nb as usize]);
                let inputs = CellInputs {
                    temperature_c: planet.temperature_c[i],
                    summer_c: summer[i],
                    winter_c: winter[i],
                    precip_mm_yr: planet.precip_mm_yr[i],
                    elevation_above_sea_m: planet.elevation_above_sea_m(cell),
                    lat_rad: mesh.latlon[i][0],
                    dist_to_ocean: dist[i],
                    lake_adjacent,
                    water_flux_high: planet.water_flux_m3_yr[i] >= flux_high,
                    ice_thickness_m: planet.ice_thickness_m[i],
                    is_ocean: planet.is_ocean(cell),
                    lake_depth_m: planet.lake_depth_m[i],
                };
                classify_cell(&inputs)
            })
            .collect();
        planet.biome = biome;
    }
}
