//! The mass-transfer primitives every surface pass goes through, and the
//! lithification pass that turns loose regolith into stratigraphy.
//!
//! # The crust-thickness contract
//!
//! `Planet::crust_thickness_m` is the isostatic column: basement plus every
//! solid layer recorded in [`iw_core::StrataColumns`]. `Planet::sediment_m` is
//! loose regolith carried *outside* that column (iw-geology applies it as a
//! separate buoyant load). Therefore:
//!
//! - removing bedrock (a `columns.erode`, or basement below an exhausted
//!   column) must subtract the same thickness from `crust_thickness_m`;
//! - adding strata (`columns.deposit`) must add the same thickness;
//! - moving material into or out of `sediment_m` must not touch either.
//!
//! Every write in this crate goes through [`MassMover`] or [`lithify`] so the
//! rule is applied in exactly one place.

use iw_core::{CrustType, MassLedger, Planet, RockType};
use iw_mesh::Mesh;

use crate::geom::Geometry;
use crate::hydro::Hydrology;

/// Crust is never eroded below this thickness; a cell that thin is already a
/// hole in the lithosphere and isostasy would send it to the mantle.
pub const MIN_CRUST_THICKNESS_M: f32 = 500.0;

/// Loose regolith depth above which subaerial sediment lithifies into strata.
pub const LITHIFY_THRESHOLD_M: f32 = 200.0;
/// Loose cover left on a submarine cell; everything above it lithifies (burial
/// under water compacts and cements far faster than on land).
pub const SUBMARINE_LOOSE_KEEP_M: f32 = 5.0;
/// Water depth separating "shallow shelf" (sand) from "deep, quiet" (mud).
pub const SHELF_DEPTH_M: f32 = 200.0;
/// Sea-surface temperature above which carbonate factories run.
pub const CARBONATE_MIN_TEMP_C: f32 = 18.0;
/// Clastic discharge delivered by neighbouring cells above which a shallow
/// warm sea is too muddy for limestone, m^3/yr.
pub const CARBONATE_MAX_CLASTIC_M3_YR: f64 = 1.0e9;
/// Neighbour relief that marks a mountain front (alluvial fans, conglomerate).
pub const MOUNTAIN_FRONT_RELIEF_M: f32 = 500.0;

/// Erodibility used when a column is exhausted and bare basement is exposed.
#[inline]
pub fn basement_erodibility(crust: CrustType) -> f32 {
    match crust {
        CrustType::Continental => RockType::Granite.erodibility(),
        CrustType::Oceanic => RockType::Basalt.erodibility(),
    }
}

/// Erodibility of whatever is currently at the surface of `cell`: loose
/// regolith if present (soft), otherwise the top stratum, otherwise basement.
#[inline]
pub fn surface_erodibility(planet: &Planet, cell: u32) -> f32 {
    if planet.sediment_m[cell as usize] > 0.0 {
        return 2.0;
    }
    planet
        .columns
        .top_rock(cell)
        .map(|r| r.erodibility())
        .unwrap_or_else(|| basement_erodibility(planet.crust_type[cell as usize]))
}

/// Ledger-aware erosion and deposition. Holds the area table (to convert
/// thicknesses into volumes) and the step's [`MassLedger`].
pub struct MassMover<'a> {
    /// Cell areas, m^2.
    pub area_m2: &'a [f64],
    /// Step ledger; erosion and deposition volumes land here.
    pub ledger: &'a mut MassLedger,
}

impl MassMover<'_> {
    /// Strip up to `depth_m` from the top of a cell: loose regolith first,
    /// bedrock afterwards. Returns the thickness actually removed and books it
    /// as `eroded_m3`.
    pub fn erode(&mut self, planet: &mut Planet, cell: u32, depth_m: f32) -> f32 {
        if !(depth_m.is_finite() && depth_m > 0.0) {
            return 0.0;
        }
        let i = cell as usize;
        let loose = planet.sediment_m[i].min(depth_m).max(0.0);
        planet.sediment_m[i] -= loose;
        let mut removed = loose;
        let rest = depth_m - loose;
        if rest > 0.0 {
            removed += self.erode_bedrock_inner(planet, cell, rest);
        }
        self.ledger.eroded_m3 += removed as f64 * self.area_m2[i];
        removed
    }

    /// Strip up to `depth_m` of *bedrock* only, leaving loose regolith alone
    /// (weathering and glacial plucking of clean rock). Books `eroded_m3`.
    pub fn erode_bedrock(&mut self, planet: &mut Planet, cell: u32, depth_m: f32) -> f32 {
        if !(depth_m.is_finite() && depth_m > 0.0) {
            return 0.0;
        }
        let removed = self.erode_bedrock_inner(planet, cell, depth_m);
        self.ledger.eroded_m3 += removed as f64 * self.area_m2[cell as usize];
        removed
    }

    /// Bedrock removal without ledger bookkeeping (callers above do it).
    fn erode_bedrock_inner(&mut self, planet: &mut Planet, cell: u32, depth_m: f32) -> f32 {
        let i = cell as usize;
        let ct = planet.crust_thickness_m[i];
        let budget = (ct - MIN_CRUST_THICKNESS_M).max(0.0);
        let want = depth_m.min(budget);
        if want <= 0.0 {
            return 0.0;
        }
        // Whatever the record cannot supply comes out of unrecorded basement;
        // either way `crust_thickness_m` drops by the same amount.
        planet.columns.erode(cell, want);
        planet.crust_thickness_m[i] = ct - want;
        want
    }

    /// Lay `thickness_m` of loose regolith on `cell` and book `deposited_m3`.
    pub fn deposit_loose(&mut self, planet: &mut Planet, cell: u32, thickness_m: f32) {
        if !(thickness_m.is_finite() && thickness_m > 0.0) {
            return;
        }
        planet.sediment_m[cell as usize] += thickness_m;
        self.ledger.deposited_m3 += thickness_m as f64 * self.area_m2[cell as usize];
    }

    /// Precipitate `thickness_m` of chemical sediment straight into the
    /// column (evaporite). This is mass entering the model from solution, so
    /// it is booked as `created_m3`, not `deposited_m3`.
    pub fn precipitate(
        &mut self,
        planet: &mut Planet,
        cell: u32,
        rock: RockType,
        thickness_m: f32,
        time_myr: f64,
    ) {
        if !(thickness_m.is_finite() && thickness_m > 0.0) {
            return;
        }
        planet.columns.deposit(cell, rock, thickness_m, time_myr);
        planet.crust_thickness_m[cell as usize] += thickness_m;
        self.ledger.created_m3 += thickness_m as f64 * self.area_m2[cell as usize];
    }
}

/// Move loose regolith that is deep enough (or drowned) into the
/// stratigraphic record, choosing the facies from the local environment.
///
/// This is a loose -> solid conversion of material that was already booked as
/// deposited when it came to rest, so it touches neither ledger counter.
pub fn lithify(planet: &mut Planet, mesh: &Mesh, geom: &Geometry, hydro: &Hydrology) {
    let sea = planet.sea_level_m;
    let time = planet.time_myr;
    for cell in 0..planet.n_cells() as u32 {
        let i = cell as usize;
        let submarine = planet.elevation_m[i] < sea;
        let keep = if submarine {
            SUBMARINE_LOOSE_KEEP_M
        } else {
            LITHIFY_THRESHOLD_M
        };
        let excess = planet.sediment_m[i] - keep;
        if excess <= 0.0 {
            continue;
        }
        let rock = facies(planet, mesh, geom, hydro, cell, submarine);
        planet.sediment_m[i] -= excess;
        planet.columns.deposit(cell, rock, excess, time);
        planet.crust_thickness_m[i] += excess;
    }
}

/// Rock type produced by burying loose clastic sediment at `cell`
/// (DESIGN.md §6 facies rules).
fn facies(
    planet: &Planet,
    mesh: &Mesh,
    _geom: &Geometry,
    hydro: &Hydrology,
    cell: u32,
    submarine: bool,
) -> RockType {
    let i = cell as usize;
    let sea = planet.sea_level_m;
    if submarine {
        let depth = sea - planet.elevation_m[i];
        if depth > SHELF_DEPTH_M {
            return RockType::Shale;
        }
        // Warm, shallow and starved of mud: carbonate platform.
        let clastic: f64 = mesh
            .neighbors_of(cell)
            .iter()
            .map(|&m| hydro.discharge_m3_yr[m as usize])
            .sum();
        if planet.temperature_c[i] > CARBONATE_MIN_TEMP_C
            && clastic < CARBONATE_MAX_CLASTIC_M3_YR
            && hydro.discharge_m3_yr[i] < CARBONATE_MAX_CLASTIC_M3_YR
        {
            return RockType::Limestone;
        }
        return RockType::Sandstone;
    }
    // Subaerial: coarse debris right at a mountain front, sand elsewhere.
    let mut relief = 0.0f32;
    for &m in mesh.neighbors_of(cell) {
        relief = relief.max(planet.elevation_m[m as usize] - planet.elevation_m[i]);
    }
    if relief > MOUNTAIN_FRONT_RELIEF_M {
        RockType::Conglomerate
    } else {
        RockType::Sandstone
    }
}
