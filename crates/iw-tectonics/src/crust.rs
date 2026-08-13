//! Crust creation and destruction primitives.
//!
//! Every routine that adds or removes stratigraphic column material goes
//! through here so the mass ledger stays honest: material appearing from the
//! mantle is `created_m3`, material destroyed at a trench (or overwritten when
//! a cell changes crust type) is `subducted_m3`. Deposition/erosion of existing
//! surface material is not tectonics' business, so `deposited_m3`/`eroded_m3`
//! are never touched here — that keeps `MassLedger::residual_m3` (which
//! `iw-sim` debug-asserts) balanced.

use iw_core::{CrustType, MassLedger, Planet, RockType};

use crate::{CONTINENTAL_DENSITY_KG_M3, OCEANIC_DENSITY_KG_M3};

/// Gabbro fraction of a fresh oceanic column (lower crust), metres.
const OCEAN_GABBRO_M: f32 = 5_000.0;
/// Basalt fraction of a fresh oceanic column (pillow lavas / sheeted dykes).
const OCEAN_BASALT_M: f32 = 2_000.0;
/// Gneissic basement of a craton column, metres.
const CRATON_GNEISS_M: f32 = 4_000.0;
/// Granitic upper basement of a craton column, metres.
const CRATON_GRANITE_M: f32 = 6_000.0;

/// Wipe a cell's column, charging the loss to the ledger.
pub(crate) fn destroy_column(
    planet: &mut Planet,
    cell: u32,
    area_m2: f64,
    ledger: &mut MassLedger,
) {
    let t = planet.columns.clear(cell);
    if t > 0.0 {
        ledger.subducted_m3 += t as f64 * area_m2;
    }
}

/// Deposit `thickness_m` of `rock` on top of a column, charging the gain to the
/// ledger as newly created (mantle-derived) material.
pub(crate) fn deposit_new(
    planet: &mut Planet,
    cell: u32,
    rock: RockType,
    thickness_m: f32,
    area_m2: f64,
    ledger: &mut MassLedger,
) {
    if thickness_m <= 0.0 {
        return;
    }
    let t = planet.time_myr;
    planet.columns.deposit(cell, rock, thickness_m, t);
    ledger.created_m3 += thickness_m as f64 * area_m2;
}

/// Reset a cell to pristine age-0 oceanic crust with a basalt-over-gabbro
/// column. Used at ridges, at breakup of over-stretched continental crust, and
/// for the global proto-crust at the very start of the run.
///
/// `thickness_m` is the cell's reference sea-floor thickness
/// (`MeshCache::ocean_thickness_m`), not the bare constant: the reference
/// carries a low-amplitude noise field so the abyssal plains are not a
/// perfectly flat sheet.
pub(crate) fn make_fresh_oceanic(
    planet: &mut Planet,
    cell: u32,
    thickness_m: f32,
    area_m2: f64,
    ledger: &mut MassLedger,
) {
    destroy_column(planet, cell, area_m2, ledger);
    planet.crust_type[cell as usize] = CrustType::Oceanic;
    planet.crust_thickness_m[cell as usize] = thickness_m;
    planet.crust_density_kg_m3[cell as usize] = OCEANIC_DENSITY_KG_M3;
    planet.crust_age_myr[cell as usize] = 0.0;
    deposit_new(
        planet,
        cell,
        RockType::Gabbro,
        OCEAN_GABBRO_M,
        area_m2,
        ledger,
    );
    deposit_new(
        planet,
        cell,
        RockType::Basalt,
        OCEAN_BASALT_M,
        area_m2,
        ledger,
    );
}

/// Turn a cell into craton basement: granite over gneiss, thick and buoyant.
pub(crate) fn make_continental(
    planet: &mut Planet,
    cell: u32,
    thickness_m: f32,
    area_m2: f64,
    ledger: &mut MassLedger,
) {
    destroy_column(planet, cell, area_m2, ledger);
    planet.crust_type[cell as usize] = CrustType::Continental;
    planet.crust_thickness_m[cell as usize] = thickness_m;
    planet.crust_density_kg_m3[cell as usize] = CONTINENTAL_DENSITY_KG_M3;
    planet.crust_age_myr[cell as usize] = 0.0;
    deposit_new(
        planet,
        cell,
        RockType::Gneiss,
        CRATON_GNEISS_M,
        area_m2,
        ledger,
    );
    deposit_new(
        planet,
        cell,
        RockType::Granite,
        CRATON_GRANITE_M,
        area_m2,
        ledger,
    );
}
