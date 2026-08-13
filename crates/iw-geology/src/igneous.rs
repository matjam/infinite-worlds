//! Igneous emplacement driven by the tectonic flags (DESIGN.md §6).
//!
//! Arcs grow dioritic/granitic roots at depth and erupt andesite and tuff on
//! top of them; continental collisions inflate granite plutons in the thickened
//! core; hotspots pile basalt on the surface.
//!
//! Emplaced material is added to `crust_thickness_m` as well as to the
//! stratigraphic column: `crust_thickness_m` is the isostatic column and the
//! strata are its record (see [`crate::isostasy`]), so both must move together
//! or the new rock would either be invisible to isostasy or counted twice.
//! Thickness is clamped at [`MAX_CRUST_THICKNESS_M`]; the strata (and the
//! ledger) still record the full emplacement.
//!
//! The pass is serial in ascending cell order and draws from `ctx.rng` only for
//! flagged cells, so the RNG stream is a function of planet state alone —
//! parallelism could not preserve that.

use iw_core::planet::cell_flags;
use iw_core::{MetamorphicGrade, Planet, RockType, StepCtx, Stratum};
use iw_mesh::Mesh;
use rand::Rng;

use crate::isostasy::MAX_CRUST_THICKNESS_M;

/// Arc pluton growth, metres of rock per Myr at `tectonic_vigor = 1`.
pub const ARC_PLUTON_M_PER_MYR: f32 = 60.0;
/// Emplacement depth of arc plutons below the surface, m.
pub const ARC_PLUTON_DEPTH_M: f32 = 5_000.0;
/// Probability per Myr that an arc cell erupts in a given step.
pub const ARC_ERUPTION_PER_MYR: f64 = 0.5;
/// Thickness range of one arc eruption, m.
pub const ARC_ERUPTION_M: std::ops::Range<f32> = 20.0..80.0;
/// Collision-melt pluton growth, metres per Myr at `tectonic_vigor = 1`.
pub const COLLISION_PLUTON_M_PER_MYR: f32 = 40.0;
/// Emplacement depth of collision plutons below the surface, m.
pub const COLLISION_PLUTON_DEPTH_M: f32 = 12_000.0;
/// Hotspot basalt effusion, metres per Myr at `tectonic_vigor = 1`.
pub const HOTSPOT_BASALT_M_PER_MYR: f32 = 150.0;

/// Run one step of igneous emplacement. Call before isostasy so the step's
/// elevation already reflects the new rock.
pub fn emplace(planet: &mut Planet, mesh: &Mesh, dt_myr: f64, ctx: &mut StepCtx) {
    let n = planet.n_cells();
    debug_assert_eq!(n, mesh.n_cells(), "planet and mesh disagree on cell count");
    let dt = dt_myr.max(0.0);
    if dt == 0.0 {
        return;
    }
    let vigor = planet.config.tectonic_vigor.max(0.0);
    let rate = dt as f32 * vigor;
    let time_myr = planet.time_myr;

    for cell in 0..n as u32 {
        let flags = planet.tectonic_flags[cell as usize];
        if flags & (cell_flags::ARC | cell_flags::COLLISION | cell_flags::HOTSPOT) == 0 {
            continue;
        }
        let area_m2 = mesh.areas_km2[cell as usize] as f64 * 1.0e6;

        if flags & cell_flags::ARC != 0 {
            let t = ARC_PLUTON_M_PER_MYR * rate;
            let rock = if ctx.rng.random_bool(0.6) {
                RockType::Diorite
            } else {
                RockType::Granite
            };
            intrude(planet, cell, ARC_PLUTON_DEPTH_M, rock, t, time_myr);
            account(planet, cell, t, area_m2, ctx);

            let p = (ARC_ERUPTION_PER_MYR * dt).clamp(0.0, 1.0);
            if ctx.rng.random_bool(p) {
                let t = ctx.rng.random_range(ARC_ERUPTION_M) * vigor;
                let rock = if ctx.rng.random_bool(0.7) {
                    RockType::Andesite
                } else {
                    RockType::Tuff
                };
                planet.columns.deposit(cell, rock, t, time_myr);
                account(planet, cell, t, area_m2, ctx);
            }
        }

        if flags & cell_flags::COLLISION != 0 {
            let t = COLLISION_PLUTON_M_PER_MYR * rate;
            intrude(
                planet,
                cell,
                COLLISION_PLUTON_DEPTH_M,
                RockType::Granite,
                t,
                time_myr,
            );
            account(planet, cell, t, area_m2, ctx);
        }

        if flags & cell_flags::HOTSPOT != 0 {
            let t = HOTSPOT_BASALT_M_PER_MYR * rate;
            planet.columns.deposit(cell, RockType::Basalt, t, time_myr);
            account(planet, cell, t, area_m2, ctx);
        }
    }
}

/// Insert a sill/pluton `depth_m` below the surface of the cell's column.
fn intrude(
    planet: &mut Planet,
    cell: u32,
    depth_m: f32,
    rock: RockType,
    thickness_m: f32,
    time_myr: f64,
) {
    if thickness_m <= 0.0 {
        return;
    }
    planet.columns.intrude(
        cell,
        depth_m,
        Stratum {
            rock,
            thickness_m,
            deposited_myr: time_myr as f32,
            grade: MetamorphicGrade::None,
        },
    );
}

/// Thicken the isostatic column and record the new rock in the mass ledger.
fn account(planet: &mut Planet, cell: u32, thickness_m: f32, area_m2: f64, ctx: &mut StepCtx) {
    if thickness_m <= 0.0 {
        return;
    }
    let h = &mut planet.crust_thickness_m[cell as usize];
    *h = (*h + thickness_m).min(MAX_CRUST_THICKNESS_M);
    ctx.ledger.created_m3 += thickness_m as f64 * area_m2;
}
