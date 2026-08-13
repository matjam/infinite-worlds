//! Airy isostasy: elevation as the buoyancy response to crustal and surface
//! loads, followed by flexural smoothing.
//!
//! # What a "column" means here
//!
//! `Planet::crust_thickness_m` is the authority for the compensating column: it
//! covers basement *plus* everything solid that geology/tectonics emplaced into
//! the stratigraphic column (arc plutons, flood basalts, lithified sediment
//! that tectonics folded into the crust). [`iw_core::StrataColumns`] is the
//! *record* of that material, not a second, additive store — isostasy never
//! adds column thickness on top of `crust_thickness_m`, or every emplacement
//! would be counted twice. `sediment_m` (loose regolith, surface-owned) and
//! `ice_thickness_m` are the only loads carried outside `crust_thickness_m`.
//!
//! # Derivation of the constant offset
//!
//! Local (Airy) isostasy: the mass per unit area of every column above the
//! compensation depth `D` is equal. Take a column whose ground surface is at
//! elevation `e`: loose sediment of thickness `S` and density `rho_s` on top,
//! crust of thickness `H` and density `rho_c` below it, mantle `rho_m` filling
//! the rest down to `-D`, plus an ice load `t_i` at `rho_i` resting on top.
//!
//! ```text
//! M = rho_i*t_i + rho_s*S + rho_c*H + rho_m*(D + e - S - H)
//! ```
//!
//! Solving for `e` with `M` and `D` constant collapses every constant into one
//! offset `C = M/rho_m - D`:
//!
//! ```text
//! e = C + H*(rho_m - rho_c)/rho_m + S*(rho_m - rho_s)/rho_m - t_i*rho_i/rho_m
//! ```
//!
//! `C` is fixed analytically by the continental anchor {35 km, 2700 kg/m^3} ->
//! +800 m (see [`ISOSTATIC_OFFSET_M`]); `D` and `M` are never needed
//! separately.
//!
//! # Thermal lid term (why pure Airy is not enough)
//!
//! Crustal Airy alone cannot reproduce ocean bathymetry. Tectonics encodes
//! ocean-floor age as a *density* ramp (fresh 3000 -> old 3300 kg/m^3), and
//! over 7 km of crust that whole ramp is worth only
//! `7000*300/3300 = 636 m`, while real (and target) ridge-to-abyssal-plain
//! relief is ~3 km. That is because thermal subsidence lives in the ~100 km
//! mantle lid, not in the crust. So the density anomaly written by tectonics is
//! also applied over an effective lid thickness [`THERMAL_LID_M`], as a
//! buoyant *lift* on young oceanic lithosphere that decays to zero once the
//! crust has aged to [`RHO_OCEAN_EQUILIBRATED_KG_M3`]:
//!
//! ```text
//! e_thermal = (rho_equilibrated - rho_c) * L_lid / rho_m     (oceanic only)
//! ```
//!
//! Continental lithosphere is thermally equilibrated and gets no lift, which is
//! also what keeps the term from wrecking the continental anchor.

use iw_core::{CrustType, Planet};
use iw_mesh::Mesh;
use rayon::prelude::*;

/// Asthenospheric mantle density the whole model compensates against.
pub const RHO_MANTLE_KG_M3: f32 = 3300.0;
/// Bulk density of loose sediment (`Planet::sediment_m`).
pub const RHO_SEDIMENT_KG_M3: f32 = 2000.0;
/// Glacier/ice-sheet density (`Planet::ice_thickness_m`).
pub const RHO_ICE_KG_M3: f32 = 917.0;
/// Oceanic crust density at which thermal subsidence is complete; tectonics
/// ramps `crust_density_kg_m3` toward this value with age.
pub const RHO_OCEAN_EQUILIBRATED_KG_M3: f32 = 3300.0;

/// Calibration anchor: mean continental crust thickness, m.
const ANCHOR_CONT_H_M: f32 = 35_000.0;
/// Calibration anchor: mean continental crust density, kg/m^3.
const ANCHOR_CONT_RHO: f32 = 2700.0;
/// Calibration anchor: elevation the anchor column must float at, m.
const ANCHOR_CONT_ELEV_M: f32 = 800.0;
/// Calibration anchor: fresh oceanic crust {7 km, 3000 kg/m^3} depth, m.
const ANCHOR_RIDGE_H_M: f32 = 7_000.0;
/// Calibration anchor: fresh oceanic crust density, kg/m^3.
const ANCHOR_RIDGE_RHO: f32 = 3000.0;
/// Calibration anchor: ridge-crest elevation, m.
const ANCHOR_RIDGE_ELEV_M: f32 = -3000.0;

/// Buoyant thickness contributed by a slab of `thickness` at `density`.
const fn buoyancy_m(thickness_m: f32, density_kg_m3: f32) -> f32 {
    thickness_m * (RHO_MANTLE_KG_M3 - density_kg_m3) / RHO_MANTLE_KG_M3
}

/// Constant `C` of the Airy solution, in metres, derived analytically from the
/// {35 km, 2700 kg/m^3} -> +800 m continental anchor (module docs).
///
/// `C = 800 - 35000*(3300-2700)/3300 = -5563.64 m`
pub const ISOSTATIC_OFFSET_M: f32 =
    ANCHOR_CONT_ELEV_M - buoyancy_m(ANCHOR_CONT_H_M, ANCHOR_CONT_RHO);

/// Effective mantle-lid thickness over which the oceanic crust-density anomaly
/// acts, in metres. Derived from the ridge anchor:
///
/// `L = (e_ridge - C - buoyancy(7 km, 3000)) * rho_m / (rho_eq - 3000)`
/// ` = (-3000 + 5563.64 - 636.36) * 3300 / 300 = 21200 m`
///
/// Equivalently a 100 km lid carrying ~21% of the crustal density anomaly,
/// i.e. ~63 kg/m^3 of cooling contrast — the right order for 600 K of
/// lithospheric cooling at `alpha = 3e-5 /K`.
pub const THERMAL_LID_M: f32 =
    (ANCHOR_RIDGE_ELEV_M - ISOSTATIC_OFFSET_M - buoyancy_m(ANCHOR_RIDGE_H_M, ANCHOR_RIDGE_RHO))
        * RHO_MANTLE_KG_M3
        / (RHO_OCEAN_EQUILIBRATED_KG_M3 - ANCHOR_RIDGE_RHO);

/// Upper bound on `crust_thickness_m` used when geology thickens crust by
/// emplacement. Continental roots do not exceed ~70-75 km on Earth.
pub const MAX_CRUST_THICKNESS_M: f32 = 75_000.0;

/// Jacobi relaxation passes applied after the local solve.
pub const FLEXURE_PASSES: usize = 3;
/// Self weight of one flexural pass; the rest is spread over the neighbour
/// mean. Three passes at 0.6 spread a single-cell load over ~2 rings while
/// leaving a continent/ocean step at ~70% of its unsmoothed height.
pub const FLEXURE_SELF_WEIGHT: f32 = 0.6;

/// Local Airy elevation of one column, metres relative to the reference geoid.
///
/// `oceanic` selects the thermal-lid lift (module docs); it is off for
/// continental lithosphere. No water load is applied: sea level is solved
/// afterwards from this surface.
#[inline]
pub fn airy_elevation_m(
    crust_thickness_m: f32,
    crust_density_kg_m3: f32,
    oceanic: bool,
    sediment_m: f32,
    ice_thickness_m: f32,
) -> f32 {
    let crust = buoyancy_m(crust_thickness_m.max(0.0), crust_density_kg_m3);
    let sediment = buoyancy_m(sediment_m.max(0.0), RHO_SEDIMENT_KG_M3);
    let ice = ice_thickness_m.max(0.0) * RHO_ICE_KG_M3 / RHO_MANTLE_KG_M3;
    let thermal = if oceanic {
        (RHO_OCEAN_EQUILIBRATED_KG_M3 - crust_density_kg_m3).max(0.0) * THERMAL_LID_M
            / RHO_MANTLE_KG_M3
    } else {
        0.0
    };
    ISOSTATIC_OFFSET_M + crust + sediment + thermal - ice
}

/// Recompute `planet.elevation_m` from scratch: local Airy solve, then
/// [`FLEXURE_PASSES`] Jacobi smoothing passes.
///
/// Fully recomputed from state every step (no incremental accumulation), so a
/// checkpoint resume reproduces elevation bit-for-bit. `scratch` is reused
/// across steps purely to avoid reallocation; it carries no state.
pub fn update_elevation(planet: &mut Planet, mesh: &Mesh, scratch: &mut Vec<f32>) {
    let n = planet.n_cells();
    debug_assert_eq!(n, mesh.n_cells(), "planet and mesh disagree on cell count");

    let mut src = std::mem::take(scratch);
    (0..n)
        .into_par_iter()
        .map(|i| {
            airy_elevation_m(
                planet.crust_thickness_m[i],
                planet.crust_density_kg_m3[i],
                planet.crust_type[i] == CrustType::Oceanic,
                planet.sediment_m[i],
                planet.ice_thickness_m[i],
            )
        })
        .collect_into_vec(&mut src);

    let mut dst = std::mem::take(&mut planet.elevation_m);
    dst.clear();
    dst.resize(n, 0.0);
    for _ in 0..FLEXURE_PASSES {
        let cur = &src;
        dst.par_iter_mut().enumerate().for_each(|(i, out)| {
            let nb = mesh.neighbors_of(i as u32);
            // Fixed CSR order: the sum is bit-reproducible.
            let mut sum = 0.0f32;
            for &m in nb {
                sum += cur[m as usize];
            }
            let mean = sum / nb.len() as f32;
            *out = FLEXURE_SELF_WEIGHT * cur[i] + (1.0 - FLEXURE_SELF_WEIGHT) * mean;
        });
        std::mem::swap(&mut src, &mut dst);
    }

    // The final pass wrote `dst` and then swapped, so the result is in `src`.
    planet.elevation_m = src;
    *scratch = dst;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_and_lid_match_hand_derivation() {
        assert!((ISOSTATIC_OFFSET_M - -5563.636).abs() < 0.01);
        assert!((THERMAL_LID_M - 21_200.0).abs() < 1.0);
    }

    #[test]
    fn anchors_are_exact_before_flexure() {
        let cont = airy_elevation_m(35_000.0, 2700.0, false, 0.0, 0.0);
        assert!((cont - 800.0).abs() < 1.0, "continental anchor {cont}");
        let ridge = airy_elevation_m(7_000.0, 3000.0, true, 0.0, 0.0);
        assert!((ridge - -3000.0).abs() < 1.0, "ridge anchor {ridge}");
    }

    #[test]
    fn ice_depresses_by_density_ratio() {
        let bare = airy_elevation_m(35_000.0, 2700.0, false, 0.0, 0.0);
        let iced = airy_elevation_m(35_000.0, 2700.0, false, 0.0, 3000.0);
        let expected = 3000.0 * RHO_ICE_KG_M3 / RHO_MANTLE_KG_M3;
        assert!((bare - iced - expected).abs() < 1.0);
    }
}
