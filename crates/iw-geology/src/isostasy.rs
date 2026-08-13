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
//! # Oceanic lid terms (why pure Airy is not enough)
//!
//! Crustal Airy alone cannot reproduce ocean bathymetry. Tectonics encodes
//! ocean-floor age as a *density* ramp (fresh 3000 -> old 3300 kg/m^3), and
//! over 7 km of crust that whole ramp is worth only
//! `7000*300/3300 = 636 m`, while real ridge-to-abyssal-plain relief is a
//! couple of km. That is because both the standing height of oceanic
//! lithosphere and its thermal subsidence live in the ~100 km mantle lid, not
//! in the crust. Two lid terms carry that, both oceanic-only (continental
//! lithosphere is thermally equilibrated and undepleted, so it gets neither,
//! which is what keeps them from wrecking the continental anchor):
//!
//! ```text
//! e_lid = OCEAN_LID_RESIDUAL_M                               (age-independent)
//!       + (rho_equilibrated - rho_c) * L_lid / rho_m         (decays with age)
//! ```
//!
//! * [`OCEAN_LID_RESIDUAL_M`] is the age-independent buoyancy of the *depleted*
//!   oceanic mantle lid: melt extraction at the ridge leaves harzburgite some
//!   tens of kg/m^3 lighter than fertile asthenosphere, and that contrast never
//!   goes away as the plate cools. 1364 m of standing height is ~45 kg/m^3 over
//!   a 100 km lid — squarely in the measured range.
//! * The second term is thermal: it lifts young lithosphere and decays to zero
//!   once the crust has aged to [`RHO_OCEAN_EQUILIBRATED_KG_M3`].
//!
//! The residual is what fixes the abyssal-plain anchor
//! ([`ANCHOR_OCEAN_OLD_ELEV_M`]), and through it the planet's freeboard: with a
//! 1.0-Earth-ocean water budget, sea level solves to roughly
//! `ANCHOR_OCEAN_OLD_ELEV_M + 4 km`, so the continent/abyss step is what
//! decides whether continents stand 0.8 km or 2.4 km above the waterline.
//!
//! # Trench flexure
//!
//! Trenches are not isostatic: they are the elastic downwarp of a plate being
//! bent into a subduction zone, and no crust-column model reproduces them —
//! near the equilibrated density the Airy term is insensitive to thickness by
//! construction. Tectonics marks the down-going cells by thinning them below
//! [`OCEANIC_REFERENCE_THICKNESS_M`]; [`trench_deflection_m`] converts that
//! thinning into a bending deflection at [`TRENCH_FLEXURE_GAIN`] metres per
//! metre. The flexural passes that follow spread it into the outer rise, so the
//! deflection reaching the map is roughly two thirds of the raw value.

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
const ANCHOR_RIDGE_ELEV_M: f32 = -2_700.0;

/// Calibration anchor: elevation of thermally equilibrated ("old") oceanic
/// lithosphere — the abyssal plain — in metres.
///
/// Earth's abyssal plains sit near -5 km with mean ocean depth -3.7 km, because
/// Earth's sea floor carries a broad spread of ages: ridge flanks and young
/// basins occupy as much area as the old deeps. This model's plates are rigid
/// and ocean cells never advect, so interior sea floor ages monotonically and
/// the *whole* basin saturates at the old-crust density — the age spread that
/// makes Earth's hypsometric curve simply does not exist here. Anchoring the
/// saturated floor at Earth's abyssal-plain depth therefore reproduces a planet
/// whose *mean* ocean is 5.5 km deep, which drops sea level 1.6 km below the
/// geoid and leaves continents standing 2.4 km up (with the lapse-rate ice age
/// that implies).
///
/// So this anchor is calibrated against Earth's **mean** ocean depth rather
/// than its abyssal extreme: -4200 m plus the ~4 km of water a 1.0-Earth-ocean
/// budget puts over a ~2/3-ocean planet lands sea level within a few hundred
/// metres of the geoid, and hence continental freeboard near Earth's 0.8 km.
/// Trenches ([`trench_deflection_m`]) still reach past -7 km from here.
pub const ANCHOR_OCEAN_OLD_ELEV_M: f32 = -4_200.0;

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

/// Age-independent standing height of depleted oceanic mantle lithosphere, in
/// metres (module docs). Fixed analytically by the abyssal anchor, since an
/// equilibrated 7 km / 3300 kg/m^3 column has zero crustal buoyancy:
///
/// `R = e_abyss - C - buoyancy(7 km, 3300) = -4200 + 5563.64 - 0 = 1363.64 m`
pub const OCEAN_LID_RESIDUAL_M: f32 = ANCHOR_OCEAN_OLD_ELEV_M
    - ISOSTATIC_OFFSET_M
    - buoyancy_m(ANCHOR_RIDGE_H_M, RHO_OCEAN_EQUILIBRATED_KG_M3);

/// Effective mantle-lid thickness over which the oceanic crust-density anomaly
/// acts, in metres. Derived from the ridge/abyss anchor pair:
///
/// `L = (e_ridge - e_abyss - buoyancy(7 km, 3000)) * rho_m / (rho_eq - 3000)`
/// ` = (-2700 + 4200 - 636.36) * 3300 / 300 = 9500 m`
///
/// Equivalently a 100 km lid carrying ~9.5% of the crustal density anomaly,
/// i.e. ~28 kg/m^3 of cooling contrast — the right order for 600 K of
/// lithospheric cooling at `alpha = 1.4e-5 /K` averaged over a lid that is only
/// hot at its base.
pub const THERMAL_LID_M: f32 = (ANCHOR_RIDGE_ELEV_M
    - ANCHOR_OCEAN_OLD_ELEV_M
    - buoyancy_m(ANCHOR_RIDGE_H_M, ANCHOR_RIDGE_RHO))
    * RHO_MANTLE_KG_M3
    / (RHO_OCEAN_EQUILIBRATED_KG_M3 - ANCHOR_RIDGE_RHO);

/// Reference thickness of undeformed oceanic crust, metres. Mirrors
/// `iw_tectonics::OCEANIC_THICKNESS_M`; iw-geology does not depend on
/// iw-tectonics, so the value is restated rather than imported. Only trench
/// cells are ever thinned below it.
pub const OCEANIC_REFERENCE_THICKNESS_M: f32 = 7_000.0;

/// Trench downwarp per metre of crustal thinning below
/// [`OCEANIC_REFERENCE_THICKNESS_M`], dimensionless (module docs).
///
/// Tectonics thins an actively subducting cell to 5 km, so the raw deflection
/// at a live trench is `2000 * 2.5 = 5000 m`; the flexural passes give back
/// about a third of that, putting Mariana-class trenches near -7.5 km with the
/// abyssal plain at -4.2 km.
pub const TRENCH_FLEXURE_GAIN: f32 = 2.5;

/// Largest trench downwarp, metres — a guard so a pathologically thin cell
/// cannot punch a hole through the planet. Never reached at the 5 km trench
/// thickness tectonics writes.
pub const MAX_TRENCH_DEFLECTION_M: f32 = 7_000.0;

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
    let lid = if oceanic {
        OCEAN_LID_RESIDUAL_M
            + (RHO_OCEAN_EQUILIBRATED_KG_M3 - crust_density_kg_m3).max(0.0) * THERMAL_LID_M
                / RHO_MANTLE_KG_M3
    } else {
        0.0
    };
    ISOSTATIC_OFFSET_M + crust + sediment + lid - ice
}

/// Downward bending deflection of a plate being flexed into a trench, metres
/// (positive = deflection to subtract from the isostatic elevation).
///
/// Zero for continental crust and for any oceanic column at or above
/// [`OCEANIC_REFERENCE_THICKNESS_M`]; tectonics thinning a cell towards its
/// trench thickness is the only thing that turns this on.
#[inline]
pub fn trench_deflection_m(crust_thickness_m: f32, oceanic: bool) -> f32 {
    if !oceanic {
        return 0.0;
    }
    let thinning = (OCEANIC_REFERENCE_THICKNESS_M - crust_thickness_m.max(0.0)).max(0.0);
    (thinning * TRENCH_FLEXURE_GAIN).min(MAX_TRENCH_DEFLECTION_M)
}

/// The full local solve one cell contributes to [`update_elevation`]: Airy
/// buoyancy plus the lid terms, less any trench bending deflection.
#[inline]
pub fn local_elevation_m(
    crust_thickness_m: f32,
    crust_density_kg_m3: f32,
    oceanic: bool,
    sediment_m: f32,
    ice_thickness_m: f32,
) -> f32 {
    airy_elevation_m(
        crust_thickness_m,
        crust_density_kg_m3,
        oceanic,
        sediment_m,
        ice_thickness_m,
    ) - trench_deflection_m(crust_thickness_m, oceanic)
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
            local_elevation_m(
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
        // Bands moved with the abyssal anchor: the lid no longer has to carry
        // the whole continent-to-abyss step on its own (see the module docs and
        // `ANCHOR_OCEAN_OLD_ELEV_M`), so the thermal part shrank from 21.2 km to
        // 9.5 km and the age-independent residual took over the rest.
        assert!((OCEAN_LID_RESIDUAL_M - 1363.636).abs() < 0.01);
        assert!((THERMAL_LID_M - 9_500.0).abs() < 1.0);
    }

    #[test]
    fn anchors_are_exact_before_flexure() {
        let cont = airy_elevation_m(35_000.0, 2700.0, false, 0.0, 0.0);
        assert!((cont - 800.0).abs() < 1.0, "continental anchor {cont}");
        // Ridge crest raised 300 m and the abyssal plain 1364 m relative to the
        // pre-calibration anchors; both are now stated constants.
        let ridge = airy_elevation_m(7_000.0, 3000.0, true, 0.0, 0.0);
        assert!(
            (ridge - ANCHOR_RIDGE_ELEV_M).abs() < 1.0,
            "ridge anchor {ridge}"
        );
        let abyss = airy_elevation_m(7_000.0, 3300.0, true, 0.0, 0.0);
        assert!(
            (abyss - ANCHOR_OCEAN_OLD_ELEV_M).abs() < 1.0,
            "abyssal anchor {abyss}"
        );
    }

    #[test]
    fn trench_bends_the_down_going_plate_below_the_abyssal_plain() {
        let abyss = local_elevation_m(7_000.0, 3300.0, true, 0.0, 0.0);
        let trench = local_elevation_m(5_000.0, 3300.0, true, 0.0, 0.0);
        assert!((abyss - trench - 5_000.0).abs() < 1.0, "trench {trench}");
        // Continental crust of the same thickness is not a trench.
        assert_eq!(trench_deflection_m(5_000.0, false), 0.0);
    }

    #[test]
    fn ice_depresses_by_density_ratio() {
        let bare = airy_elevation_m(35_000.0, 2700.0, false, 0.0, 0.0);
        let iced = airy_elevation_m(35_000.0, 2700.0, false, 0.0, 3000.0);
        let expected = 3000.0 * RHO_ICE_KG_M3 / RHO_MANTLE_KG_M3;
        assert!((bare - iced - expected).abs() < 1.0);
    }
}
