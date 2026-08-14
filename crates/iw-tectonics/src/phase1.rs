//! Phase 1, Pangaea-first: the planet is born with ONE supercontinent (plus at
//! most two microcontinents) whose fractal outline and interior structure come
//! from [`crate::craton::Genesis`]. The crustal-formation era differentiates it
//! (shield cores, mobile belts, weakness network) and the drift era then carves
//! it apart along rifts — so fragment shapes come from the rift graph and fit
//! each other like South America fits Africa, instead of betraying a bottom-up
//! assembly of separate blobs.
//!
//! Everything is a pure function of `(seed, config, mesh)`: the genesis shapes
//! are rebuilt identically by any process instance, and nothing about them is
//! stored outside `Planet`.

use glam::DVec3;
use iw_core::noise::Noise3;
use iw_core::planet::cell_flags;
use iw_core::{rng_for, CrustType, Planet, Plate, StepCtx};
use iw_mesh::Mesh;
use rand::Rng;
use rayon::prelude::*;

use crate::craton::Genesis;
use crate::crust::{make_continental, make_fresh_oceanic};
use crate::geom::{omega_for_velocity, random_unit, tangent_toward};
use crate::topology::{compact_plates, enforce_contiguity, UNASSIGNED};
use crate::{
    MeshCache, Scratch, AGE_DENSITY_COEFF, OCEANIC_DENSITY_KG_M3, OCEANIC_DENSITY_MAX_KG_M3,
};

/// Landmass drift during the crustal-formation era, m/yr. Slow: the era is
/// about differentiation, not travel — real motion starts with the hand-off.
const GENESIS_SPEED_M_YR: f64 = 0.02;
/// Speed given to pure-ocean plates invented at the hand-off, m/yr.
const SEED_PLATE_SPEED_M_YR: f32 = 0.04;
/// Divergence kick applied to both sides of every hand-off split, m/yr.
/// Gentler than a live rift's separation (0.07): this is a nudge for slab
/// pull to amplify, not an imposed breakup.
const HANDOFF_SPLIT_KICK_M_YR: f32 = 0.03;
/// Radial push given to every continental-share plate away from Pangaea's
/// centre of mass at the hand-off, m/yr — the coherent break-up bias that
/// keeps fragments from starting out clumped. Comparable to a live rift's
/// separation; slab pull and collisions take over from there.
const CONTINENTAL_DISPERSAL_KICK_M_YR: f32 = 0.05;
/// Hard ceiling on the initial plate count (safety, not a target).
const HANDOFF_MAX_PLATES: usize = 24;
/// Meander amplitude of a hand-off cut, in units of the axis coordinate
/// (which spans -1..1 across the sphere): the boundary wanders by roughly
/// this fraction of a hemisphere instead of running as a clean circle.
const HANDOFF_CUT_MEANDER: f32 = 0.22;
/// Frequency of the meander noise on the unit sphere.
const HANDOFF_CUT_FREQ: f32 = 2.6;
/// Fraction of cells flagged as inherited weakness (proto-sutures) at the
/// hand-off, so later rifting can follow noise creases across the whole
/// supercontinent.
const WEAKNESS_FRACTION: f32 = 0.03;
/// Frequency of the ridged field those creases are drawn from.
const WEAKNESS_FREQ: f32 = 3.5;
const WEAKNESS_OCTAVES: u32 = 4;

/// Minimum great-circle separation, in metres, the genesis placement enforces
/// between any two shield-core centres for this `(seed, count)`.
///
/// Exposed so tests (and tuning UIs) can check spacing without duplicating the
/// sampling constants. The pitch only affects outline octaves, never radii, so
/// any pitch reproduces the same radii stream.
pub fn craton_min_separation_m(seed: u64, count: u32) -> f64 {
    let genesis = Genesis::new(seed, count as usize, 70_000.0);
    let radii = genesis.core_radii_m();
    let mut min = f64::INFINITY;
    for i in 0..radii.len() {
        for j in i + 1..radii.len() {
            min = min.min((radii[i] + radii[j]) * 0.55);
        }
    }
    if min.is_finite() {
        min
    } else {
        0.0
    }
}

/// One CrustalFormation step. The landmasses are static (bar a slow wobble the
/// hand-off inherits); the era's work — outline, cores, thickness texture — is
/// stamped once at seeding, and the ocean ages underneath.
/// Fraction of the crustal-formation era over which the supercontinent
/// assembles (shield cores first, platforms accreting outward to the fractal
/// rim). The remainder of the era is quiet differentiation and ocean aging.
const ASSEMBLY_FRACTION: f64 = 0.7;

#[allow(clippy::too_many_arguments)]
pub(crate) fn step(
    planet: &mut Planet,
    mesh: &Mesh,
    dt_myr: f64,
    ctx: &mut StepCtx,
    cache: &MeshCache,
    genesis: &Genesis,
    members: &[Option<(u16, f32)>],
    scratch: &mut Scratch,
) {
    let _ = scratch;
    if planet.plates.is_empty() {
        seed_proto_crust(planet, ctx, cache, genesis);
    }
    // Progressive assembly: a cell joins the continent once the era has
    // advanced past its radial coordinate (0 at a shield core, 1 at the rim),
    // so the landmass grows outward from its cores on screen instead of the
    // whole map being stamped at step zero and nothing visibly happening for
    // 200 Myr. Deterministic: purely a function of (time, membership).
    let duration = planet
        .config
        .duration_myr(iw_core::Phase::CrustalFormation)
        .max(1e-6);
    let reveal = (planet.time_myr / (duration * ASSEMBLY_FRACTION)).min(1.0) as f32;
    let mut grown = 0usize;
    let mut total = 0usize;
    for (c, member) in members.iter().enumerate() {
        if let Some((mass, f)) = member {
            total += 1;
            if *f <= reveal && planet.crust_type[c] != CrustType::Continental {
                let thickness = genesis.target_thickness_m(*mass, mesh.centers[c], *f);
                make_continental(planet, c as u32, thickness, cache.area_m2[c], ctx.ledger);
                planet.plate_id[c] = *mass;
                grown += 1;
            }
        }
    }
    if grown > 0 && reveal >= 1.0 {
        ctx.progress
            .event(iw_core::ProgressEvent::Milestone(format!(
                "the supercontinent is assembled: {:.0}% of the surface, {} landmass{}",
                100.0 * total as f64 / planet.n_cells() as f64,
                genesis.n_masses(),
                if genesis.n_masses() == 1 { "" } else { "es" },
            )));
    }
    age_ocean(planet, dt_myr);
}

/// Lay down the global basaltic proto-crust and the (slowly wobbling) plate
/// per landmass; the continent itself accretes step by step in [`step`].
fn seed_proto_crust(planet: &mut Planet, ctx: &mut StepCtx, cache: &MeshCache, genesis: &Genesis) {
    let n = planet.n_cells();
    for c in 0..n as u32 {
        planet.plate_id[c as usize] = UNASSIGNED;
        make_fresh_oceanic(
            planet,
            c,
            cache.ocean_thickness_m[c as usize],
            cache.area_m2[c as usize],
            ctx.ledger,
        );
    }

    // One plate per landmass, wobbling slowly until the hand-off.
    let vigor = planet.config.tectonic_vigor as f64;
    let omega = GENESIS_SPEED_M_YR * vigor * 1.0e6 / iw_mesh::EARTH_RADIUS_M;
    for _ in 0..genesis.n_masses() {
        let pole = random_unit(&mut ctx.rng).as_dvec3();
        let pole = if pole.length_squared() > 1e-12 {
            pole.normalize()
        } else {
            DVec3::Z
        };
        planet.plates.push(Plate {
            euler_pole: pole,
            omega_rad_myr: omega,
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        });
    }

    ctx.progress
        .event(iw_core::ProgressEvent::Milestone(format!(
            "crustal accretion begins: {} landmass{} forming",
            genesis.n_masses(),
            if genesis.n_masses() == 1 { "" } else { "es" },
        )));
}

/// Advance oceanic crust age and its density (thermal subsidence channel).
pub(crate) fn age_ocean(planet: &mut Planet, dt_myr: f64) {
    let dt = dt_myr as f32;
    for c in 0..planet.n_cells() {
        if planet.crust_type[c] == CrustType::Oceanic {
            let age = planet.crust_age_myr[c] + dt;
            planet.crust_age_myr[c] = age;
            planet.crust_density_kg_m3[c] = (OCEANIC_DENSITY_KG_M3
                + AGE_DENSITY_COEFF * age.max(0.0).sqrt())
            .min(OCEANIC_DENSITY_MAX_KG_M3);
        } else {
            planet.crust_age_myr[c] = 0.0;
        }
    }
}

/// True when the phase-1 -> drift hand-off still has to run. Derived purely
/// from `Planet`: cells outside any landmass carry the [`UNASSIGNED`] plate id.
pub(crate) fn needs_partition(planet: &Planet) -> bool {
    let np = planet.plates.len();
    np == 0 || planet.plate_id.iter().any(|p| *p as usize >= np)
}

/// Partition every cell into the initial plate set by RECURSIVE SPLITTING:
/// the whole lithosphere starts as one plate, and the largest plate is
/// repeatedly cut at a random 30:70..50:50 area ratio along a
/// noise-meandered great-circle band, until no plate holds more than
/// [`HANDOFF_MAX_PLATE_FRAC`] of the surface. Cuts land wherever the split
/// axis puts them — which statistically means THROUGH the supercontinent,
/// since it covers a third of the sphere. The result is a mosaic of
/// varied-size plates spread over the whole globe instead of one continental
/// giant with a cluster of fillers at its antipode.
pub(crate) fn partition_into_plates(
    planet: &mut Planet,
    mesh: &Mesh,
    ctx: &mut StepCtx,
    scratch: &mut Scratch,
) {
    let n = planet.n_cells();
    let ncr = planet.plates.len();
    let area = |c: usize| mesh.areas_km2[c] as f64;
    let total_area: f64 = mesh.areas_km2.iter().map(|a| *a as f64).sum();

    // Everything starts on plate 0; recursive splits carve it up.
    scratch.u16a.fill(0);
    let mut plate_area: Vec<f64> = vec![total_area];
    // (parent, child) of each split, in order: rift partners come from here.
    let mut splits: Vec<(u16, u16)> = Vec::new();
    let cut_noise = Noise3::new(noise_seed(planet.config.seed, "tectonics/handoff-cuts"));

    // The largest-plate cap is the user's mosaic knob (config
    // `max_plate_fraction`): 0.15 yields ~10-14 varied plates, large values
    // approach one supercontinental plate.
    let max_frac = planet.config.max_plate_fraction.clamp(0.08, 0.60);
    while plate_area.len() < HANDOFF_MAX_PLATES {
        let (p, biggest) = plate_area
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("at least one plate");
        if *biggest <= max_frac * total_area {
            break;
        }
        let p = p as u16;
        let ratio: f64 = ctx.rng.random_range(0.30..0.50);
        let axis = random_unit(&mut ctx.rng);
        // Score = position along the axis plus a meander term, so the cut is
        // a wandering band instead of a clean small circle. The noise is
        // offset per split so successive cuts do not share their wiggles.
        let offset = glam::Vec3::splat(splits.len() as f32 * 13.7 + 3.1);
        let mut members: Vec<(f32, u32)> = (0..n as u32)
            .filter(|c| scratch.u16a[*c as usize] == p)
            .map(|c| {
                let d = mesh.centers[c as usize];
                let score = d.dot(axis)
                    + HANDOFF_CUT_MEANDER
                        * cut_noise.fbm(d * HANDOFF_CUT_FREQ + offset, 3, 2.0, 0.5);
                (score, c)
            })
            .collect();
        if members.len() < 4 {
            break;
        }
        members.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let target = plate_area[p as usize] * ratio;
        let child = plate_area.len() as u16;
        let mut moved = 0.0f64;
        for (_, c) in &members {
            if moved >= target {
                break;
            }
            scratch.u16a[*c as usize] = child;
            moved += area(*c as usize);
        }
        if moved <= 0.0 || moved >= plate_area[p as usize] {
            break;
        }
        plate_area[p as usize] -= moved;
        plate_area.push(moved);
        splits.push((p, child));
    }

    // Build the plate table. A plate inherits the (mass-weighted) motion of
    // the genesis landmasses under it; pure ocean plates get a slow start in
    // a random direction. Everyone gets a plain centroid for the kicks.
    let np = plate_area.len();
    let mut cont_omega = vec![DVec3::ZERO; np];
    let mut cont_mass = vec![0f64; np];
    let mut centroid = vec![DVec3::ZERO; np];
    for c in 0..n {
        let l = scratch.u16a[c] as usize;
        centroid[l] += mesh.centers[c].as_dvec3() * area(c);
        let old = planet.plate_id[c] as usize;
        if old < ncr && planet.crust_type[c] == CrustType::Continental {
            cont_omega[l] +=
                planet.plates[old].euler_pole * planet.plates[old].omega_rad_myr * area(c);
            cont_mass[l] += area(c);
        }
    }
    let mut plates: Vec<Plate> = Vec::with_capacity(np);
    for i in 0..np {
        let w = if cont_mass[i] > 0.0 {
            cont_omega[i] / cont_mass[i]
        } else {
            let r = centroid[i].normalize_or(DVec3::Z).as_vec3();
            let dir = tangent_toward(r, random_unit(&mut ctx.rng));
            omega_for_velocity(r, dir * SEED_PLATE_SPEED_M_YR)
        };
        let omega = w.length();
        plates.push(Plate {
            euler_pole: if omega > 1e-12 { w / omega } else { DVec3::Z },
            omega_rad_myr: omega,
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        });
    }

    // Each split pair starts with near-identical motion, and QUIET_MERGE
    // would stitch the mosaic back together within a dozen Myr — before slab
    // pull has differentiated anything. Give every pair the same young-rift
    // immunity as a live rift plus a gentle divergence kick to amplify.
    // Later splits overwrite a plate's partner: its freshest boundary is the
    // one that needs the protection most.
    for &(a, b) in &splits {
        let (ai, bi) = (a as usize, b as usize);
        let ca = centroid[ai].normalize_or(DVec3::Z).as_vec3();
        let cb = centroid[bi].normalize_or(DVec3::Z).as_vec3();
        let kick = |plate: &mut Plate, from: glam::Vec3, away_from: glam::Vec3| {
            let w = plate.euler_pole * plate.omega_rad_myr
                + omega_for_velocity(
                    from,
                    tangent_toward(from, away_from) * -HANDOFF_SPLIT_KICK_M_YR,
                );
            let omega = w.length();
            plate.euler_pole = if omega > 1e-12 { w / omega } else { DVec3::Z };
            plate.omega_rad_myr = omega;
        };
        kick(&mut plates[ai], ca, cb);
        kick(&mut plates[bi], cb, ca);
        plates[ai].rift_partner = Some(b);
        plates[ai].rift_born_myr = planet.time_myr;
        plates[bi].rift_partner = Some(a);
        plates[bi].rift_born_myr = planet.time_myr;
    }

    // Continental dispersal: every plate carrying a real share of the
    // supercontinent also gets a push AWAY from Pangaea's centre of mass —
    // a coherent radial break-up bias, so the fragments start by separating
    // instead of milling about clumped together. Collisions later are still
    // free to happen; this only sets the opening act.
    let total_cont: f64 = cont_mass.iter().sum();
    if total_cont > 0.0 {
        let pangaea: DVec3 = (0..np)
            .fold(DVec3::ZERO, |acc, i| {
                acc + centroid[i].normalize_or(DVec3::ZERO) * cont_mass[i]
            })
            .normalize_or(DVec3::Z);
        let pv = pangaea.as_vec3();
        for i in 0..np {
            if cont_mass[i] < 0.02 * total_cont {
                continue;
            }
            let here = centroid[i].normalize_or(DVec3::Z).as_vec3();
            let w = plates[i].euler_pole * plates[i].omega_rad_myr
                + omega_for_velocity(
                    here,
                    tangent_toward(here, pv) * -CONTINENTAL_DISPERSAL_KICK_M_YR,
                );
            let omega = w.length();
            plates[i].euler_pole = if omega > 1e-12 { w / omega } else { DVec3::Z };
            plates[i].omega_rad_myr = omega;
        }
    }

    planet.plate_id.copy_from_slice(&scratch.u16a);
    planet.plates = plates;
    // Absorb every stray fragment (no calving here): the hand-off must produce
    // exactly the plate count the Voronoi asked for.
    enforce_contiguity(
        planet,
        mesh,
        &mut scratch.u32a,
        usize::MAX,
        crate::MAX_PLATES,
    );
    compact_plates(planet);
    mark_inherited_weakness(planet, mesh);

    ctx.progress
        .event(iw_core::ProgressEvent::Milestone(format!(
            "the drift era begins: {} plates",
            planet.plates.len()
        )));
    log::debug!("initial partition: {} plates", planet.plates.len());
}

/// Lay a sparse network of proto-sutures along the creases of a ridged noise
/// field. `SUTURE` is the only flag that survives a step, and `drift::split_plate`
/// prefers to nucleate rifts on it, so this is how the supercontinent inherits
/// the weakness web its breakup will follow. The threshold is a rank, not a
/// value, so the flagged fraction is exactly [`WEAKNESS_FRACTION`] whatever the
/// field's distribution.
fn mark_inherited_weakness(planet: &mut Planet, mesh: &Mesh) {
    let n = planet.n_cells();
    if n == 0 {
        return;
    }
    let noise = Noise3::new(noise_seed(planet.config.seed, "tectonics/weakness"));
    let field: Vec<f32> = mesh
        .centers
        .par_iter()
        .map(|d| noise.ridged(*d * WEAKNESS_FREQ, WEAKNESS_OCTAVES, 2.0, 0.5))
        .collect();
    let mut sorted = field.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let idx = (((1.0 - WEAKNESS_FRACTION) * n as f32) as usize).min(n - 1);
    let threshold = sorted[idx];
    for (c, v) in field.iter().enumerate() {
        if *v >= threshold {
            planet.tectonic_flags[c] |= cell_flags::SUTURE;
        }
    }
}

/// Stable per-purpose noise seed, derived the same way every other stream in
/// the crate is: from the planet seed and a fixed label.
pub(crate) fn noise_seed(seed: u64, label: &str) -> u64 {
    rng_for(seed, label, 0).random::<u64>()
}
