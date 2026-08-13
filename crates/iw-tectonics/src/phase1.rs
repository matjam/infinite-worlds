//! Phase 1: craton seeding, drift, accretion, and the hand-off partition into
//! the initial plate set (DESIGN.md §5 Phase 1).
//!
//! Cratons are continental nuclei whose outline comes from 3D noise (see
//! [`crate::craton`]). Their *shape* is a pure function of
//! `(seed, craton index, mesh pitch)` so it can be recomputed at any time; their
//! *position* is recomputed from the cells they currently own. Nothing about a
//! craton is stored outside `Planet`, which is what lets a fresh process
//! instance continue a checkpoint exactly.

use glam::{DVec3, Vec3};
use iw_core::noise::Noise3;
use iw_core::planet::cell_flags;
use iw_core::{rng_for, CrustType, Phase, Planet, Plate, StepCtx};
use iw_mesh::{Mesh, EARTH_RADIUS_M};
use rand::Rng;
use rayon::prelude::*;

use crate::craton::{CratonSet, CratonShape};
use crate::crust::{make_continental, make_fresh_oceanic};
use crate::geom::{arc_m, omega_for_velocity, random_unit, rotate_about, tangent_toward};
use crate::topology::{
    bfs_distance, compact_plates, enforce_contiguity, flood_labels_warped, UnionFind, UNASSIGNED,
};
use crate::{
    MeshCache, Scratch, AGE_DENSITY_COEFF, OCEANIC_DENSITY_KG_M3, OCEANIC_DENSITY_MAX_KG_M3,
};

/// Poisson-disc rejection distance as a multiple of the two radii summed.
const SEPARATION_FACTOR: f64 = 1.02;
/// Floor the rejection distance relaxes to when the sphere gets crowded.
const SEPARATION_FLOOR: f64 = 0.72;
/// Rejection attempts per craton before the seeder gives up on that one.
const PLACEMENT_TRIES: u32 = 8_192;
/// The rejection distance is relaxed by 2% after every this many failed tries.
const RELAX_EVERY: u32 = 96;
/// Craton drift speed at vigor 1, m/yr. Early-Earth convection is vigorous;
/// this is what lets cratons cross an ocean inside a 200 Myr phase.
const CRATON_SPEED_M_YR: f64 = 0.09;
/// Random-walk rate of a craton's Euler pole, radians per Myr.
const POLE_WANDER_RAD_MYR: f64 = 0.035;
/// Fraction of the phase after which the supercontinent attractor engages.
const ATTRACTOR_START_FRAC: f64 = 0.75;
/// Blend rate of a craton's pole toward the supercontinent attractor, per Myr.
const ATTRACTOR_RATE_PER_MYR: f64 = 0.10;
/// Plate count the hand-off partition aims for.
const TARGET_PLATES_MIN: usize = 5;
/// Plate count the hand-off partition will not exceed.
const TARGET_PLATES_MAX: usize = 8;
/// Speed given to the ocean plates invented at the hand-off, m/yr.
const SEED_PLATE_SPEED_M_YR: f32 = 0.04;
/// Rate at which a craton cell's thickness relaxes onto the craton's radial
/// profile, per Myr (tau = 25 Myr).
///
/// Without this, thickness is only ever written at the instant a cell is
/// claimed, so every cell of a drifting craton keeps the *leading-edge* value
/// it was born with and the whole landmass ends up uniformly thin — the
/// core-to-edge profile only ever existed for cratons that never moved. Letting
/// it relax makes the profile travel with the craton and also erases any
/// residual edge-flicker banding.
const PROFILE_RATE_PER_MYR: f64 = 0.04;
/// Distance-metric warp for the hand-off Voronoi, as a fraction of the base
/// cost. Straight great-circle plate boundaries are the tell-tale of a Voronoi
/// partition; a noisy metric makes them meander.
const PARTITION_WARP: f32 = 0.45;
/// Frequency of that warp field on the unit sphere.
const PARTITION_WARP_FREQ: f32 = 3.0;
/// Fraction of cells flagged as inherited weakness (proto-sutures) at the
/// hand-off, so later rifting can follow noise creases and not only the seams
/// where cratons actually collided.
const WEAKNESS_FRACTION: f32 = 0.03;
/// Frequency of the ridged field those creases are drawn from.
const WEAKNESS_FREQ: f32 = 3.5;
const WEAKNESS_OCTAVES: u32 = 4;

/// Minimum great-circle separation, in metres, that the craton seeder enforces
/// between any two craton centres for this `(seed, count)`.
///
/// Exposed so tests (and tuning UIs) can check spacing without duplicating the
/// sampling constants.
pub fn craton_min_separation_m(seed: u64, count: u32) -> f64 {
    let radii = CratonSet::radii_m(seed, count as usize);
    let mut min = f64::INFINITY;
    for i in 0..radii.len() {
        for j in i + 1..radii.len() {
            min = min.min((radii[i] + radii[j]) * SEPARATION_FLOOR);
        }
    }
    if min.is_finite() {
        min
    } else {
        0.0
    }
}

/// One CrustalFormation step.
pub(crate) fn step(
    planet: &mut Planet,
    mesh: &Mesh,
    dt_myr: f64,
    ctx: &mut StepCtx,
    cache: &MeshCache,
    cratons: &CratonSet,
    scratch: &mut Scratch,
) {
    let shapes = cratons.shapes();
    let centers = if planet.plates.is_empty() {
        seed_planet(planet, ctx, cache, shapes)
    } else {
        advect(planet, mesh, dt_myr, ctx)
    };

    rasterize_and_apply(planet, mesh, cache, ctx, shapes, &centers, dt_myr, scratch);
    age_ocean(planet, dt_myr);
    weld_contacts(planet, mesh, ctx);
}

/// Lay down the global basaltic proto-crust and place the craton nuclei.
/// Returns the craton centres, in plate-index order.
fn seed_planet(
    planet: &mut Planet,
    ctx: &mut StepCtx,
    cache: &MeshCache,
    shapes: &[CratonShape],
) -> Vec<Vec3> {
    for c in 0..planet.n_cells() as u32 {
        planet.plate_id[c as usize] = UNASSIGNED;
        make_fresh_oceanic(
            planet,
            c,
            cache.ocean_thickness_m[c as usize],
            cache.area_m2[c as usize],
            ctx.ledger,
        );
    }

    // Poisson-disc rejection sampling. Cratons sized for a quarter of the
    // planet pack tightly, so the required spacing relaxes toward
    // `SEPARATION_FLOOR` rather than dropping the craton entirely.
    let mut centers: Vec<Vec3> = Vec::with_capacity(shapes.len());
    for (i, shape) in shapes.iter().enumerate() {
        let mut factor = SEPARATION_FACTOR;
        for attempt in 0..PLACEMENT_TRIES {
            if attempt > 0 && attempt % RELAX_EVERY == 0 {
                factor = (factor * 0.98).max(SEPARATION_FLOOR);
            }
            let cand = random_unit(&mut ctx.rng);
            let ok = centers
                .iter()
                .enumerate()
                .all(|(j, c)| arc_m(*c, cand) >= (shape.radius_m + shapes[j].radius_m) * factor);
            if ok {
                centers.push(cand);
                break;
            }
        }
        if centers.len() == i {
            log::debug!("craton {i} could not be placed with the required spacing; skipped");
        }
    }

    // One plate per craton. Poles are perpendicular to the craton centre so the
    // surface speed there is exactly omega * R.
    let vigor = planet.config.tectonic_vigor as f64;
    let omega = CRATON_SPEED_M_YR * vigor * 1.0e6 / EARTH_RADIUS_M;
    for center in &centers {
        let t = random_unit(&mut ctx.rng);
        let mut pole = center.as_dvec3().cross(t.as_dvec3());
        if pole.length_squared() < 1e-12 {
            pole = center.as_dvec3().cross(DVec3::Z);
        }
        planet.plates.push(Plate {
            euler_pole: pole.normalize(),
            omega_rad_myr: omega,
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
        });
    }
    log::debug!(
        "seeded {} cratons of {} requested",
        centers.len(),
        shapes.len()
    );
    centers
}

/// Advance every craton's Euler pole, then rotate its centre by one step.
fn advect(planet: &mut Planet, mesh: &Mesh, dt_myr: f64, ctx: &mut StepCtx) -> Vec<Vec3> {
    let ncr = planet.plates.len();
    let n = planet.n_cells();

    // Current craton positions: centroid of the cells each one owns.
    let mut sum = vec![DVec3::ZERO; ncr];
    let mut count = vec![0u32; ncr];
    for c in 0..n {
        let p = planet.plate_id[c];
        if (p as usize) < ncr && planet.crust_type[c] == CrustType::Continental {
            sum[p as usize] += mesh.centers[c].as_dvec3();
            count[p as usize] += 1;
        }
    }

    // Welded cratons form rigid groups and share a pole.
    let mut uf = UnionFind::new(ncr);
    for k in 0..ncr {
        if let Some(w) = planet.plates[k].welded_to {
            if (w as usize) < ncr {
                uf.union(k as u16, w);
            }
        }
    }
    let root: Vec<u16> = (0..ncr).map(|k| uf.find(k as u16)).collect();

    // Global continental centroid: the supercontinent attractor's target.
    let mut global = DVec3::ZERO;
    for s in sum.iter() {
        global += *s;
    }
    let global = if global.length_squared() > 1e-12 {
        global.normalize().as_vec3()
    } else {
        Vec3::Z
    };
    let duration = planet
        .config
        .duration_myr(Phase::CrustalFormation)
        .max(1e-9);
    let late = planet.time_myr / duration >= ATTRACTOR_START_FRAC;

    for r in 0..ncr {
        if root[r] != r as u16 {
            continue;
        }
        // Group centroid, weighted by cell count.
        let mut g = DVec3::ZERO;
        let mut members = 0u32;
        for (k, rk) in root.iter().enumerate() {
            if *rk == r as u16 {
                g += sum[k];
                members += count[k];
            }
        }
        if members == 0 {
            continue;
        }
        let g = g.normalize();

        let mut pole = planet.plates[r].euler_pole;
        // Slow random walk: the low-order convection proxy.
        let axis = random_unit(&mut ctx.rng).as_dvec3();
        let wander = POLE_WANDER_RAD_MYR * dt_myr * ctx.rng.random_range(-1.0f64..1.0);
        pole = rotate_about(pole, axis, wander).normalize();

        if late {
            let gv = g.as_vec3();
            let u = tangent_toward(gv, global);
            let target = gv.as_dvec3().cross(u.as_dvec3());
            if target.length_squared() > 1e-12 {
                let w = (ATTRACTOR_RATE_PER_MYR * dt_myr).clamp(0.0, 1.0);
                pole = (pole * (1.0 - w) + target.normalize() * w).normalize();
            }
        }

        let omega = planet.plates[r].omega_rad_myr;
        for (k, rk) in root.iter().enumerate() {
            if *rk == r as u16 {
                planet.plates[k].euler_pole = pole;
                planet.plates[k].omega_rad_myr = omega;
            }
        }
    }

    (0..ncr)
        .map(|k| {
            if count[k] == 0 {
                return Vec3::ZERO;
            }
            let c = sum[k].normalize();
            let plate = &planet.plates[k];
            rotate_about(c, plate.euler_pole, plate.omega_rad_myr * dt_myr)
                .normalize()
                .as_vec3()
        })
        .collect()
}

/// Stamp the craton shapes onto the mesh and reconcile with the previous step:
/// leading-edge cells become continental, trailing-edge cells revert to ocean.
///
/// The test is done in each craton's own frame (see [`crate::craton`]), which
/// is what makes the outline rigid under drift. It is a pure per-cell map —
/// each cell asks every craton whose bounding cone it falls in, lowest index
/// wins — so it parallelizes without any order dependence.
#[allow(clippy::too_many_arguments)]
fn rasterize_and_apply(
    planet: &mut Planet,
    mesh: &Mesh,
    cache: &MeshCache,
    ctx: &mut StepCtx,
    shapes: &[CratonShape],
    centers: &[Vec3],
    dt_myr: f64,
    scratch: &mut Scratch,
) {
    let n = planet.n_cells();

    // Per-craton placement: local frame, cap pole in world space, cull cone.
    struct Placed<'a> {
        k: u16,
        shape: &'a CratonShape,
        frame: glam::Quat,
        pole: Vec3,
        cos_bound: f32,
    }
    let placed: Vec<Placed> = centers
        .iter()
        .enumerate()
        .filter(|(_, c)| c.length_squared() >= 0.5) // Vec3::ZERO == extinct
        .map(|(k, center)| {
            let shape = &shapes[k.min(shapes.len() - 1)];
            let frame = shape.frame(*center);
            Placed {
                k: k as u16,
                shape,
                frame,
                pole: shape.pole_world(frame),
                cos_bound: shape.cos_bound(),
            }
        })
        .collect();

    // scratch.u16a: craton claiming each cell (UNASSIGNED = ocean).
    // scratch.f32a: that craton's target crustal thickness there, metres.
    scratch
        .u16a
        .par_iter_mut()
        .zip(scratch.f32a.par_iter_mut())
        .zip(mesh.centers.par_iter())
        .for_each(|((claim, thickness), cc)| {
            *claim = UNASSIGNED;
            *thickness = 0.0;
            for p in &placed {
                if cc.dot(p.pole) < p.cos_bound {
                    continue;
                }
                let q = p.frame * *cc;
                if let Some(f) = p.shape.contains(q) {
                    *claim = p.k;
                    *thickness = p.shape.thickness_m(q, f);
                    return;
                }
            }
        });

    // A craton always keeps the cell under its pole, however the noise fell.
    for p in &placed {
        let c = mesh.cell_at(p.pole) as usize;
        if scratch.u16a[c] == UNASSIGNED {
            scratch.u16a[c] = p.k;
            scratch.f32a[c] = p.shape.thickness_m(p.frame * mesh.centers[c], 0.0);
        }
    }

    let profile_w = (PROFILE_RATE_PER_MYR * dt_myr).clamp(0.0, 1.0) as f32;
    for c in 0..n as u32 {
        let claim = scratch.u16a[c as usize];
        let was_continental = planet.crust_type[c as usize] == CrustType::Continental;
        let area = cache.area_m2[c as usize];
        if claim != UNASSIGNED {
            planet.plate_id[c as usize] = claim;
            let target = scratch.f32a[c as usize];
            if !was_continental {
                make_continental(planet, c, target, area, ctx.ledger);
            } else {
                // Relax onto the profile so it drifts with the craton.
                let th = planet.crust_thickness_m[c as usize];
                planet.crust_thickness_m[c as usize] = th + (target - th) * profile_w;
            }
        } else {
            planet.plate_id[c as usize] = UNASSIGNED;
            if was_continental {
                make_fresh_oceanic(
                    planet,
                    c,
                    cache.ocean_thickness_m[c as usize],
                    area,
                    ctx.ledger,
                );
            }
        }
    }
}

/// Oceanic crust ages and densifies; thermal subsidence follows in geology.
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

/// Weld cratons that have come into contact and flag the seam.
fn weld_contacts(planet: &mut Planet, mesh: &Mesh, ctx: &mut StepCtx) {
    let ncr = planet.plates.len();
    let mut uf = UnionFind::new(ncr);
    for k in 0..ncr {
        if let Some(w) = planet.plates[k].welded_to {
            if (w as usize) < ncr {
                uf.union(k as u16, w);
            }
        }
    }
    let mut new_welds = 0u32;
    for c in 0..planet.n_cells() as u32 {
        let p = planet.plate_id[c as usize];
        if p as usize >= ncr {
            continue;
        }
        for &m in mesh.neighbors_of(c) {
            let q = planet.plate_id[m as usize];
            if q as usize >= ncr || q == p {
                continue;
            }
            planet.tectonic_flags[c as usize] |= cell_flags::SUTURE | cell_flags::COLLISION;
            planet.tectonic_flags[m as usize] |= cell_flags::SUTURE | cell_flags::COLLISION;
            if uf.union(p, q) {
                new_welds += 1;
            }
        }
    }
    if new_welds == 0 {
        return;
    }

    // Mass-weighted merge of each group's motion, then propagate to members.
    let mut count = vec![0f64; ncr];
    for c in 0..planet.n_cells() {
        let p = planet.plate_id[c];
        if (p as usize) < ncr {
            count[p as usize] += 1.0;
        }
    }
    let root: Vec<u16> = (0..ncr).map(|k| uf.find(k as u16)).collect();
    for r in 0..ncr {
        if root[r] != r as u16 {
            continue;
        }
        let mut w = DVec3::ZERO;
        let mut total = 0.0;
        for (k, rk) in root.iter().enumerate() {
            if *rk == r as u16 {
                w += planet.plates[k].euler_pole * planet.plates[k].omega_rad_myr * count[k];
                total += count[k];
            }
        }
        if total <= 0.0 {
            continue;
        }
        w /= total;
        let omega = w.length();
        let pole = if omega > 1e-12 {
            w / omega
        } else {
            planet.plates[r].euler_pole
        };
        for (k, rk) in root.iter().enumerate() {
            if *rk != r as u16 {
                continue;
            }
            planet.plates[k].euler_pole = pole;
            planet.plates[k].omega_rad_myr = omega;
            planet.plates[k].welded_to = if k == r { None } else { Some(r as u16) };
        }
    }
    ctx.progress
        .event(iw_core::ProgressEvent::Milestone(format!(
            "{new_welds} craton{} accreted at {:.0} Myr",
            if new_welds == 1 { "" } else { "s" },
            planet.time_myr
        )));
}

/// True when the phase-1 -> drift hand-off still has to run. Derived purely
/// from `Planet`: cells outside any craton carry the [`UNASSIGNED`] plate id.
pub(crate) fn needs_partition(planet: &Planet) -> bool {
    let np = planet.plates.len();
    np == 0 || planet.plate_id.iter().any(|p| *p as usize >= np)
}

/// Partition every cell into the initial plate set: one plate per welded craton
/// cluster plus its surrounding ocean (graph Voronoi), topped up with pure
/// ocean plates so the planet starts Drift with 5-8 plates.
pub(crate) fn partition_into_plates(
    planet: &mut Planet,
    mesh: &Mesh,
    ctx: &mut StepCtx,
    scratch: &mut Scratch,
) {
    let n = planet.n_cells();
    let ncr = planet.plates.len();

    // Craton -> cluster.
    let mut uf = UnionFind::new(ncr.max(1));
    for k in 0..ncr {
        if let Some(w) = planet.plates[k].welded_to {
            if (w as usize) < ncr {
                uf.union(k as u16, w);
            }
        }
    }
    let mut cluster_of = vec![u16::MAX; ncr];
    let mut cluster_omega: Vec<DVec3> = Vec::new();
    let mut cluster_mass: Vec<f64> = Vec::new();
    let mut mass = vec![0f64; ncr];
    for c in 0..n {
        let p = planet.plate_id[c];
        if (p as usize) < ncr && planet.crust_type[c] == CrustType::Continental {
            mass[p as usize] += 1.0;
        }
    }
    for k in 0..ncr {
        if mass[k] == 0.0 {
            continue;
        }
        let r = uf.find(k as u16) as usize;
        if cluster_of[r] == u16::MAX {
            cluster_of[r] = cluster_omega.len() as u16;
            cluster_omega.push(DVec3::ZERO);
            cluster_mass.push(0.0);
        }
        let ci = cluster_of[r] as usize;
        cluster_omega[ci] += planet.plates[k].euler_pole * planet.plates[k].omega_rad_myr * mass[k];
        cluster_mass[ci] += mass[k];
        cluster_of[k] = cluster_of[r];
    }

    // Seed labels from continental cells.
    scratch.u16a.fill(u16::MAX);
    let mut cluster_sum = vec![DVec3::ZERO; cluster_mass.len()];
    for c in 0..n {
        let p = planet.plate_id[c];
        if (p as usize) < ncr
            && planet.crust_type[c] == CrustType::Continental
            && cluster_of[p as usize] != u16::MAX
        {
            let ci = cluster_of[p as usize];
            scratch.u16a[c] = ci;
            cluster_sum[ci as usize] += mesh.centers[c].as_dvec3();
        }
    }

    // Too many clusters: fold the smallest into its nearest neighbour.
    while cluster_mass.len() > TARGET_PLATES_MAX {
        let small = (0..cluster_mass.len())
            .min_by(|a, b| cluster_mass[*a].total_cmp(&cluster_mass[*b]))
            .expect("non-empty");
        let sc = cluster_sum[small].normalize_or(DVec3::Z);
        let near = (0..cluster_mass.len())
            .filter(|k| *k != small)
            .max_by(|a, b| {
                sc.dot(cluster_sum[*a].normalize_or(DVec3::Z))
                    .total_cmp(&sc.dot(cluster_sum[*b].normalize_or(DVec3::Z)))
            })
            .expect("at least two clusters");
        let (w, m, s) = (
            cluster_omega[small],
            cluster_mass[small],
            cluster_sum[small],
        );
        cluster_omega[near] += w;
        cluster_mass[near] += m;
        cluster_sum[near] += s;
        cluster_omega.remove(small);
        cluster_mass.remove(small);
        cluster_sum.remove(small);
        for l in scratch.u16a.iter_mut() {
            if *l == u16::MAX {
                continue;
            }
            let k = *l as usize;
            *l = if k == small {
                (if near > small { near - 1 } else { near }) as u16
            } else if k > small {
                (k - 1) as u16
            } else {
                k as u16
            };
        }
    }

    // Too few: invent ocean plates at the points furthest from any continent.
    let n_extra = TARGET_PLATES_MIN.saturating_sub(cluster_mass.len());
    let mut extra_seeds: Vec<u32> = Vec::with_capacity(n_extra);
    if n_extra > 0 {
        scratch.u32a.resize(n, 0);
        for _ in 0..n_extra {
            let sources: Vec<u32> = (0..n as u32)
                .filter(|c| scratch.u16a[*c as usize] != u16::MAX || extra_seeds.contains(c))
                .collect();
            if sources.is_empty() {
                extra_seeds.push(0);
                continue;
            }
            bfs_distance(mesh, &sources, &mut scratch.u32a);
            let far = (0..n)
                .filter(|c| scratch.u32a[*c] != u32::MAX)
                .max_by_key(|c| scratch.u32a[*c])
                .unwrap_or(0) as u32;
            extra_seeds.push(far);
        }
    }

    let n_clusters = cluster_mass.len();
    for (i, &s) in extra_seeds.iter().enumerate() {
        scratch.u16a[s as usize] = (n_clusters + i) as u16;
    }
    // Noise-warped graph Voronoi: cheap-to-cross cells pull a plate's territory
    // out along the noise's low ridges, so the initial boundaries wander the way
    // real ones do instead of running as clean great-circle bisectors.
    let warp = Noise3::new(noise_seed(planet.config.seed, "tectonics/plate-warp"));
    let cost: Vec<f32> = mesh
        .centers
        .par_iter()
        .map(|d| (1.0 + PARTITION_WARP * warp.fbm(*d * PARTITION_WARP_FREQ, 4, 2.0, 0.5)).max(0.2))
        .collect();
    flood_labels_warped(mesh, &mut scratch.u16a, &cost);
    debug_assert!(scratch.u16a.iter().all(|l| *l != u16::MAX));

    // Build the plate table.
    let mut plates: Vec<Plate> = Vec::with_capacity(n_clusters + extra_seeds.len());
    for i in 0..n_clusters {
        let w = if cluster_mass[i] > 0.0 {
            cluster_omega[i] / cluster_mass[i]
        } else {
            DVec3::Z * 1e-6
        };
        let omega = w.length();
        plates.push(Plate {
            euler_pole: if omega > 1e-12 { w / omega } else { DVec3::Z },
            omega_rad_myr: omega,
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
        });
    }
    for &s in &extra_seeds {
        let r = mesh.centers[s as usize];
        let dir = {
            let t = random_unit(&mut ctx.rng);
            tangent_toward(r, t)
        };
        let w = omega_for_velocity(r, dir * SEED_PLATE_SPEED_M_YR);
        let omega = w.length();
        plates.push(Plate {
            euler_pole: if omega > 1e-12 { w / omega } else { DVec3::Z },
            omega_rad_myr: omega,
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
        });
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
            "supercontinent broken into {} plates",
            planet.plates.len()
        )));
    log::debug!("initial partition: {} plates", planet.plates.len());
}

/// Lay a sparse network of proto-sutures along the creases of a ridged noise
/// field. `SUTURE` is the only flag that survives a step, and `drift::split_plate`
/// prefers to nucleate rifts on it, so this is how a planet inherits weakness
/// from its accretion history rather than only from the seams where cratons
/// happened to collide. The threshold is a rank, not a value, so the flagged
/// fraction is exactly [`WEAKNESS_FRACTION`] whatever the field's distribution.
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
