//! Phase 1: craton seeding, drift, accretion, and the hand-off partition into
//! the initial plate set (DESIGN.md §5 Phase 1).
//!
//! Cratons are continental discs. Their *shape* (radius, lobe wobble) is a pure
//! function of `(seed, craton index)` so it can be recomputed at any time; their
//! *position* is recomputed from the cells they currently own. Nothing about a
//! craton is stored outside `Planet`, which is what lets a fresh process
//! instance continue a checkpoint exactly.

use std::collections::VecDeque;

use glam::{DVec3, Vec3};
use iw_core::planet::cell_flags;
use iw_core::{rng_for, CrustType, Phase, Planet, Plate, StepCtx};
use iw_mesh::{Mesh, EARTH_RADIUS_M};
use rand::Rng;

use crate::crust::{make_continental, make_fresh_oceanic};
use crate::geom::{arc_m, omega_for_velocity, random_unit, rotate_about, tangent_toward};
use crate::topology::{
    bfs_distance, compact_plates, enforce_contiguity, flood_labels, UnionFind, UNASSIGNED,
};
use crate::{
    MeshCache, Scratch, AGE_DENSITY_COEFF, CRATON_CORE_THICKNESS_M, CRATON_EDGE_THICKNESS_M,
    OCEANIC_DENSITY_KG_M3, OCEANIC_DENSITY_MAX_KG_M3,
};

/// Fraction of the planet the craton nuclei are sized to cover in total.
/// Earth's continental crust (shelves included) is a little under 40%; starting
/// nearer 26% leaves room for arc growth and matches the land fractions the
/// golden planets expect once sea level is solved.
const CONTINENTAL_TARGET_FRACTION: f64 = 0.26;
/// Per-craton area spread about the mean, as a multiplier.
const CRATON_AREA_SPREAD: (f64, f64) = (0.65, 1.35);
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

/// Deterministic per-craton shape, derived from `(seed, index)` alone.
struct CratonShape {
    radius_m: f64,
    /// Reference direction fixing the azimuth zero of the lobe pattern.
    ref_axis: Vec3,
    lobe3: f32,
    lobe5: f32,
    phase3: f32,
    phase5: f32,
}

impl CratonShape {
    /// Radius at azimuth `phi` (radians) in the craton's own frame.
    fn radius_at(&self, phi: f32) -> f64 {
        let w = 1.0
            + self.lobe3 * (3.0 * phi + self.phase3).sin()
            + self.lobe5 * (5.0 * phi + self.phase5).sin();
        self.radius_m * w.clamp(0.6, 1.4) as f64
    }
}

/// Craton shapes for a planet. Uses its own RNG stream so the values can be
/// regenerated at any step without disturbing the per-step stream.
fn craton_shapes(seed: u64, count: usize) -> Vec<CratonShape> {
    let mut rng = rng_for(seed, "tectonics/craton-shape", 0);
    (0..count)
        .map(|_| CratonShape {
            radius_m: cap_radius_m(
                CONTINENTAL_TARGET_FRACTION / count.max(1) as f64
                    * rng.random_range(CRATON_AREA_SPREAD.0..CRATON_AREA_SPREAD.1),
            ),
            ref_axis: random_unit(&mut rng),
            lobe3: rng.random_range(0.06f32..0.18),
            lobe5: rng.random_range(0.03f32..0.10),
            phase3: rng.random_range(0.0f32..std::f32::consts::TAU),
            phase5: rng.random_range(0.0f32..std::f32::consts::TAU),
        })
        .collect()
}

/// Great-circle radius of a spherical cap covering `fraction` of the sphere.
fn cap_radius_m(fraction: f64) -> f64 {
    let cos_theta = (1.0 - 2.0 * fraction).clamp(-0.99, 1.0);
    cos_theta.acos() * EARTH_RADIUS_M
}

/// Minimum great-circle separation, in metres, that the craton seeder enforces
/// between any two craton centres for this `(seed, count)`.
///
/// Exposed so tests (and tuning UIs) can check spacing without duplicating the
/// sampling constants.
pub fn craton_min_separation_m(seed: u64, count: u32) -> f64 {
    let shapes = craton_shapes(seed, count as usize);
    let mut min = f64::INFINITY;
    for i in 0..shapes.len() {
        for j in i + 1..shapes.len() {
            min = min.min((shapes[i].radius_m + shapes[j].radius_m) * SEPARATION_FLOOR);
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
    scratch: &mut Scratch,
) {
    let shapes = craton_shapes(planet.config.seed, planet.config.craton_count as usize);
    let centers = if planet.plates.is_empty() {
        seed_planet(planet, ctx, cache, &shapes)
    } else {
        advect(planet, mesh, dt_myr, ctx)
    };

    rasterize_and_apply(planet, mesh, cache, ctx, &shapes, &centers, scratch);
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
        make_fresh_oceanic(planet, c, cache.area_m2[c as usize], ctx.ledger);
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

/// Stamp the craton discs onto the mesh and reconcile with the previous step:
/// leading-edge cells become continental, trailing-edge cells revert to ocean.
fn rasterize_and_apply(
    planet: &mut Planet,
    mesh: &Mesh,
    cache: &MeshCache,
    ctx: &mut StepCtx,
    shapes: &[CratonShape],
    centers: &[Vec3],
    scratch: &mut Scratch,
) {
    let n = planet.n_cells();
    // scratch.u16a: craton claiming each cell (UNASSIGNED = ocean).
    // scratch.f32a: normalized distance from the claiming craton's centre.
    scratch.u16a.fill(UNASSIGNED);
    scratch.f32a.fill(0.0);

    let mut queue: VecDeque<u32> = VecDeque::new();
    for (k, center) in centers.iter().enumerate() {
        if center.length_squared() < 0.5 {
            continue; // extinct craton
        }
        let shape = &shapes[k.min(shapes.len() - 1)];
        // Body frame for the lobe pattern.
        let t = {
            let a = shape.ref_axis - *center * center.dot(shape.ref_axis);
            if a.length_squared() > 1e-10 {
                a.normalize()
            } else {
                crate::geom::any_tangent(*center)
            }
        };
        let b = center.cross(t);
        let cos_bound = (1.4 * shape.radius_m / EARTH_RADIUS_M).cos();

        let start = mesh.cell_at(*center);
        queue.clear();
        queue.push_back(start);
        scratch.flags[start as usize] = true;
        scratch.cells.clear();
        while let Some(c) = queue.pop_front() {
            scratch.cells.push(c);
            let cc = mesh.centers[c as usize];
            if cc.dot(*center) as f64 <= cos_bound {
                continue;
            }
            for &m in mesh.neighbors_of(c) {
                if !scratch.flags[m as usize] {
                    scratch.flags[m as usize] = true;
                    queue.push_back(m);
                }
            }
        }
        for &c in &scratch.cells {
            let cc = mesh.centers[c as usize];
            let d = cc - *center * center.dot(cc);
            let phi = if d.length_squared() > 1e-12 {
                d.dot(b).atan2(d.dot(t))
            } else {
                0.0
            };
            let r_eff = shape.radius_at(phi);
            let dist = arc_m(*center, cc);
            if dist <= r_eff && scratch.u16a[c as usize] == UNASSIGNED {
                scratch.u16a[c as usize] = k as u16;
                scratch.f32a[c as usize] = (dist / r_eff) as f32;
            }
        }
        // The centre cell always belongs to its craton.
        if scratch.u16a[start as usize] == UNASSIGNED {
            scratch.u16a[start as usize] = k as u16;
            scratch.f32a[start as usize] = 0.0;
        }
        for &c in &scratch.cells {
            scratch.flags[c as usize] = false;
        }
    }

    for c in 0..n as u32 {
        let claim = scratch.u16a[c as usize];
        let was_continental = planet.crust_type[c as usize] == CrustType::Continental;
        let area = cache.area_m2[c as usize];
        if claim != UNASSIGNED {
            planet.plate_id[c as usize] = claim;
            if !was_continental {
                let f = scratch.f32a[c as usize];
                let thickness = CRATON_CORE_THICKNESS_M
                    - (CRATON_CORE_THICKNESS_M - CRATON_EDGE_THICKNESS_M) * f * f;
                make_continental(planet, c, thickness, area, ctx.ledger);
            }
        } else {
            planet.plate_id[c as usize] = UNASSIGNED;
            if was_continental {
                make_fresh_oceanic(planet, c, area, ctx.ledger);
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
    flood_labels(mesh, &mut scratch.u16a);
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

    ctx.progress
        .event(iw_core::ProgressEvent::Milestone(format!(
            "supercontinent broken into {} plates",
            planet.plates.len()
        )));
    log::debug!("initial partition: {} plates", planet.plates.len());
}
