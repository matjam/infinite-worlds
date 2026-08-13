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
use crate::topology::{
    bfs_distance, compact_plates, enforce_contiguity, flood_labels_warped, UnionFind, UNASSIGNED,
};
use crate::{
    MeshCache, Scratch, AGE_DENSITY_COEFF, OCEANIC_DENSITY_KG_M3, OCEANIC_DENSITY_MAX_KG_M3,
};

/// Landmass drift during the crustal-formation era, m/yr. Slow: the era is
/// about differentiation, not travel — real motion starts with the hand-off.
const GENESIS_SPEED_M_YR: f64 = 0.02;
/// Plate count the hand-off partition aims for.
const TARGET_PLATES_MIN: usize = 5;
/// Plate count the hand-off partition will not exceed.
const TARGET_PLATES_MAX: usize = 8;
/// Speed given to the ocean plates invented at the hand-off, m/yr.
const SEED_PLATE_SPEED_M_YR: f32 = 0.04;
/// Divergence kick applied to the two halves of a split supercontinent at
/// the drift handoff, m/yr. Gentler than a live rift's separation (0.07):
/// this is a nudge for slab pull to amplify, not an imposed breakup.
const HANDOFF_SPLIT_KICK_M_YR: f32 = 0.03;
/// Distance-metric warp for the hand-off Voronoi, as a fraction of the base
/// cost. Straight great-circle plate boundaries are the tell-tale of a Voronoi
/// partition; a noisy metric makes them meander.
const PARTITION_WARP: f32 = 0.45;
/// Frequency of that warp field on the unit sphere.
const PARTITION_WARP_FREQ: f32 = 3.0;
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
pub(crate) fn step(
    planet: &mut Planet,
    mesh: &Mesh,
    dt_myr: f64,
    ctx: &mut StepCtx,
    cache: &MeshCache,
    genesis: &Genesis,
    scratch: &mut Scratch,
) {
    let _ = scratch;
    if planet.plates.is_empty() {
        seed_supercontinent(planet, mesh, ctx, cache, genesis);
    }
    age_ocean(planet, dt_myr);
}

/// Lay down the global basaltic proto-crust, stamp the supercontinent and any
/// microcontinents, and give each landmass its (slow) plate motion.
fn seed_supercontinent(
    planet: &mut Planet,
    mesh: &Mesh,
    ctx: &mut StepCtx,
    cache: &MeshCache,
    genesis: &Genesis,
) {
    let n = planet.n_cells();

    // Pure per-cell membership map, in parallel; all mutation stays serial so
    // the mass ledger is bit-identical across thread counts.
    let members: Vec<Option<(u16, f32)>> = mesh
        .centers
        .par_iter()
        .map(|d| genesis.membership(*d))
        .collect();

    let mut continental_cells = 0usize;
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
    for (c, member) in members.iter().enumerate() {
        if let Some((mass, f)) = member {
            let thickness = genesis.target_thickness_m(*mass, mesh.centers[c], *f);
            make_continental(planet, c as u32, thickness, cache.area_m2[c], ctx.ledger);
            planet.plate_id[c] = *mass;
            continental_cells += 1;
        }
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
            "supercontinent formed: {:.0}% of the surface, {} landmass{}",
            100.0 * continental_cells as f64 / n as f64,
            genesis.n_masses(),
            if genesis.n_masses() == 1 { "" } else { "es" },
        )));
    log::debug!(
        "genesis: {} landmasses, {continental_cells} continental cells",
        genesis.n_masses()
    );
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

/// Partition every cell into the initial plate set: one plate per landmass
/// plus its surrounding ocean (graph Voronoi), topped up with pure ocean
/// plates so the planet starts Drift with 5-8 plates.
pub(crate) fn partition_into_plates(
    planet: &mut Planet,
    mesh: &Mesh,
    ctx: &mut StepCtx,
    scratch: &mut Scratch,
) {
    let n = planet.n_cells();
    let ncr = planet.plates.len();

    // Landmass -> cluster (welds can only come from later eras; kept for
    // resilience when a re-run enters here with welded plates).
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

    // A welded supercontinent arrives here as ONE cluster, and the ocean-seed
    // filler below then places every remaining plate at maximal distance from
    // it — the drift era reliably started with all boundaries in the ocean
    // and the whole landmass on a single immortal plate. Split the dominant
    // cluster in two along its craton structure (Laurasia/Gondwana): group
    // its member cratons around the farthest pair of craton centroids, so the
    // initial boundary transects the landmass between shields.
    let mut split_pair: Option<(u16, u16)> = None;
    {
        let total_cont: f64 = cluster_mass.iter().sum();
        let big =
            (0..cluster_mass.len()).max_by(|a, b| cluster_mass[*a].total_cmp(&cluster_mass[*b]));
        if let Some(big) = big.filter(|b| total_cont > 0.0 && cluster_mass[*b] > 0.55 * total_cont)
        {
            let mut csum = vec![DVec3::ZERO; ncr];
            for c in 0..n {
                let p = planet.plate_id[c] as usize;
                if p < ncr && planet.crust_type[c] == CrustType::Continental {
                    csum[p] += mesh.centers[c].as_dvec3();
                }
            }
            let members: Vec<usize> = (0..ncr)
                .filter(|k| mass[*k] > 0.0 && cluster_of[*k] == big as u16)
                .collect();
            if members.len() >= 2 {
                let dir = |k: usize| csum[k].normalize_or(DVec3::Z);
                let (mut sa, mut sb, mut best) = (members[0], members[1], f64::INFINITY);
                for (i, &a) in members.iter().enumerate() {
                    for &b in &members[i + 1..] {
                        let d = dir(a).dot(dir(b));
                        if d < best {
                            (sa, sb, best) = (a, b, d);
                        }
                    }
                }
                let new_id = cluster_omega.len() as u16;
                cluster_omega.push(DVec3::ZERO);
                cluster_mass.push(0.0);
                let (da, db) = (dir(sa), dir(sb));
                let mut moved = 0.0f64;
                for &k in &members {
                    if dir(k).dot(db) > dir(k).dot(da) {
                        cluster_of[k] = new_id;
                        cluster_omega[big] -=
                            planet.plates[k].euler_pole * planet.plates[k].omega_rad_myr * mass[k];
                        cluster_omega[new_id as usize] +=
                            planet.plates[k].euler_pole * planet.plates[k].omega_rad_myr * mass[k];
                        moved += mass[k];
                    }
                }
                cluster_mass[big] -= moved;
                *cluster_mass.last_mut().expect("just pushed") = moved;
                if moved > 0.0 {
                    split_pair = Some((big as u16, new_id));
                }
            }
        }
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
        // Keep the split pair pointing at the right clusters through the
        // renumbering; drop it if either half was folded away.
        split_pair = split_pair.and_then(|(a, b)| {
            let remap = |x: u16| {
                let x = x as usize;
                if x == small {
                    None
                } else if x > small {
                    Some((x - 1) as u16)
                } else {
                    Some(x as u16)
                }
            };
            remap(a).zip(remap(b))
        });
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
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
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
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        });
    }

    // The two halves of a split supercontinent start with near-identical
    // motion, and QUIET_MERGE would re-weld the handoff boundary within a
    // dozen Myr — before slab pull has differentiated them. Give the pair
    // the same young-rift immunity as a live rift, plus a gentle divergence
    // kick for the forces to amplify.
    if let Some((a, b)) = split_pair {
        let (ai, bi) = (a as usize, b as usize);
        if ai < plates.len() && bi < plates.len() {
            let ca = cluster_sum[ai].normalize_or(DVec3::Z).as_vec3();
            let cb = cluster_sum[bi].normalize_or(DVec3::Z).as_vec3();
            let kick = |plate: &mut Plate, from: glam::Vec3, toward: glam::Vec3| {
                let w = plate.euler_pole * plate.omega_rad_myr
                    + omega_for_velocity(
                        from,
                        -tangent_toward(from, toward) * HANDOFF_SPLIT_KICK_M_YR,
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
            ctx.progress
                .event(iw_core::ProgressEvent::Milestone(format!(
                    "the supercontinent hands off as two plates ({a} and {b})"
                )));
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
