//! Drift v2: rigid-cap advection of crustal fields (docs/drift-v2.md).
//!
//! Each plate accumulates its rotation; once the implied surface displacement
//! reaches one cell pitch the plate's entire field set — ownership, crust
//! type/thickness/density/age, sediment, SUTURE flags, and the stratigraphic
//! columns — is remapped through the rotation. Cells nobody claims are
//! seafloor spreading (fresh age-0 ridge crust); cells claimed twice are
//! convergence, resolved by subduction polarity or continental collision.
//! This is what lets continents genuinely travel, rifts open into oceans,
//! hotspot chains form, and old boundary scars heal.
//!
//! Ledger discipline: a moved column charges nothing (transport); a column
//! lost in an overlap is `subducted_m3`; ridge crust filling a gap is
//! `created_m3`. Ledger sums are accumulated serially in cell order so the
//! totals are bit-identical across thread counts.

use glam::DQuat;
use iw_core::planet::cell_flags;
use iw_core::{CrustType, MassLedger, Planet, RockType, StrataColumns};
use iw_mesh::Mesh;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::crust::{OCEAN_BASALT_M, OCEAN_GABBRO_M};
use crate::{MeshCache, MAX_CRUST_THICKNESS_M, OCEANIC_DENSITY_KG_M3};

/// A plate remaps when its pending displacement reaches this many cell pitches.
const REMAP_TRIGGER_PITCHES: f64 = 1.0;

/// Fraction of the losing plate's crust thickness folded into the winner in a
/// continent–continent overlap (crustal shortening; mirrors drift.rs's
/// COLLIDE_SHORTENING for the continuous phase of the same process).
const COLLIDE_FOLD: f32 = 0.32;

/// One claimant on a destination cell: (plate, source cell).
type Claim = (u16, u32);

enum Decision {
    /// Pull every advected field from `src` (owned by `plate`).
    Keep { plate: u16, src: u32 },
    /// Divergent gap: fresh ridge crust, owned by `plate`.
    Fresh { plate: u16 },
    /// Convergent overlap: winner pulled from `src`, losers destroyed.
    Converge {
        plate: u16,
        src: u32,
        /// Thickness added to the winner (continental fold), metres.
        fold_m: f32,
        /// Sources whose columns subduct here.
        losers: SmallVec<[u32; 2]>,
        continental_collision: bool,
    },
}

/// Advance plate rotations and remap fields when due. Returns true when a
/// remap ran (topology may have changed).
pub(crate) fn accumulate_and_remap(
    planet: &mut Planet,
    mesh: &Mesh,
    cache: &MeshCache,
    dt_myr: f64,
    ledger: &mut MassLedger,
) -> bool {
    let np = planet.plates.len();
    if np == 0 {
        return false;
    }
    for p in &mut planet.plates {
        p.accumulate(dt_myr);
    }
    let trigger = cache.pitch_m * REMAP_TRIGGER_PITCHES;
    let due: Vec<bool> = planet
        .plates
        .iter()
        .map(|p| p.pending_displacement_m() >= trigger)
        .collect();
    if !due.iter().any(|d| *d) {
        return false;
    }

    let n = planet.n_cells();
    let fwd: Vec<DQuat> = planet
        .plates
        .iter()
        .enumerate()
        .map(|(i, p)| if due[i] { p.accum } else { DQuat::IDENTITY })
        .collect();
    let inv: Vec<DQuat> = fwd.iter().map(|q| q.conjugate()).collect();

    // -- Pull pass. Push-based discovery assumed near-uniform cell sizes (one
    // pushed claim per destination): on the terrain-sized Voronoi mesh a big
    // cell's image covers hundreds of small cells, so instead every
    // DESTINATION cell asks "which plates could reach me?" — the plates
    // present in any chunk whose cone lies within the maximum pending
    // displacement — and pull-tests each through its inverse rotation. Exact
    // for any displacement and any cell-size contrast.
    let old_plate = &planet.plate_id;
    let chunk_plates: Vec<SmallVec<[u16; 8]>> = mesh
        .chunks
        .iter()
        .map(|ch| {
            let mut set: SmallVec<[u16; 8]> = SmallVec::new();
            for &c in &ch.cells {
                let p = old_plate[c as usize];
                if !set.contains(&p) {
                    set.push(p);
                }
            }
            set
        })
        .collect();
    let max_disp_rad = planet
        .plates
        .iter()
        .enumerate()
        .filter(|(i, _)| due[*i])
        .map(|(_, p)| 2.0 * p.accum.w.abs().clamp(0.0, 1.0).acos())
        .fold(0.0f64, f64::max) as f32;
    // Reachability per chunk: cos(chunk_radius + max displacement).
    let chunk_cos_reach: Vec<f32> = mesh
        .chunks
        .iter()
        .map(|ch| {
            (ch.cos_radius.clamp(-1.0, 1.0).acos() + max_disp_rad)
                .min(std::f32::consts::PI)
                .cos()
        })
        .collect();

    let mut claims_per_cell: Vec<SmallVec<[Claim; 2]>> = (0..n)
        .into_par_iter()
        .map(|d| {
            let dir = mesh.centers[d];
            let mut cands: SmallVec<[u16; 6]> = SmallVec::new();
            for (ci, ch) in mesh.chunks.iter().enumerate() {
                if ch.center.dot(dir) >= chunk_cos_reach[ci] {
                    for &p in &chunk_plates[ci] {
                        if (p as usize) < np && !cands.contains(&p) {
                            cands.push(p);
                        }
                    }
                }
            }
            let ddir = dir.as_dvec3();
            let mut claims: SmallVec<[Claim; 2]> = SmallVec::new();
            for &p in &cands {
                let src = mesh.cell_at((inv[p as usize] * ddir).as_vec3());
                if old_plate[src as usize] == p {
                    claims.push((p, src));
                }
            }
            claims
        })
        .collect();

    // -- Push supplement: pull alone loses a zero-claim boundary band to
    // quantization jitter (a cell just outside the back-rotated outline claims
    // nothing and turns to ridge crust). Pushing every source cell's content
    // to its forward image guarantees it lands SOMEWHERE, restoring area
    // conservation; duplicates against the pull claims are reconciled by the
    // per-plate best-alignment dedupe in `resolve_overlap`.
    let dst_of: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|c| {
            let p = old_plate[c] as usize;
            if p >= np || !due[p] {
                c as u32
            } else {
                mesh.cell_at((fwd[p] * mesh.centers[c].as_dvec3()).as_vec3())
            }
        })
        .collect();
    for c in 0..n {
        let d = dst_of[c] as usize;
        let claim = (old_plate[c], c as u32);
        if !claims_per_cell[d].contains(&claim) {
            claims_per_cell[d].push(claim);
        }
    }

    // -- Resolve every destination cell.
    let decisions: Vec<Decision> = (0..n as u32)
        .into_par_iter()
        .map(|d| resolve(d, planet, mesh, &claims_per_cell, &inv))
        .collect();

    // -- Materialize the new field set.
    let mut new_plate = vec![0u16; n];
    let mut new_type = vec![CrustType::Oceanic; n];
    let mut new_thick = vec![0f32; n];
    let mut new_density = vec![0f32; n];
    let mut new_age = vec![0f32; n];
    let mut new_sediment = vec![0f32; n];
    let mut new_elev = vec![0f32; n];
    let mut new_flags = vec![0u8; n];
    let mut new_columns = StrataColumns::new(n);
    let mut subducted_m3 = 0f64;
    let mut created_m3 = 0f64;

    for d in 0..n {
        match &decisions[d] {
            Decision::Keep { plate, src } | Decision::Converge { plate, src, .. } => {
                let s = *src as usize;
                new_plate[d] = *plate;
                new_type[d] = planet.crust_type[s];
                new_thick[d] = planet.crust_thickness_m[s];
                new_density[d] = planet.crust_density_kg_m3[s];
                new_age[d] = planet.crust_age_myr[s];
                new_sediment[d] = planet.sediment_m[s];
                new_elev[d] = planet.elevation_m[s];
                new_flags[d] = planet.tectonic_flags[s] & cell_flags::SUTURE;
                new_columns.copy_col_from(d as u32, &planet.columns, *src);
                if let Decision::Converge {
                    fold_m,
                    losers,
                    continental_collision,
                    ..
                } = &decisions[d]
                {
                    new_thick[d] = (new_thick[d] + fold_m).min(MAX_CRUST_THICKNESS_M);
                    for &l in losers {
                        let ls = l as usize;
                        if planet.crust_type[ls] == CrustType::Continental {
                            // Underthrust: the record survives beneath the
                            // winner. Mass moved, not destroyed — no charge.
                            new_columns.prepend_col_from(d as u32, &planet.columns, l);
                        } else {
                            subducted_m3 += planet.columns.total_thickness_m(l) as f64
                                * cache.area_m2[ls]
                                + planet.sediment_m[ls] as f64 * cache.area_m2[ls];
                        }
                    }
                    if *continental_collision {
                        new_flags[d] |= cell_flags::COLLISION | cell_flags::SUTURE;
                    } else {
                        new_flags[d] |= cell_flags::SUBDUCTING;
                    }
                }
            }
            Decision::Fresh { plate } => {
                let thick = cache.ocean_thickness_m[d];
                new_plate[d] = *plate;
                new_type[d] = CrustType::Oceanic;
                new_thick[d] = thick;
                new_density[d] = OCEANIC_DENSITY_KG_M3;
                new_age[d] = 0.0;
                new_sediment[d] = 0.0;
                // Ridge-fresh elevation placeholder; isostasy recomputes at the
                // next geology pass in the same step.
                new_elev[d] = planet.elevation_m[d];
                new_flags[d] = cell_flags::RIFT;
                let t = planet.time_myr;
                new_columns.deposit(d as u32, RockType::Gabbro, OCEAN_GABBRO_M, t);
                new_columns.deposit(d as u32, RockType::Basalt, OCEAN_BASALT_M, t);
                created_m3 += (OCEAN_GABBRO_M + OCEAN_BASALT_M) as f64 * cache.area_m2[d];
            }
        }
    }

    ledger.subducted_m3 += subducted_m3;
    ledger.created_m3 += created_m3;

    planet.plate_id = new_plate;
    planet.crust_type = new_type;
    planet.crust_thickness_m = new_thick;
    planet.crust_density_kg_m3 = new_density;
    planet.crust_age_myr = new_age;
    planet.sediment_m = new_sediment;
    planet.elevation_m = new_elev;
    planet.tectonic_flags = new_flags;
    planet.columns = new_columns;
    for (i, p) in planet.plates.iter_mut().enumerate() {
        if due[i] {
            p.accum = DQuat::IDENTITY;
        }
    }
    true
}

/// Decide what lands on destination cell `d`.
fn resolve(
    d: u32,
    planet: &Planet,
    mesh: &Mesh,
    claims: &[SmallVec<[Claim; 2]>],
    inv: &[DQuat],
) -> Decision {
    let list = &claims[d as usize];
    match list.len() {
        1 => Decision::Keep {
            plate: list[0].0,
            src: list[0].1,
        },
        0 => Decision::Fresh {
            // A genuine divergence gap: no plate's crust reaches this cell.
            // The trailing plate (the cell's previous owner) keeps the new
            // ridge crust — deterministic and physically sensible.
            plate: planet.plate_id[d as usize],
        },
        _ => resolve_overlap(planet, mesh, d, inv, list),
    }
}

/// Two or more claims land on the same cell. Same-plate duplicates are
/// discretization collisions (a rigid rotation on a discrete grid is only
/// near-bijective): keep the best-aligned source and drop the rest without
/// ledger charges. Distinct plates are genuine convergence.
fn resolve_overlap(
    planet: &Planet,
    mesh: &Mesh,
    d: u32,
    inv: &[DQuat],
    list: &[Claim],
) -> Decision {
    // Per-plate dedupe: keep the source whose back-rotated destination lies
    // closest to it (max dot), tie on lower source id.
    let dir = mesh.centers[d as usize].as_dvec3();
    let mut per_plate: SmallVec<[Claim; 2]> = SmallVec::new();
    for &(p, s) in list {
        let dot = |src: u32| (inv[p as usize] * dir).dot(mesh.centers[src as usize].as_dvec3());
        match per_plate.iter_mut().find(|(q, _)| *q == p) {
            None => per_plate.push((p, s)),
            Some((_, best)) => {
                let (db, ds) = (dot(*best), dot(s));
                if ds > db || (ds == db && s < *best) {
                    *best = s;
                }
            }
        }
    }
    if per_plate.len() == 1 {
        return Decision::Keep {
            plate: per_plate[0].0,
            src: per_plate[0].1,
        };
    }

    // Winner: continental beats oceanic; between oceanic the older (denser)
    // side subducts so the younger survives; between continental the thicker
    // crust wins. Ties break on lower plate id.
    let winner = *per_plate
        .iter()
        .max_by(|a, b| {
            let key = |&(p, s): &Claim| {
                let s = s as usize;
                let continental = planet.crust_type[s] == CrustType::Continental;
                let k: f32 = if continental {
                    planet.crust_thickness_m[s]
                } else {
                    -planet.crust_age_myr[s]
                };
                (continental, k, std::cmp::Reverse(p))
            };
            let (ac, ak, ap) = key(a);
            let (bc, bk, bp) = key(b);
            ac.cmp(&bc).then(ak.total_cmp(&bk)).then(ap.cmp(&bp))
        })
        .expect("non-empty");
    let mut fold_m = 0.0f32;
    let mut losers: SmallVec<[u32; 2]> = SmallVec::new();
    let mut continental_collision = false;
    for &(p, s) in per_plate.iter() {
        if (p, s) == winner {
            continue;
        }
        if planet.crust_type[s as usize] == CrustType::Continental
            && planet.crust_type[winner.1 as usize] == CrustType::Continental
        {
            continental_collision = true;
            fold_m += planet.crust_thickness_m[s as usize] * COLLIDE_FOLD;
        }
        losers.push(s);
    }
    Decision::Converge {
        plate: winner.0,
        src: winner.1,
        fold_m,
        losers,
        continental_collision,
    }
}
