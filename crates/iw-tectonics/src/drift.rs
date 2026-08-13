//! The main tectonic engine: force balance, boundary resolution, rifting,
//! welding, hotspots and oceanic aging (DESIGN.md §5 Phase 2-4).

use glam::{DVec3, Vec3};
use iw_core::planet::cell_flags;
use iw_core::{CrustType, Hotspot, Phase, Planet, Plate, ProgressEvent, RockType, StepCtx};
use iw_mesh::{Mesh, EARTH_RADIUS_M};
use rand::Rng;
use rustc_hash::FxHashMap;

use crate::boundary::{self, Kind};
use crate::crust::{deposit_new, destroy_column, make_fresh_oceanic};
use crate::geom::{omega_for_velocity, random_unit, reproject, tangent_toward};
use crate::phase1;
use crate::topology::{absorb_tiny_plates, compact_plates, enforce_contiguity};
use crate::{
    MeshCache, Scratch, MAX_CRUST_THICKNESS_M, MAX_PLATES, MIN_RIFTABLE_CELLS, OCEANIC_THICKNESS_M,
    RIFT_BREAKUP_THICKNESS_M, TRENCH_THICKNESS_M,
};

// --- force balance ---

/// Slab pull per metre of subducting boundary (reference units).
const SLAB_PULL: f64 = 1.0;
/// Ridge push per metre of spreading boundary, relative to slab pull.
const RIDGE_PUSH: f64 = 0.22;
/// Viscous braking per metre of collisional boundary, per (m/yr) of closure.
const COLLISION_VISC: f64 = 22.0;
/// Converts driving torque density into angular velocity, rad*m/Myr.
/// Calibrated so a plate with a normal share of subducting boundary settles
/// around 4-6 cm/yr at `tectonic_vigor == 1`.
const TORQUE_GAIN: f64 = 1.1e5;
/// Smallest plate area, as a fraction of the planet, that the drag term will
/// use. Without it a microplate's perimeter-to-area ratio drives it straight
/// into the speed cap; real microplates are dragged along by their neighbours.
const MIN_DRAG_AREA_FRAC: f64 = 0.10;
/// Extra basal drag from thick continental keels, as a fraction.
const CONTINENTAL_DRAG: f64 = 1.4;
/// Relaxation time of a plate's angular velocity toward the force balance, Myr.
const MOTION_TAU_MYR: f64 = 12.0;
/// Random mantle-torque walk, rad/Myr per Myr. Keeps plates whose boundaries
/// are all transform from stalling completely.
const MANTLE_JITTER_RAD_MYR: f64 = 0.0016;
/// Absolute plate speed ceiling, m/yr.
const MAX_PLATE_SPEED_M_YR: f64 = 0.15;
/// Floor for the per-step stability cap, m/yr, so the 2-10 cm/yr target range
/// stays reachable at high subdivision levels where 0.7 cell pitch is small.
const MIN_SPEED_CAP_M_YR: f64 = 0.12;

// --- boundary effects ---

/// Reference closure rate that the volcanic rates below are quoted at, m/yr.
const REF_CONVERGENCE_M_YR: f32 = 0.05;
/// Arc lava deposited on the nearest arc cell, m/Myr at the reference rate.
const ARC_DEPOSIT_M_MYR: f32 = 45.0;
/// Arc crustal thickening, m/Myr at the reference rate.
const ARC_THICKEN_M_MYR: f32 = 120.0;
/// Ceiling on arc thickening for continental / oceanic overriding plates, m.
const ARC_MAX_CONTINENTAL_M: f32 = 55_000.0;
const ARC_MAX_OCEANIC_M: f32 = 26_000.0;
/// Fraction of the crust-thickness deficit a trench cell closes per unit of
/// (closure * dt / pitch).
const TRENCH_FLEX_RATE: f32 = 1.5;
/// Fraction of shortening strain converted into crustal thickening.
///
/// Calibration: thickening is proportional to the crust already there, so
/// thinning the craton seeds (45 -> 40 km) compounded through every collision
/// and cost the tallest orogens ~2 km. 0.32 puts collisional belts back at
/// 60-70 km of root, i.e. 4-6 km of relief, without letting them pin at the
/// 70 km ceiling everywhere.
const COLLIDE_SHORTENING: f32 = 0.32;
/// Fraction of extensional strain converted into crustal thinning.
const RIFT_STRETCH: f32 = 0.07;
/// Basalt veneer left on a freshly consumed trench cell, m.
const TRENCH_VENEER_M: f32 = 300.0;
/// Oceanic crust relaxes back toward its reference thickness with this time
/// constant once it is no longer flexing into a trench, Myr.
const OCEAN_RELAX_TAU_MYR: f32 = 25.0;
/// Orogenic collapse: thickened continental crust creeps back toward this
/// thickness with the time constant below.
///
/// Calibration: 38 km floats at +1.35 km, so every continent that had time to
/// relax ended up a kilometre-and-a-half plateau and nothing was lowland. 34 km
/// floats at +545 m and 36 km at +982 m — a mature continental platform, near
/// Earth's 840 m mean land elevation once orogens and shields are averaged in.
const CONTINENTAL_REST_M: f32 = 36_000.0;
const CONTINENTAL_RELAX_TAU_MYR: f32 = 400.0;

// --- rifting / welding ---

/// Plate area fraction above which a mostly continental plate is unstable.
const SUPERPLATE_AREA_FRAC: f64 = 0.25;
/// Continental area fraction that makes a big plate a rift candidate.
const SUPERPLATE_CONT_FRAC: f64 = 0.22;
/// Rift nucleation probability per Myr for an unstable superplate.
const RIFT_PROB_SUPER_PER_MYR: f64 = 0.025;
/// Background rift nucleation probability per Myr for any other plate.
const RIFT_PROB_BASE_PER_MYR: f64 = 0.0002;
/// Opening speed handed to the two halves of a fresh rift, m/yr.
const RIFT_SEPARATION_M_YR: f32 = 0.03;
/// Smallest fragment a rift may produce, in cells.
const MIN_RIFT_FRAGMENT: usize = 6;
/// Collisional boundary length, in cell pitches, needed before plates may weld.
const WELD_MIN_PITCHES: f64 = 3.0;
/// Closure rate below which a collision is considered locked, m/yr.
const WELD_SPEED_M_YR: f64 = 0.010;
/// A long boundary with almost no relative motion is not a boundary: the two
/// plates are merged. This is what keeps the mosaic from fragmenting forever.
const QUIET_MERGE_PITCHES: f64 = 8.0;
const QUIET_MERGE_SPEED_M_YR: f64 = 0.004;

/// Basalt erupted over a plume, m/Myr at strength 1.
const HOTSPOT_DEPOSIT_M_MYR: f32 = 40.0;
const HOTSPOT_MAX_OCEANIC_M: f32 = 25_000.0;
const HOTSPOT_MAX_CONTINENTAL_M: f32 = 50_000.0;

/// A cell reassignment queued by the classification pass and resolved against
/// `ctx.rng` afterwards. See the crate docs for why this is stateless.
enum Transfer {
    /// Renew the older of two cells straddling a spreading centre.
    Spread { a: u32, b: u32, rate_m_yr: f32 },
    /// Consume `s` and hand the trench floor to the overriding plate.
    Subduct {
        s: u32,
        ovr_plate: u16,
        rate_m_yr: f32,
    },
}

/// Length-weighted kinematics of the whole boundary between one plate pair.
#[derive(Default, Clone, Copy)]
struct PairAcc {
    /// Total boundary length, metres.
    len_m: f64,
    /// Integral of |relative speed| along the boundary, m/yr * m.
    rel_len: f64,
    /// Length of the continent-continent part of the boundary, metres.
    coll_len_m: f64,
    /// Integral of the closure rate over that part, m/yr * m.
    coll_conv_len: f64,
}

/// One Drift / Refinement / RecentPast step.
pub(crate) fn step(
    planet: &mut Planet,
    mesh: &Mesh,
    dt_myr: f64,
    ctx: &mut StepCtx,
    cache: &MeshCache,
    scratch: &mut Scratch,
) {
    if phase1::needs_partition(planet) {
        phase1::partition_into_plates(planet, mesh, ctx, scratch);
    }
    ensure_hotspots(planet, ctx);
    let np = planet.plates.len();
    if np == 0 {
        return;
    }
    let n = planet.n_cells();
    let vigor = planet.config.tectonic_vigor as f64;
    let frozen = planet.phase == Phase::RecentPast;
    let pitch = cache.pitch_m;

    let edges = boundary::build_edges(planet, mesh);

    let mut area = vec![0f64; np];
    let mut cont_area = vec![0f64; np];
    let mut cell_count = vec![0u32; np];
    for c in 0..n {
        let p = planet.plate_id[c] as usize;
        if p >= np {
            continue;
        }
        area[p] += cache.area_m2[c];
        cell_count[p] += 1;
        if planet.crust_type[c] == CrustType::Continental {
            cont_area[p] += cache.area_m2[c];
        }
    }

    let mut torque = vec![DVec3::ZERO; np];
    scratch.f32a.fill(0.0); // arc lava, metres
    scratch.f32b.fill(0.0); // crustal thickening, metres
    scratch.flags.fill(false); // cells already reassigned this step
    let mut transfers: Vec<Transfer> = Vec::new();
    let mut pairs: FxHashMap<(u16, u16), PairAcc> = FxHashMap::default();

    for e in &edges {
        let (pa, pb) = (e.pa as usize, e.pb as usize);
        let key = if e.pa < e.pb {
            (e.pa, e.pb)
        } else {
            (e.pb, e.pa)
        };
        {
            let acc = pairs.entry(key).or_default();
            acc.len_m += e.len_m as f64;
            acc.rel_len += e.len_m as f64 * e.rel.length() as f64;
        }
        match e.kind {
            Kind::Transform => {
                planet.tectonic_flags[e.a as usize] |= cell_flags::TRANSFORM;
                planet.tectonic_flags[e.b as usize] |= cell_flags::TRANSFORM;
            }
            Kind::Divergent => {
                let open = -e.conv_m_yr;
                planet.tectonic_flags[e.a as usize] |= cell_flags::RIFT;
                planet.tectonic_flags[e.b as usize] |= cell_flags::RIFT;
                let f = RIDGE_PUSH * e.len_m as f64;
                torque[pa] += e.mid.as_dvec3().cross((-e.n).as_dvec3() * f);
                torque[pb] += e.mid.as_dvec3().cross(e.n.as_dvec3() * f);

                let strain = (open as f64 * dt_myr * 1.0e6 / pitch) as f32;
                for &c in &[e.a, e.b] {
                    if planet.crust_type[c as usize] == CrustType::Continental {
                        let th = planet.crust_thickness_m[c as usize];
                        planet.crust_thickness_m[c as usize] =
                            (th - th * strain * RIFT_STRETCH).max(1_000.0);
                    }
                }
                transfers.push(Transfer::Spread {
                    a: e.a,
                    b: e.b,
                    rate_m_yr: open,
                });
            }
            Kind::Convergent => {
                let conv = e.conv_m_yr;
                let ta = planet.crust_type[e.a as usize];
                let tb = planet.crust_type[e.b as usize];
                if ta == CrustType::Continental && tb == CrustType::Continental {
                    planet.tectonic_flags[e.a as usize] |= cell_flags::COLLISION;
                    planet.tectonic_flags[e.b as usize] |= cell_flags::COLLISION;
                    let strain = (conv as f64 * dt_myr * 1.0e6 / pitch) as f32;
                    for &c in &[e.a, e.b] {
                        let th = planet.crust_thickness_m[c as usize];
                        let room = (1.0 - th / MAX_CRUST_THICKNESS_M).max(0.0);
                        scratch.f32b[c as usize] += th * strain * COLLIDE_SHORTENING * room;
                    }
                    let f = COLLISION_VISC * e.len_m as f64;
                    torque[pa] += e.mid.as_dvec3().cross(e.rel.as_dvec3() * f);
                    torque[pb] += e.mid.as_dvec3().cross(-e.rel.as_dvec3() * f);
                    let acc = pairs.entry(key).or_default();
                    acc.coll_len_m += e.len_m as f64;
                    acc.coll_conv_len += e.len_m as f64 * conv.max(0.0) as f64;
                } else {
                    // Polarity: continental crust never subducts; between two
                    // oceanic plates the older (colder, denser) side goes down.
                    let a_down = match (ta, tb) {
                        (CrustType::Oceanic, CrustType::Continental) => true,
                        (CrustType::Continental, CrustType::Oceanic) => false,
                        _ => {
                            planet.crust_age_myr[e.a as usize] >= planet.crust_age_myr[e.b as usize]
                        }
                    };
                    let (s, o, sub_plate, ovr_plate, dir) = if a_down {
                        (e.a, e.b, e.pa, e.pb, e.n)
                    } else {
                        (e.b, e.a, e.pb, e.pa, -e.n)
                    };
                    planet.tectonic_flags[s as usize] |= cell_flags::SUBDUCTING;

                    // Trench: the down-going plate flexes and thins.
                    let w = ((conv as f64 * dt_myr * 1.0e6 / pitch) as f32 * TRENCH_FLEX_RATE)
                        .clamp(0.0, 1.0);
                    let th = planet.crust_thickness_m[s as usize];
                    planet.crust_thickness_m[s as usize] = th + (TRENCH_THICKNESS_M - th) * w;

                    // Slab pull acts on the subducting plate, toward the trench.
                    let age = planet.crust_age_myr[s as usize].max(0.0);
                    let age_f = (age / 80.0).sqrt().clamp(0.2, 1.5) as f64;
                    let f = SLAB_PULL * e.len_m as f64 * age_f;
                    torque[sub_plate as usize] += e.mid.as_dvec3().cross(dir.as_dvec3() * f);

                    arc_volcanism(
                        planet,
                        mesh,
                        o,
                        dir,
                        ovr_plate,
                        conv,
                        dt_myr,
                        vigor as f32,
                        scratch,
                    );

                    transfers.push(Transfer::Subduct {
                        s,
                        ovr_plate,
                        rate_m_yr: conv,
                    });
                }
            }
        }
    }

    // Discrete reassignment: probability = displacement / cell pitch.
    let mut topology_dirty = false;
    for t in &transfers {
        let rate = match t {
            Transfer::Spread { rate_m_yr, .. } | Transfer::Subduct { rate_m_yr, .. } => *rate_m_yr,
        };
        let p = (rate as f64 * dt_myr * 1.0e6 / pitch).clamp(0.0, 1.0);
        let draw: f64 = ctx.rng.random();
        if draw >= p {
            continue;
        }
        match *t {
            Transfer::Spread { a, b, .. } => {
                if planet.crust_type[a as usize] != CrustType::Oceanic
                    || planet.crust_type[b as usize] != CrustType::Oceanic
                {
                    continue;
                }
                let renew = if planet.crust_age_myr[a as usize] >= planet.crust_age_myr[b as usize]
                {
                    a
                } else {
                    b
                };
                if scratch.flags[renew as usize] {
                    continue;
                }
                scratch.flags[renew as usize] = true;
                make_fresh_oceanic(planet, renew, cache.area_m2[renew as usize], ctx.ledger);
                planet.tectonic_flags[renew as usize] |= cell_flags::RIFT;
            }
            Transfer::Subduct { s, ovr_plate, .. } => {
                if scratch.flags[s as usize] {
                    continue;
                }
                scratch.flags[s as usize] = true;
                let area = cache.area_m2[s as usize];
                destroy_column(planet, s, area, ctx.ledger);
                planet.plate_id[s as usize] = ovr_plate;
                planet.crust_type[s as usize] = CrustType::Oceanic;
                planet.crust_thickness_m[s as usize] = TRENCH_THICKNESS_M;
                planet.tectonic_flags[s as usize] |= cell_flags::SUBDUCTING;
                deposit_new(
                    planet,
                    s,
                    RockType::Basalt,
                    TRENCH_VENEER_M,
                    area,
                    ctx.ledger,
                );
                topology_dirty = true;
            }
        }
    }

    // Apply the accumulated continuous effects.
    for c in 0..n as u32 {
        let lava = scratch.f32a[c as usize];
        if lava > 0.0 {
            // Arc products: mostly andesite lava with a pyroclastic cap.
            deposit_new(
                planet,
                c,
                RockType::Andesite,
                lava * 0.7,
                cache.area_m2[c as usize],
                ctx.ledger,
            );
            deposit_new(
                planet,
                c,
                RockType::Tuff,
                lava * 0.3,
                cache.area_m2[c as usize],
                ctx.ledger,
            );
        }
        let grow = scratch.f32b[c as usize];
        if grow > 0.0 {
            let cap = if planet.crust_type[c as usize] == CrustType::Continental {
                MAX_CRUST_THICKNESS_M
            } else {
                ARC_MAX_OCEANIC_M
            };
            planet.crust_thickness_m[c as usize] =
                (planet.crust_thickness_m[c as usize] + grow).min(cap);
        }
    }

    relax_thickness(planet, dt_myr);
    breakup_stretched_crust(planet, cache, ctx);
    phase1::age_ocean(planet, dt_myr);
    hotspots(planet, mesh, cache, dt_myr, ctx);

    if !frozen {
        update_motion(
            planet,
            &torque,
            &area,
            &cont_area,
            dt_myr,
            vigor,
            pitch,
            cache.total_area_m2,
            ctx,
        );
        if rift_step(
            planet,
            mesh,
            cache,
            ctx,
            scratch,
            &area,
            &cont_area,
            &cell_count,
            dt_myr,
        ) {
            topology_dirty = true;
        }
        if weld_step(planet, &edges, &pairs, &cell_count, pitch, ctx) {
            topology_dirty = true;
        }
    }

    if topology_dirty {
        // Calve only substantial fragments; slivers rejoin a neighbour.
        let calve_min = (n / 150).max(24);
        enforce_contiguity(planet, mesh, &mut scratch.u32a, calve_min, MAX_PLATES);
        absorb_tiny_plates(planet, mesh, (n / 250).max(12));
        compact_plates(planet);
    }
    debug_assert!(planet
        .plate_id
        .iter()
        .all(|p| (*p as usize) < planet.plates.len()));
}

/// Walk 2-4 cells into the overriding plate along the boundary normal and lay
/// down an arc there.
#[allow(clippy::too_many_arguments)]
fn arc_volcanism(
    planet: &mut Planet,
    mesh: &Mesh,
    start: u32,
    dir: Vec3,
    ovr_plate: u16,
    conv_m_yr: f32,
    dt_myr: f64,
    vigor: f32,
    scratch: &mut Scratch,
) {
    const WEIGHTS: [f32; 3] = [1.0, 0.8, 0.5];
    let rate = (conv_m_yr / REF_CONVERGENCE_M_YR) * dt_myr as f32 * vigor;
    if rate <= 0.0 {
        return;
    }
    let mut cur = start;
    let mut heading = reproject(dir, mesh.centers[start as usize]);
    for hop in 1..=4usize {
        let cc = mesh.centers[cur as usize];
        let mut best = cur;
        let mut best_dot = f32::NEG_INFINITY;
        for &m in mesh.neighbors_of(cur) {
            let step = (mesh.centers[m as usize] - cc).normalize_or_zero();
            let d = step.dot(heading);
            if d > best_dot {
                best_dot = d;
                best = m;
            }
        }
        if best == cur || planet.plate_id[best as usize] != ovr_plate {
            return;
        }
        cur = best;
        heading = reproject(heading, mesh.centers[cur as usize]);
        if hop >= 2 {
            let w = WEIGHTS[hop - 2];
            planet.tectonic_flags[cur as usize] |= cell_flags::ARC;
            scratch.f32a[cur as usize] += ARC_DEPOSIT_M_MYR * w * rate;
            let cap = if planet.crust_type[cur as usize] == CrustType::Continental {
                ARC_MAX_CONTINENTAL_M
            } else {
                ARC_MAX_OCEANIC_M
            };
            let head = (cap - planet.crust_thickness_m[cur as usize]).max(0.0);
            scratch.f32b[cur as usize] += (ARC_THICKEN_M_MYR * w * rate).min(head);
        }
    }
}

/// Flexural rebound of ocean floor and slow orogenic collapse of thick crust.
fn relax_thickness(planet: &mut Planet, dt_myr: f64) {
    let dt = dt_myr as f32;
    for c in 0..planet.n_cells() {
        let flexing = planet.tectonic_flags[c] & cell_flags::SUBDUCTING != 0;
        let th = planet.crust_thickness_m[c];
        match planet.crust_type[c] {
            CrustType::Oceanic if !flexing => {
                let w = (dt / OCEAN_RELAX_TAU_MYR).clamp(0.0, 1.0);
                planet.crust_thickness_m[c] = th + (OCEANIC_THICKNESS_M - th) * w;
            }
            CrustType::Continental if th > CONTINENTAL_REST_M => {
                let w = (dt / CONTINENTAL_RELAX_TAU_MYR).clamp(0.0, 1.0);
                planet.crust_thickness_m[c] = th + (CONTINENTAL_REST_M - th) * w;
            }
            _ => {}
        }
    }
}

/// Continental crust stretched past its breaking point becomes ocean floor:
/// this is how a rift finishes turning into a spreading centre.
fn breakup_stretched_crust(planet: &mut Planet, cache: &MeshCache, ctx: &mut StepCtx) {
    for c in 0..planet.n_cells() as u32 {
        if planet.crust_type[c as usize] == CrustType::Continental
            && planet.crust_thickness_m[c as usize] < RIFT_BREAKUP_THICKNESS_M
        {
            make_fresh_oceanic(planet, c, cache.area_m2[c as usize], ctx.ledger);
            planet.tectonic_flags[c as usize] |= cell_flags::RIFT;
        }
    }
}

/// Create the fixed mantle plumes on the first non-CrustalFormation step.
fn ensure_hotspots(planet: &mut Planet, ctx: &mut StepCtx) {
    if !planet.hotspots.is_empty() || planet.config.hotspot_count == 0 {
        return;
    }
    for _ in 0..planet.config.hotspot_count {
        let pos = random_unit(&mut ctx.rng);
        let strength = ctx.rng.random_range(0.5f32..2.0);
        planet.hotspots.push(Hotspot { pos, strength });
    }
}

/// Plume volcanism: basalt piles up on whatever cell sits over each plume.
fn hotspots(planet: &mut Planet, mesh: &Mesh, cache: &MeshCache, dt_myr: f64, ctx: &mut StepCtx) {
    let vigor = planet.config.tectonic_vigor;
    let plumes: Vec<(Vec3, f32)> = planet
        .hotspots
        .iter()
        .map(|h| (h.pos, h.strength))
        .collect();
    for (pos, strength) in plumes {
        let c = mesh.cell_at(pos);
        planet.tectonic_flags[c as usize] |= cell_flags::HOTSPOT;
        let d = HOTSPOT_DEPOSIT_M_MYR * strength * vigor * dt_myr as f32;
        if d <= 0.0 {
            continue;
        }
        deposit_new(
            planet,
            c,
            RockType::Basalt,
            d,
            cache.area_m2[c as usize],
            ctx.ledger,
        );
        let cap = if planet.crust_type[c as usize] == CrustType::Continental {
            HOTSPOT_MAX_CONTINENTAL_M
        } else {
            HOTSPOT_MAX_OCEANIC_M
        };
        planet.crust_thickness_m[c as usize] =
            (planet.crust_thickness_m[c as usize] + d * 0.8).min(cap);
    }
}

/// Update every plate's Euler vector from the accumulated torque.
#[allow(clippy::too_many_arguments)]
fn update_motion(
    planet: &mut Planet,
    torque: &[DVec3],
    area: &[f64],
    cont_area: &[f64],
    dt_myr: f64,
    vigor: f64,
    pitch_m: f64,
    total_area_m2: f64,
    ctx: &mut StepCtx,
) {
    // Stability: a boundary may migrate at most ~0.7 cells per step. At high
    // subdivision that cap would fall below the target speed range, so it is
    // floored -- boundary migration then saturates at one cell per step while
    // the plate velocity field stays physical.
    let v_cap = (0.7 * pitch_m / (dt_myr * 1.0e6)).clamp(MIN_SPEED_CAP_M_YR, MAX_PLATE_SPEED_M_YR);
    let w_cap = v_cap * 1.0e6 / EARTH_RADIUS_M;
    let blend = 1.0 - (-dt_myr / MOTION_TAU_MYR).exp();

    for p in 0..planet.plates.len() {
        if area[p] <= 0.0 {
            continue;
        }
        let cont_frac = (cont_area[p] / area[p]).clamp(0.0, 1.0);
        let drag = 1.0 + CONTINENTAL_DRAG * cont_frac;
        let area_eff = area[p].max(MIN_DRAG_AREA_FRAC * total_area_m2);
        let target = torque[p] * (TORQUE_GAIN * vigor / (area_eff * drag));
        let mut w = planet.plates[p].euler_pole * planet.plates[p].omega_rad_myr;
        w += (target - w) * blend;
        let jitter =
            random_unit(&mut ctx.rng).as_dvec3() * (MANTLE_JITTER_RAD_MYR * dt_myr * vigor);
        w += jitter;
        let mag = w.length();
        if mag > w_cap {
            w *= w_cap / mag;
        }
        let mag = w.length();
        if mag > 1.0e-12 {
            planet.plates[p].euler_pole = w / mag;
            planet.plates[p].omega_rad_myr = mag;
        } else {
            planet.plates[p].omega_rad_myr = 0.0;
        }
    }
}

/// Maybe split one plate along a great circle through its weakest crust.
/// Returns true when a rift actually opened.
#[allow(clippy::too_many_arguments)]
fn rift_step(
    planet: &mut Planet,
    mesh: &Mesh,
    cache: &MeshCache,
    ctx: &mut StepCtx,
    scratch: &mut Scratch,
    area: &[f64],
    cont_area: &[f64],
    cell_count: &[u32],
    dt_myr: f64,
) -> bool {
    let np = planet.plates.len();
    if np >= MAX_PLATES {
        return false;
    }
    let vigor = planet.config.tectonic_vigor as f64;
    let min_riftable = MIN_RIFTABLE_CELLS.max(planet.n_cells() / 80);
    let mut fired: Option<u16> = None;
    for p in 0..np {
        let frac = area[p] / cache.total_area_m2;
        let cfrac = if area[p] > 0.0 {
            cont_area[p] / area[p]
        } else {
            0.0
        };
        let per_myr = if frac > SUPERPLATE_AREA_FRAC && cfrac > SUPERPLATE_CONT_FRAC {
            RIFT_PROB_SUPER_PER_MYR
        } else {
            RIFT_PROB_BASE_PER_MYR
        } * vigor;
        let draw: f64 = ctx.rng.random();
        if draw < (per_myr * dt_myr).clamp(0.0, 1.0)
            && fired.is_none()
            && cell_count[p] as usize >= min_riftable
        {
            fired = Some(p as u16);
        }
    }
    let Some(p) = fired else { return false };
    split_plate(planet, mesh, ctx, scratch, p)
}

/// Cut plate `p` with a great circle through one of its weak points.
fn split_plate(
    planet: &mut Planet,
    mesh: &Mesh,
    ctx: &mut StepCtx,
    scratch: &mut Scratch,
    p: u16,
) -> bool {
    let n = planet.n_cells();
    let members: Vec<u32> = (0..n as u32)
        .filter(|c| planet.plate_id[*c as usize] == p)
        .collect();
    if members.len() < MIN_RIFTABLE_CELLS.max(n / 80) {
        return false;
    }
    // Prefer an ancient suture; otherwise the thinnest crust in the plate.
    let sutures: Vec<u32> = members
        .iter()
        .copied()
        .filter(|c| planet.tectonic_flags[*c as usize] & cell_flags::SUTURE != 0)
        .collect();
    let seed = if !sutures.is_empty() {
        sutures[ctx.rng.random_range(0..sutures.len())]
    } else {
        *members
            .iter()
            .min_by(|a, b| {
                planet.crust_thickness_m[**a as usize]
                    .total_cmp(&planet.crust_thickness_m[**b as usize])
            })
            .expect("non-empty")
    };
    let sc = mesh.centers[seed as usize];
    let axis = tangent_toward(sc, random_unit(&mut ctx.rng));
    // Cut plane: contains `sc` and `axis`; its normal is perpendicular to both.
    let cut = sc.cross(axis).normalize_or_zero();
    if cut.length_squared() < 0.5 {
        return false;
    }

    // Largest connected component of the plate on the negative side.
    let negative: Vec<u32> = members
        .iter()
        .copied()
        .filter(|c| mesh.centers[*c as usize].dot(cut) < 0.0)
        .collect();
    if negative.len() < MIN_RIFT_FRAGMENT || members.len() - negative.len() < MIN_RIFT_FRAGMENT {
        return false;
    }
    scratch.u32b.fill(u32::MAX);
    let mut comp_id = 0u32;
    let mut best: (u32, usize) = (u32::MAX, 0);
    let mut stack: Vec<u32> = Vec::new();
    let in_neg = |c: u32, planet: &Planet, mesh: &Mesh| -> bool {
        planet.plate_id[c as usize] == p && mesh.centers[c as usize].dot(cut) < 0.0
    };
    for &c in &negative {
        if scratch.u32b[c as usize] != u32::MAX {
            continue;
        }
        let id = comp_id;
        comp_id += 1;
        let mut size = 0usize;
        scratch.u32b[c as usize] = id;
        stack.push(c);
        while let Some(x) = stack.pop() {
            size += 1;
            for &m in mesh.neighbors_of(x) {
                if scratch.u32b[m as usize] == u32::MAX && in_neg(m, planet, mesh) {
                    scratch.u32b[m as usize] = id;
                    stack.push(m);
                }
            }
        }
        if size > best.1 {
            best = (id, size);
        }
    }
    if best.1 < MIN_RIFT_FRAGMENT || members.len() - best.1 < MIN_RIFT_FRAGMENT {
        return false;
    }

    let new_id = planet.plates.len() as u16;
    let mut new_sum = DVec3::ZERO;
    let mut old_sum = DVec3::ZERO;
    for &c in &members {
        if scratch.u32b[c as usize] == best.0 {
            planet.plate_id[c as usize] = new_id;
            new_sum += mesh.centers[c as usize].as_dvec3();
        } else {
            old_sum += mesh.centers[c as usize].as_dvec3();
        }
    }
    let r_new = new_sum.normalize_or(DVec3::Z).as_vec3();
    let r_old = old_sum.normalize_or(DVec3::Z).as_vec3();
    let w_old = planet.plates[p as usize].euler_pole * planet.plates[p as usize].omega_rad_myr;
    let push_new = omega_for_velocity(r_new, reproject(-cut, r_new) * RIFT_SEPARATION_M_YR);
    let push_old = omega_for_velocity(r_old, reproject(cut, r_old) * RIFT_SEPARATION_M_YR);
    let base = planet.plates[p as usize].clone();
    set_omega(&mut planet.plates[p as usize], w_old + push_old);
    let mut fresh = Plate {
        welded_to: None,
        ..base
    };
    set_omega(&mut fresh, w_old + push_new);
    planet.plates.push(fresh);

    // Mark and weaken the new boundary.
    for &c in &members {
        let mine = planet.plate_id[c as usize];
        for &m in mesh.neighbors_of(c) {
            if planet.plate_id[m as usize] != mine
                && (planet.plate_id[m as usize] == new_id || mine == new_id)
            {
                planet.tectonic_flags[c as usize] |= cell_flags::RIFT;
                let th = &mut planet.crust_thickness_m[c as usize];
                *th = (*th * 0.9).max(1_000.0);
                break;
            }
        }
    }
    ctx.progress.event(ProgressEvent::Milestone(format!(
        "rift opened: plate {p} split at {:.0} Myr",
        planet.time_myr
    )));
    true
}

fn set_omega(plate: &mut Plate, w: DVec3) {
    let mag = w.length();
    if mag > 1.0e-12 {
        plate.euler_pole = w / mag;
        plate.omega_rad_myr = mag;
    } else {
        plate.omega_rad_myr = 0.0;
    }
}

/// Merge the first plate pair that has stopped behaving like two plates:
/// either a continent-continent collision that has locked up (a true weld,
/// which leaves a suture), or any long boundary with negligible slip.
fn weld_step(
    planet: &mut Planet,
    edges: &[boundary::Edge],
    pairs: &FxHashMap<(u16, u16), PairAcc>,
    cell_count: &[u32],
    pitch_m: f64,
    ctx: &mut StepCtx,
) -> bool {
    if pairs.is_empty() || planet.plates.len() < 2 {
        return false;
    }
    let mut keys: Vec<(u16, u16)> = pairs.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let acc = pairs[&key];
        let welded = acc.coll_len_m >= WELD_MIN_PITCHES * pitch_m
            && acc.coll_conv_len / acc.coll_len_m.max(1.0) < WELD_SPEED_M_YR;
        let quiet = acc.len_m >= QUIET_MERGE_PITCHES * pitch_m
            && acc.rel_len / acc.len_m.max(1.0) < QUIET_MERGE_SPEED_M_YR;
        if !welded && !quiet {
            continue;
        }
        let (x, y) = key;
        let (keep, gone) = if cell_count[x as usize] >= cell_count[y as usize] {
            (x, y)
        } else {
            (y, x)
        };
        let mx = cell_count[keep as usize].max(1) as f64;
        let my = cell_count[gone as usize].max(1) as f64;
        let w = (planet.plates[keep as usize].euler_pole
            * planet.plates[keep as usize].omega_rad_myr
            * mx
            + planet.plates[gone as usize].euler_pole
                * planet.plates[gone as usize].omega_rad_myr
                * my)
            / (mx + my);
        set_omega(&mut planet.plates[keep as usize], w);
        for c in 0..planet.n_cells() {
            if planet.plate_id[c] == gone {
                planet.plate_id[c] = keep;
            }
        }
        if welded {
            for e in edges {
                if (e.pa == x && e.pb == y) || (e.pa == y && e.pb == x) {
                    planet.tectonic_flags[e.a as usize] |= cell_flags::SUTURE;
                    planet.tectonic_flags[e.b as usize] |= cell_flags::SUTURE;
                }
            }
        }
        ctx.progress.event(ProgressEvent::Milestone(format!(
            "plates {x} and {y} {} at {:.0} Myr",
            if welded { "welded" } else { "merged" },
            planet.time_myr
        )));
        return true;
    }
    false
}
