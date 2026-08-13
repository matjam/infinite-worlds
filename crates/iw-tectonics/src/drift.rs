//! The main tectonic engine: force balance, boundary resolution, rifting,
//! welding, hotspots and oceanic aging (DESIGN.md §5 Phase 2-4).

use glam::{DVec3, Vec3};
use iw_core::planet::cell_flags;
use iw_core::{CrustType, Hotspot, Phase, Planet, Plate, ProgressEvent, RockType, StepCtx};
use iw_mesh::{Mesh, EARTH_RADIUS_M};
use rand::Rng;
use rustc_hash::FxHashMap;

use crate::advect;
use crate::boundary::{self, Kind};
use crate::crust::{deposit_new, make_fresh_oceanic};
use crate::geom::{omega_for_velocity, random_unit, reproject, tangent_toward};
use crate::phase1;
use crate::topology::{absorb_tiny_plates, compact_plates, enforce_contiguity};
use crate::{
    MeshCache, Scratch, MAX_CRUST_THICKNESS_M, MAX_PLATES, MIN_RIFTABLE_CELLS,
    RIFT_BREAKUP_THICKNESS_M, TRENCH_THICKNESS_M,
};

// --- force balance ---

/// Slab pull per metre of subducting boundary (reference units).
const SLAB_PULL: f64 = 1.0;
/// Ridge push per metre of spreading boundary, relative to slab pull.
const RIDGE_PUSH: f64 = 0.22;
/// Viscous braking per metre of collisional boundary, per (m/yr) of closure.
const COLLISION_VISC: f64 = 100.0;
/// Converts driving torque density into angular velocity, rad*m/Myr.
/// Calibrated so a plate with a normal share of subducting boundary settles
/// around 4-6 cm/yr at `tectonic_vigor == 1`.
const TORQUE_GAIN: f64 = 2.0e5;
/// Smallest plate area, as a fraction of the planet, that the drag term will
/// use. Without it a microplate's perimeter-to-area ratio drives it straight
/// into the speed cap; real microplates are dragged along by their neighbours.
const MIN_DRAG_AREA_FRAC: f64 = 0.10;
/// Extra basal drag from thick continental keels, as a fraction. Kept mild:
/// heavy keel drag is one of the three things that parked continents entirely
/// (with eager welding and collisional braking).
const CONTINENTAL_DRAG: f64 = 0.6;
/// Relaxation time of a plate's angular velocity toward the force balance,
/// Myr. Long on purpose: real plates hold a heading for 100+ Myr (the Atlantic
/// has opened monotonically since Pangaea), and short relaxation turns drift
/// into a random walk that nets almost no displacement over an era. 25 Myr
/// balances heading persistence against a tolerable spin-up time from rest.
const MOTION_TAU_MYR: f64 = 25.0;
/// Random mantle-torque walk, rad/Myr per Myr. Keeps plates whose boundaries
/// are all transform from stalling completely.
const MANTLE_JITTER_RAD_MYR: f64 = 0.0008;
/// Absolute plate speed ceiling, m/yr.
const MAX_PLATE_SPEED_M_YR: f64 = 0.15;

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
/// Oceanic arc crust this thick has differentiated into juvenile continental
/// crust (island-arc accretion). Slightly denser than craton basement.
const ARC_MATURE_THICKNESS_M: f32 = 20_000.0;
const ARC_MATURE_DENSITY_KG_M3: f32 = 2_800.0;
/// Fraction of the crust-thickness deficit a trench cell closes per unit of
/// (closure * dt / pitch).
const TRENCH_FLEX_RATE: f32 = 4.0;
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
const RIFT_SEPARATION_M_YR: f32 = 0.05;
/// Smallest fragment a rift may produce, in cells.
const MIN_RIFT_FRAGMENT: usize = 6;
/// Collisional boundary length, in cell pitches, needed before plates may weld.
const WELD_MIN_PITCHES: f64 = 2.0;
/// Closure rate below which a collision is considered locked, m/yr. Welding
/// waits for a truly stalled collision — eager welding freezes continents in
/// place and kills visible drift; the braking term does the slowing first.
const WELD_SPEED_M_YR: f64 = 0.008;
/// A long boundary with almost no relative motion is not a boundary: the two
/// plates are merged. This is what keeps the mosaic from fragmenting forever.
/// Kept strict (long + truly dead) so merging can't collapse the mosaic into
/// a single superplate.
const QUIET_MERGE_PITCHES: f64 = 12.0;
const QUIET_MERGE_SPEED_M_YR: f64 = 0.002;
/// Area fraction beyond which any plate rifts on size alone.
const GIANT_PLATE_AREA_FRAC: f64 = 0.40;

/// Basalt erupted over a plume, m/Myr at strength 1.
const HOTSPOT_DEPOSIT_M_MYR: f32 = 40.0;
const HOTSPOT_MAX_OCEANIC_M: f32 = 25_000.0;
const HOTSPOT_MAX_CONTINENTAL_M: f32 = 50_000.0;

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

    // Drift v2: move the crust itself, then let boundary physics act on the
    // moved geometry (docs/drift-v2.md).
    let mut topology_dirty = advect::accumulate_and_remap(planet, mesh, cache, dt_myr, ctx.ledger);

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
                // Sub-cell spreading between remaps: the ridge keeps erupting
                // even while accumulated displacement is below one pitch, so
                // spreading centres always carry age-0 basalt. The advect
                // remap then carries this young floor away.
                if planet.crust_type[e.a as usize] == CrustType::Oceanic
                    && planet.crust_type[e.b as usize] == CrustType::Oceanic
                {
                    let p_renew = (open as f64 * dt_myr * 1.0e6 / pitch).clamp(0.0, 1.0);
                    if ctx.rng.random::<f64>() < p_renew {
                        let renew = if planet.crust_age_myr[e.a as usize]
                            >= planet.crust_age_myr[e.b as usize]
                        {
                            e.a
                        } else {
                            e.b
                        };
                        if !scratch.flags[renew as usize] {
                            scratch.flags[renew as usize] = true;
                            make_fresh_oceanic(
                                planet,
                                renew,
                                cache.ocean_thickness_m[renew as usize],
                                cache.area_m2[renew as usize],
                                ctx.ledger,
                            );
                            planet.tectonic_flags[renew as usize] |= cell_flags::RIFT;
                        }
                    }
                }
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

                    // Consumption of the subducting cell happens in the advect
                    // remap when the overriding plate's crust arrives; here we
                    // only apply the continuous effects (flexure, arc, pull).
                    let _ = ovr_plate;
                }
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
            // Arc maturation: sustained arc magmatism turns thick oceanic arc
            // crust into juvenile continental crust (island-arc accretion) —
            // the source term that balances the continental area consumed by
            // collisions, as on Earth.
            if planet.crust_type[c as usize] == CrustType::Oceanic
                && planet.tectonic_flags[c as usize] & cell_flags::ARC != 0
                && planet.crust_thickness_m[c as usize] >= ARC_MATURE_THICKNESS_M
            {
                planet.crust_type[c as usize] = CrustType::Continental;
                planet.crust_density_kg_m3[c as usize] = ARC_MATURE_DENSITY_KG_M3;
                planet.crust_age_myr[c as usize] = 0.0;
            }
        }
    }

    relax_thickness(planet, cache, dt_myr);
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
fn relax_thickness(planet: &mut Planet, cache: &MeshCache, dt_myr: f64) {
    let dt = dt_myr as f32;
    for c in 0..planet.n_cells() {
        let flexing = planet.tectonic_flags[c] & cell_flags::SUBDUCTING != 0;
        let th = planet.crust_thickness_m[c];
        match planet.crust_type[c] {
            CrustType::Oceanic if !flexing => {
                let w = (dt / OCEAN_RELAX_TAU_MYR).clamp(0.0, 1.0);
                planet.crust_thickness_m[c] = th + (cache.ocean_thickness_m[c] - th) * w;
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
            make_fresh_oceanic(
                planet,
                c,
                cache.ocean_thickness_m[c as usize],
                cache.area_m2[c as usize],
                ctx.ledger,
            );
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
    // Drift v2: the advection remap handles any per-step displacement, so the
    // only cap is the physical plate-speed ceiling — no per-pitch stability
    // floor, no migration saturation at high subdivision.
    let _ = pitch_m;
    let w_cap = MAX_PLATE_SPEED_M_YR * 1.0e6 / EARTH_RADIUS_M;
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
        // A mostly-continental superplate rifts along its sutures; but ANY
        // plate past the giant threshold rifts on sheer size — slab pull tears
        // huge oceanic plates apart regardless of composition. Without the
        // size-only branch the mosaic collapses into one immortal superplate
        // once aggressive welding has merged everything.
        let per_myr = if frac > SUPERPLATE_AREA_FRAC && cfrac > SUPERPLATE_CONT_FRAC {
            RIFT_PROB_SUPER_PER_MYR
        } else if frac > GIANT_PLATE_AREA_FRAC {
            RIFT_PROB_SUPER_PER_MYR * ((frac - GIANT_PLATE_AREA_FRAC) / 0.2).min(2.0)
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

/// Split plate `p` along a crooked rift path grown through its weakest crust.
///
/// The path follows a weakness field (ancient sutures, ridged noise creases,
/// thin crust) with directional persistence — the East-African-Rift look —
/// instead of a great circle. Drift v2's advection then genuinely separates
/// the halves and floors the widening gap with ridge crust.
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

    // Weakness field: suture bonus + ridged-noise creases + crustal thinness.
    let crease = iw_core::noise::Noise3::new(planet.config.seed ^ 0x52_49_46_54); // "RIFT"
    let weakness = |c: u32, planet: &Planet| -> f32 {
        let ci = c as usize;
        let suture = if planet.tectonic_flags[ci] & cell_flags::SUTURE != 0 {
            1.0
        } else {
            0.0
        };
        let ridge = crease.ridged(mesh.centers[ci] * 3.0, 4, 2.0, 0.5);
        let thin = (1.0 - planet.crust_thickness_m[ci] / 45_000.0).clamp(0.0, 1.0);
        suture + 0.8 * ridge + 0.4 * thin
    };

    // Nucleate at the weakest cell (deterministic argmax, rng only for ties
    // via the jitter inside path growth).
    let seed = *members
        .iter()
        .max_by(|a, b| {
            weakness(**a, planet)
                .total_cmp(&weakness(**b, planet))
                .then(b.cmp(a))
        })
        .expect("non-empty");

    // Grow the path from the seed in two opposite directions.
    scratch.u32b.fill(u32::MAX); // u32::MAX = untouched; 0 = path member
    let mut path: Vec<u32> = vec![seed];
    scratch.u32b[seed as usize] = 0;
    let init_dir = tangent_toward(mesh.centers[seed as usize], random_unit(&mut ctx.rng));
    for leg_sign in [1.0f32, -1.0] {
        let mut cur = seed;
        let mut dir = init_dir * leg_sign;
        loop {
            let mut best: Option<(f32, u32)> = None;
            for &m in mesh.neighbors_of(cur) {
                if planet.plate_id[m as usize] != p {
                    // Reached the plate edge: this leg is complete.
                    best = None;
                    break;
                }
                if scratch.u32b[m as usize] == 0 {
                    continue;
                }
                let step = reproject(
                    mesh.centers[m as usize] - mesh.centers[cur as usize],
                    mesh.centers[cur as usize],
                )
                .normalize_or_zero();
                let persist = step.dot(dir);
                if persist < -0.2 {
                    continue; // no hairpins
                }
                let jitter: f32 = ctx.rng.random_range(0.0..0.10);
                let score = weakness(m, planet) + 0.9 * persist + jitter;
                if best.map(|(s, _)| score > s).unwrap_or(true) {
                    best = Some((score, m));
                }
            }
            let Some((_, next)) = best else { break };
            let step = reproject(
                mesh.centers[next as usize] - mesh.centers[cur as usize],
                mesh.centers[next as usize],
            )
            .normalize_or_zero();
            dir = (0.6 * dir + 0.4 * step).normalize_or_zero();
            scratch.u32b[next as usize] = 0;
            path.push(next);
            cur = next;
            if path.len() > members.len() {
                break; // safety, unreachable in practice
            }
        }
    }
    if path.len() < 3 {
        return false;
    }

    // Components of the plate minus the path; the path must sever it.
    let mut comp_id = 1u32; // 0 is the path marker
    let mut sizes: Vec<(u32, usize)> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    for &c in &members {
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
                if scratch.u32b[m as usize] == u32::MAX && planet.plate_id[m as usize] == p {
                    scratch.u32b[m as usize] = id;
                    stack.push(m);
                }
            }
        }
        sizes.push((id, size));
    }
    sizes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if sizes.len() < 2 || sizes[1].1 < MIN_RIFT_FRAGMENT {
        return false; // path failed to sever anything worth calving
    }
    let new_comp = sizes[1].0;

    // The second-largest component becomes the new plate; every other
    // component (including the path cells, resolved by neighbor majority)
    // stays with the old plate.
    let new_id = planet.plates.len() as u16;
    let mut new_sum = DVec3::ZERO;
    let mut old_sum = DVec3::ZERO;
    for &c in &members {
        let comp = scratch.u32b[c as usize];
        let to_new = if comp == 0 {
            let mut votes = 0i32;
            for &m in mesh.neighbors_of(c) {
                match scratch.u32b[m as usize] {
                    x if x == new_comp => votes += 1,
                    0 | u32::MAX => {}
                    _ => votes -= 1,
                }
            }
            votes > 0
        } else {
            comp == new_comp
        };
        if to_new {
            planet.plate_id[c as usize] = new_id;
            new_sum += mesh.centers[c as usize].as_dvec3();
        } else {
            old_sum += mesh.centers[c as usize].as_dvec3();
        }
    }
    let r_new = new_sum.normalize_or(DVec3::Z).as_vec3();
    let r_old = old_sum.normalize_or(DVec3::Z).as_vec3();
    // Push the halves apart along the line between their centroids.
    let sep = reproject(r_new - r_old, r_new).normalize_or_zero();
    let w_old = planet.plates[p as usize].euler_pole * planet.plates[p as usize].omega_rad_myr;
    let push_new = omega_for_velocity(r_new, sep * RIFT_SEPARATION_M_YR);
    let push_old = omega_for_velocity(r_old, reproject(-sep, r_old) * RIFT_SEPARATION_M_YR);
    let base = planet.plates[p as usize].clone();
    set_omega(&mut planet.plates[p as usize], w_old + push_old);
    let mut fresh = Plate {
        welded_to: None,
        accum: glam::DQuat::IDENTITY,
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
