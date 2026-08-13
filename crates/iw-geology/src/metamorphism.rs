//! Burial metamorphism: transform strata in place once pressure/temperature
//! pass their thresholds (DESIGN.md §6).
//!
//! Temperature is the criterion. Depth sets the baseline through a linear
//! geotherm, and the tectonic flags add the magmatic contribution that lets
//! contact aureoles form at shallower depth than burial alone would allow.
//! Pressure is lithostatic and monotone in depth, so it carries no independent
//! information here; [`lithostatic_pressure_mpa`] exists for reporting.
//!
//! The sweep is **idempotent**: the resulting (rock, grade) is a pure function
//! of (rock, grade, temperature), and re-running at the same temperature is a
//! fixed point. Grade never decreases, so exhumed rock keeps its record.
//! Thickness is preserved, so metamorphism is volume-neutral in the ledger
//! (the density change is not accounted; the rock is the same material).

use iw_core::planet::cell_flags;
use iw_core::{MetamorphicGrade, Planet, RockType};
use rayon::prelude::*;

/// Linear geotherm, degrees C per km of burial.
pub const GEOTHERM_C_PER_KM: f32 = 25.0;
/// Extra temperature inside an active continent-continent collision, degrees C.
pub const COLLISION_BONUS_C: f32 = 200.0;
/// Extra temperature under a volcanic arc, degrees C.
pub const ARC_BONUS_C: f32 = 100.0;

/// Onset of low-grade metamorphism (~8 km of burial), degrees C.
pub const T_LOW_C: f32 = 200.0;
/// Onset of medium grade (~14 km), degrees C.
pub const T_MEDIUM_C: f32 = 350.0;
/// Onset of high grade (~22 km), degrees C.
pub const T_HIGH_C: f32 = 550.0;

/// Mean rock density used for the reported lithostatic pressure, kg/m^3.
const RHO_OVERBURDEN_KG_M3: f32 = 2700.0;

/// Lithostatic pressure at `depth_m`, MPa. Reporting only — grade thresholds
/// are temperature-based (module docs).
#[inline]
pub fn lithostatic_pressure_mpa(depth_m: f32) -> f32 {
    RHO_OVERBURDEN_KG_M3 * 9.81 * depth_m.max(0.0) * 1.0e-6
}

/// Grade implied by a temperature on its own.
#[inline]
pub fn grade_for_temperature(t_c: f32) -> MetamorphicGrade {
    if t_c >= T_HIGH_C {
        MetamorphicGrade::High
    } else if t_c >= T_MEDIUM_C {
        MetamorphicGrade::Medium
    } else if t_c >= T_LOW_C {
        MetamorphicGrade::Low
    } else {
        MetamorphicGrade::None
    }
}

/// Temperature of rock buried `depth_m` deep under a cell carrying
/// `tectonic_flags`, degrees C.
#[inline]
pub fn temperature_c(depth_m: f32, tectonic_flags: u8) -> f32 {
    let mut t = GEOTHERM_C_PER_KM * depth_m.max(0.0) / 1000.0;
    if tectonic_flags & cell_flags::COLLISION != 0 {
        t += COLLISION_BONUS_C;
    }
    if tectonic_flags & cell_flags::ARC != 0 {
        t += ARC_BONUS_C;
    }
    t
}

/// The transition table. Returns the new (rock, grade) when the stratum changes,
/// `None` when it is already stable at these conditions.
///
/// - shale -> slate -> schist -> gneiss (low / medium / high)
/// - limestone -> marble, sandstone -> quartzite (low and above)
/// - basalt, gabbro -> amphibolite (medium and above)
///
/// Everything else (already-igneous plutonic rock, evaporite, conglomerate,
/// tuff) is left alone.
pub fn metamorphose(
    rock: RockType,
    grade: MetamorphicGrade,
    t_c: f32,
) -> Option<(RockType, MetamorphicGrade)> {
    // Grade is a high-water mark: cooling does not un-metamorphose rock.
    let g = grade_for_temperature(t_c).max(grade);
    if g == MetamorphicGrade::None {
        return None;
    }
    let new_rock = match rock {
        RockType::Shale | RockType::Slate | RockType::Schist | RockType::Gneiss => match g {
            MetamorphicGrade::Low => RockType::Slate,
            MetamorphicGrade::Medium => RockType::Schist,
            MetamorphicGrade::High => RockType::Gneiss,
            MetamorphicGrade::None => unreachable!("guarded above"),
        },
        RockType::Limestone | RockType::Marble => RockType::Marble,
        RockType::Sandstone | RockType::Quartzite => RockType::Quartzite,
        RockType::Basalt | RockType::Gabbro | RockType::Amphibolite
            if g >= MetamorphicGrade::Medium =>
        {
            RockType::Amphibolite
        }
        _ => return None,
    };
    if new_rock == rock && g == grade {
        None
    } else {
        Some((new_rock, g))
    }
}

/// One stratum's pending transformation. Indices stay valid because the sweep
/// only rewrites strata in place — it never inserts, removes or reorders.
#[derive(Clone, Copy)]
struct Change {
    cell: u32,
    idx: u32,
    rock: RockType,
    grade: MetamorphicGrade,
}

/// Walk every column bottom-up and apply the transition table.
///
/// Burial depth of a stratum is measured to its mid-point through everything
/// above it: the strata above plus `sediment_m` (loose cover the column does
/// not record).
pub fn sweep(planet: &mut Planet) {
    let columns = &planet.columns;
    let sediment = &planet.sediment_m;
    let flags = &planet.tectonic_flags;

    let changes: Vec<Change> = (0..planet.n_cells() as u32)
        .into_par_iter()
        .flat_map_iter(|cell| {
            let col = columns.col(cell);
            let mut out: Vec<Change> = Vec::new();
            if col.is_empty() {
                return out.into_iter();
            }
            let total: f32 = col.iter().map(|s| s.thickness_m).sum();
            let cover = sediment[cell as usize].max(0.0);
            let f = flags[cell as usize];
            let mut below = 0.0f32;
            for (idx, s) in col.iter().enumerate() {
                let depth = cover + total - below - 0.5 * s.thickness_m;
                below += s.thickness_m;
                if let Some((rock, grade)) = metamorphose(s.rock, s.grade, temperature_c(depth, f))
                {
                    out.push(Change {
                        cell,
                        idx: idx as u32,
                        rock,
                        grade,
                    });
                }
            }
            out.into_iter()
        })
        .collect();

    for ch in changes {
        let s = &mut planet.columns.col_mut(ch.cell)[ch.idx as usize];
        s.rock = ch.rock;
        s.grade = ch.grade;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pelitic_chain_by_temperature() {
        let none = MetamorphicGrade::None;
        assert_eq!(metamorphose(RockType::Shale, none, 100.0), None);
        assert_eq!(
            metamorphose(RockType::Shale, none, 250.0),
            Some((RockType::Slate, MetamorphicGrade::Low))
        );
        assert_eq!(
            metamorphose(RockType::Shale, none, 400.0),
            Some((RockType::Schist, MetamorphicGrade::Medium))
        );
        assert_eq!(
            metamorphose(RockType::Shale, none, 700.0),
            Some((RockType::Gneiss, MetamorphicGrade::High))
        );
    }

    #[test]
    fn transformation_is_a_fixed_point() {
        for t in [150.0f32, 250.0, 400.0, 700.0] {
            for rock in RockType::ALL {
                let mut r = rock;
                let mut g = MetamorphicGrade::None;
                if let Some((nr, ng)) = metamorphose(r, g, t) {
                    r = nr;
                    g = ng;
                }
                assert_eq!(metamorphose(r, g, t), None, "{rock:?} at {t} C churns");
            }
        }
    }

    #[test]
    fn cooling_does_not_downgrade() {
        assert_eq!(
            metamorphose(RockType::Gneiss, MetamorphicGrade::High, 210.0),
            None
        );
    }

    #[test]
    fn basalt_needs_medium_grade() {
        assert_eq!(
            metamorphose(RockType::Basalt, MetamorphicGrade::None, 250.0),
            None
        );
        assert_eq!(
            metamorphose(RockType::Basalt, MetamorphicGrade::None, 400.0),
            Some((RockType::Amphibolite, MetamorphicGrade::Medium))
        );
    }
}
