//! Hypsometric ocean fill: solve the water surface that exactly holds the
//! configured water budget (DESIGN.md §5, "Ocean fill").

use iw_core::PlanetConfig;
use iw_mesh::Mesh;

/// Volume of Earth's oceans, km^3. `water_budget = 1.0` means this much water,
/// scaled by the mesh's surface area relative to Earth's.
pub const EARTH_OCEAN_VOLUME_KM3: f64 = 1.335e9;

/// Total water the planet must hold, km^3.
///
/// The mesh area is summed rather than assumed so a mesh built at a different
/// radius still gets a proportionate ocean.
pub fn water_volume_km3(config: &PlanetConfig, mesh: &Mesh) -> f64 {
    let mesh_area: f64 = mesh.areas_km2.iter().map(|a| *a as f64).sum();
    let r = iw_mesh::EARTH_RADIUS_KM as f64;
    let earth_area = 4.0 * std::f64::consts::PI * r * r;
    config.water_budget * EARTH_OCEAN_VOLUME_KM3 * (mesh_area / earth_area)
}

/// Sea level (metres, same datum as `elevation_m`) such that
/// `sum over cells with elevation < level of (level - elevation) * area`
/// equals `water_volume_km3`.
///
/// Exact rather than iterative: cells are sorted by elevation and the level is
/// solved in closed form on the segment that contains it, so the residual is at
/// float precision (far inside the 1 m convergence requirement). A zero or
/// negative budget puts the level 1 m below the global minimum, leaving no
/// flooded cell.
///
/// Allocates its sort scratch; the per-step path uses
/// [`solve_sea_level_m_with`] to reuse it.
pub fn solve_sea_level_m(elevation_m: &[f32], areas_km2: &[f32], water_volume_km3: f64) -> f32 {
    let mut order = Vec::new();
    solve_sea_level_m_with(elevation_m, areas_km2, water_volume_km3, &mut order)
}

/// [`solve_sea_level_m`] reusing a caller-owned ordering buffer. The buffer is
/// rebuilt every call, so it carries no state between steps.
pub fn solve_sea_level_m_with(
    elevation_m: &[f32],
    areas_km2: &[f32],
    water_volume_km3: f64,
    order: &mut Vec<u32>,
) -> f32 {
    debug_assert_eq!(elevation_m.len(), areas_km2.len());
    if elevation_m.is_empty() {
        return 0.0;
    }

    order.clear();
    order.extend(0..elevation_m.len() as u32);
    // Total order (elevation, cell id): an unstable sort is still deterministic.
    order.sort_unstable_by(|a, b| {
        elevation_m[*a as usize]
            .total_cmp(&elevation_m[*b as usize])
            .then(a.cmp(b))
    });

    let lowest = elevation_m[order[0] as usize];
    if water_volume_km3 <= 0.0 || water_volume_km3.is_nan() {
        return lowest - 1.0;
    }

    // Flood the k lowest cells and ask what level that volume implies:
    //   V(level) = sum_{i<k} (level_m - e_i)/1000 * A_i
    //   level_m  = (V + sum e_i/1000*A_i) * 1000 / sum A_i
    // The answer is valid as soon as it does not reach the next cell up.
    let mut cum_area_km2 = 0.0f64;
    let mut cum_ea_km3 = 0.0f64;
    for k in 0..order.len() {
        let c = order[k] as usize;
        let a = areas_km2[c] as f64;
        cum_area_km2 += a;
        cum_ea_km3 += elevation_m[c] as f64 / 1000.0 * a;
        if cum_area_km2 <= 0.0 {
            continue;
        }
        let level = (water_volume_km3 + cum_ea_km3) * 1000.0 / cum_area_km2;
        let next = order.get(k + 1).map(|c| elevation_m[*c as usize] as f64);
        match next {
            Some(next) if level > next => continue,
            _ => return level as f32,
        }
    }
    // Unreachable: the last iteration has no `next` and always returns.
    lowest - 1.0
}

/// Water volume held below `level`, km^3 — the inverse used by tests and by
/// callers checking convergence.
pub fn flooded_volume_km3(elevation_m: &[f32], areas_km2: &[f32], level_m: f32) -> f64 {
    elevation_m
        .iter()
        .zip(areas_km2)
        .filter(|(e, _)| **e < level_m)
        .map(|(e, a)| (level_m - *e) as f64 / 1000.0 * *a as f64)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_level_world_is_analytic() {
        // Four cells of 100 km^2: two at -1000 m, two at +1000 m.
        let elev = [-1000.0, 1000.0, -1000.0, 1000.0];
        let areas = [100.0f32; 4];
        // Fill the low pair to -500 m: 0.5 km * 200 km^2 = 100 km^3.
        let level = solve_sea_level_m(&elev, &areas, 100.0);
        assert!((level - -500.0).abs() < 1.0, "level {level}");
    }

    #[test]
    fn budget_zero_leaves_everything_dry() {
        let elev = [-1000.0, 1000.0];
        let areas = [100.0f32; 2];
        let level = solve_sea_level_m(&elev, &areas, 0.0);
        assert!((level - -1001.0).abs() < 1e-3);
        assert_eq!(flooded_volume_km3(&elev, &areas, level), 0.0);
    }

    #[test]
    fn huge_budget_drowns_everything() {
        let elev = [-1000.0, 1000.0];
        let areas = [100.0f32; 2];
        let level = solve_sea_level_m(&elev, &areas, 1.0e6);
        assert!(level > 1000.0, "level {level}");
        let v = flooded_volume_km3(&elev, &areas, level);
        assert!((v - 1.0e6).abs() < 1.0, "volume {v}");
    }
}
