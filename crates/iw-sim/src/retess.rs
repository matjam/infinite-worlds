//! Adaptive re-tessellation (docs/voronoi-v2.md §2): between eras the planet's
//! Voronoi mesh is rebuilt with density derived from the terrain the last era
//! produced — freshly risen mountain belts, coastlines and active margins get
//! small cells; plains and abyssal floor get large ones — and every field is
//! resampled onto the new cells. Pure function of `(planet, seed, epoch)`, so
//! determinism and checkpoint resume survive.

use std::sync::Arc;

use glam::Vec3;
use iw_core::planet::{cell_flags, FLOW_NONE};
use iw_core::{Planet, StrataColumns};
use iw_mesh::{sample::sample_generators, Mesh};

/// Slope (m per m) that saturates the density term: ~4 m/km is serious relief
/// at continental scale.
const SLOPE_FULL: f32 = 0.004;
/// Density floor (abyssal plains) and land base.
const DENSITY_OCEAN: f32 = 0.10;
const DENSITY_LAND: f32 = 0.30;
const DENSITY_COAST: f32 = 0.95;
const DENSITY_ACTIVE: f32 = 0.80;

/// Density of every OLD cell from the current terrain.
fn state_density(planet: &Planet, mesh: &Mesh) -> Vec<f32> {
    use rayon::prelude::*;
    let sea = planet.sea_level_m;
    (0..planet.n_cells())
        .into_par_iter()
        .map(|c| {
            let e = planet.elevation_m[c];
            let land = e >= sea;
            let mut d = if land { DENSITY_LAND } else { DENSITY_OCEAN };
            let mut max_slope = 0.0f32;
            let mut coast = false;
            for &m in mesh.neighbors_of(c as u32) {
                let em = planet.elevation_m[m as usize];
                let dist =
                    iw_mesh::great_circle_km(mesh.centers[c], mesh.centers[m as usize]) * 1_000.0;
                if dist > 0.0 {
                    max_slope = max_slope.max((e - em).abs() / dist);
                }
                if (em >= sea) != land {
                    coast = true;
                }
            }
            d = d.max((max_slope / SLOPE_FULL).clamp(0.0, 1.0));
            if coast {
                d = d.max(DENSITY_COAST);
            }
            let active = cell_flags::SUBDUCTING | cell_flags::COLLISION | cell_flags::RIFT;
            if planet.tectonic_flags[c] & active != 0 {
                d = d.max(DENSITY_ACTIVE);
            }
            d.clamp(0.06, 1.0)
        })
        .collect()
}

/// Re-tessellate `mesh` for the coming era and resample `planet` onto it.
/// No-op for legacy meshes without generators.
pub(crate) fn retessellate(planet: &mut Planet, mesh: &mut Arc<Mesh>, epoch: u64) {
    if mesh.generators.is_empty() {
        return;
    }
    let density_per_cell = state_density(planet, mesh);
    let old = Arc::clone(mesh);
    let density = move |dir: Vec3| density_per_cell[old.cell_at(dir) as usize];
    let gens = sample_generators(
        planet.config.cell_budget.max(64) as usize,
        planet
            .config
            .seed
            .wrapping_add(epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        &density,
        2,
    );
    let new_mesh = Mesh::build_from_generators(&gens);
    resample_in_place(planet, mesh, &new_mesh);
    planet.mesh_generators = gens;
    *mesh = Arc::new(new_mesh);
}

/// Nearest-cell resampling of every per-cell field (and column) onto the new
/// tessellation. Not mass-exact — cell areas change — but conservative enough
/// between eras; the isostasy/sea-level solve re-equilibrates on the first
/// step of the new era.
fn resample_in_place(planet: &mut Planet, old_mesh: &Mesh, new_mesh: &Mesh) {
    use rayon::prelude::*;
    let n = new_mesh.n_cells();
    let src: Vec<u32> = new_mesh
        .centers
        .par_iter()
        .map(|c| old_mesh.cell_at(*c))
        .collect();

    macro_rules! pull {
        ($field:ident) => {{
            let new: Vec<_> = src
                .iter()
                .map(|s| planet.$field[*s as usize].clone())
                .collect();
            planet.$field = new;
        }};
    }
    pull!(plate_id);
    pull!(crust_type);
    pull!(crust_thickness_m);
    pull!(crust_density_kg_m3);
    pull!(crust_age_myr);
    pull!(elevation_m);
    pull!(sediment_m);
    pull!(temperature_c);
    pull!(precip_mm_yr);
    pull!(wind_m_s);
    pull!(ice_thickness_m);
    pull!(water_flux_m3_yr);
    pull!(lake_depth_m);
    pull!(biome);
    pull!(tectonic_flags);
    let mut columns = StrataColumns::new(n);
    for (d, s) in src.iter().enumerate() {
        columns.copy_col_from(d as u32, &planet.columns, *s);
    }
    planet.columns = columns;
    // Flow routing is mesh-topological: recomputed by the surface process.
    planet.flow_to = vec![FLOW_NONE; n];
}
