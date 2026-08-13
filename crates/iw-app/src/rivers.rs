//! River ribbon geometry for the renderer, built from the drainage graph
//! (`flow_to` + `water_flux_m3_yr`). One quad per drainage edge, widths and
//! opacity scaled by flux. Widths are exaggerated relative to real rivers —
//! at marble distance a pixel is ~13 km, and the point is that the Amazon
//! reads from orbit.

use glam::Vec3;
use iw_core::view::ViewCells;
use iw_mesh::Mesh;
use iw_render_vulkan::globe::RiverVertex;

/// Flux below which a river is not drawn, m^3/yr (~160 m^3/s; anything
/// smaller is sub-pixel noise at globe distance).
///
/// Calibration (seed 42 @160k): land flux p99 = 2.3e10, max = 2.1e11 — the
/// fragmented continents top out at Mississippi-order rivers, not the
/// Amazon, so the ramp is normalised to what these worlds actually produce.
const MIN_FLUX_M3_YR: f32 = 5.0e9;
/// Flux at which a ribbon reaches full width and opacity.
const MAX_FLUX_M3_YR: f32 = 3.0e11;
const MIN_HALF_WIDTH_KM: f32 = 2.5;
const MAX_HALF_WIDTH_KM: f32 = 10.0;
/// Slightly lighter than the ocean fill so rivers read against coastal water.
const RIVER_RGB: [f32; 3] = [0.13, 0.33, 0.55];

/// Build the ribbon vertex list. Empty when the view has no drainage data
/// (early phases) — the renderer treats that as "no rivers".
pub fn build(mesh: &Mesh, cells: &ViewCells, sea_level_m: f32) -> Vec<RiverVertex> {
    let n = mesh.n_cells();
    if cells.flow_to.len() != n || cells.water_flux_m3_yr.len() != n {
        return Vec::new();
    }
    let ln_min = MIN_FLUX_M3_YR.ln();
    let ln_max = MAX_FLUX_M3_YR.ln();
    let mut verts = Vec::new();
    for i in 0..n {
        let j = cells.flow_to[i];
        if j as usize >= n {
            continue;
        }
        let flux = cells.water_flux_m3_yr[i];
        if flux < MIN_FLUX_M3_YR {
            continue;
        }
        // Source must be land (submarine "flow" is basin bookkeeping); the
        // mouth may dip into the sea, clamped to the waterline so the ribbon
        // meets the ocean surface instead of diving under it.
        if cells.elevation_m[i] < sea_level_m {
            continue;
        }
        let j = j as usize;
        // Width and opacity are evaluated at BOTH ends from each end's own
        // flux, so a segment tapers as the river grows and every segment
        // meeting at a node shares that node's width — joints stop popping.
        let ramp = |f: f32| ((f.max(1.0).ln() - ln_min) / (ln_max - ln_min)).clamp(0.0, 1.0);
        let ta = ramp(flux);
        let tb = ramp(cells.water_flux_m3_yr[j].max(flux));
        let width = |t: f32| MIN_HALF_WIDTH_KM + (MAX_HALF_WIDTH_KM - MIN_HALF_WIDTH_KM) * t;
        let shade = |t: f32| [RIVER_RGB[0], RIVER_RGB[1], RIVER_RGB[2], 0.45 + 0.45 * t];

        let a: Vec3 = mesh.centers[i];
        let b: Vec3 = mesh.centers[j];
        let mid = (a + b).normalize_or_zero();
        let chord = b - a;
        let dir = (chord - mid * chord.dot(mid)).normalize_or_zero();
        if dir == Vec3::ZERO || mid == Vec3::ZERO {
            continue;
        }
        let side = mid.cross(dir).normalize_or_zero();
        let off_a = side * (width(ta) / iw_mesh::EARTH_RADIUS_KM);
        let off_b = side * (width(tb) / iw_mesh::EARTH_RADIUS_KM);

        let ea = (cells.elevation_m[i] - sea_level_m).max(0.0) + sea_level_m;
        let eb = cells.elevation_m[j].max(sea_level_m);
        let corner = |p: Vec3, off: Vec3, elev: f32, color: [f32; 4]| RiverVertex {
            pos: (p + off).normalize().to_array(),
            elevation_m: elev,
            color,
        };
        let (a1, a2) = (
            corner(a, -off_a, ea, shade(ta)),
            corner(a, off_a, ea, shade(ta)),
        );
        let (b1, b2) = (
            corner(b, -off_b, eb, shade(tb)),
            corner(b, off_b, eb, shade(tb)),
        );
        verts.extend_from_slice(&[a1, b1, a2, a2, b1, b2]);
    }
    verts
}

#[cfg(test)]
mod tests {
    /// Diagnostic: how many ribbons does a saved planet yield, and what does
    /// its land flux distribution look like? Run with
    /// `IW_PLANET_DIR=<dir> cargo test -p iw-app probe_river_geometry -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_river_geometry() {
        let dir = std::path::PathBuf::from(std::env::var("IW_PLANET_DIR").expect("IW_PLANET_DIR"));
        let store = iw_store_postcard::FileStore::new(dir).unwrap();
        let planet = iw_core::CheckpointStore::load(&store, "phase-recent_past").unwrap();
        let mesh = iw_mesh::Mesh::build_from_generators(&planet.mesh_generators);
        let view = iw_core::PlanetView::capture(1, &planet, &mesh);
        let n = mesh.n_cells();
        let mut fluxes: Vec<f32> = (0..n)
            .filter(|&i| {
                view.cells.elevation_m[i] >= view.sea_level_m
                    && (view.cells.flow_to[i] as usize) < n
            })
            .map(|i| view.cells.water_flux_m3_yr[i])
            .collect();
        fluxes.sort_by(f32::total_cmp);
        let pct = |q: f64| fluxes[((fluxes.len() as f64 - 1.0) * q) as usize];
        println!(
            "land flow cells {} | flux p50 {:.2e} p90 {:.2e} p99 {:.2e} max {:.2e}",
            fluxes.len(),
            pct(0.5),
            pct(0.9),
            pct(0.99),
            fluxes.last().copied().unwrap_or(0.0)
        );
        let verts = super::build(&mesh, &view.cells, view.sea_level_m);
        println!(
            "river ribbon vertices: {} ({} quads)",
            verts.len(),
            verts.len() / 6
        );
    }
}
