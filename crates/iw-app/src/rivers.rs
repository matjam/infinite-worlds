//! River geometry for the renderer, built from the drainage graph
//! (`flow_to` + `water_flux_m3_yr`).
//!
//! Rivers are drawn as SMOOTH TAPERED STRIPS: downstream chains of cells are
//! walked into polylines, each polyline is Catmull-Rom smoothed through its
//! cell centres, and a triangle strip is laid along the curve whose width and
//! opacity grow with flux. Tributaries end by flowing INTO the cell of the
//! river they join, so confluences connect instead of butting.
//!
//! The draw threshold is RELATIVE — the top slice of this planet's own flux
//! distribution — not an absolute discharge. During a glacial epoch every
//! river's flux halves; with an absolute cutoff the map's rivers "mostly
//! disappeared" at the end of generation, when what a viewer wants is the
//! biggest rivers of the world as it now is.

use glam::Vec3;
use iw_core::view::ViewCells;
use iw_mesh::Mesh;
use iw_render_vulkan::globe::RiverVertex;

/// Fraction of positive-flux land cells drawn (by flux rank).
const DRAW_TOP_FRACTION: f64 = 0.10;
/// Absolute floor under the relative threshold: a bone-dry world should show
/// few rivers, not its top 10% of trickles.
const ABS_MIN_FLUX_M3_YR: f32 = 1.0e9;
const MIN_HALF_WIDTH_KM: f32 = 2.0;
const MAX_HALF_WIDTH_KM: f32 = 9.0;
/// Curve samples per cell-to-cell hop.
const SUBDIV: usize = 4;
/// Slightly lighter than the ocean fill so rivers read against coastal water.
const RIVER_RGB: [f32; 3] = [0.13, 0.33, 0.55];

/// Build the strip vertex list. Empty when the view has no drainage data
/// (early phases) — the renderer treats that as "no rivers".
pub fn build(mesh: &Mesh, cells: &ViewCells, sea_level_m: f32) -> Vec<RiverVertex> {
    let n = mesh.n_cells();
    if cells.flow_to.len() != n || cells.water_flux_m3_yr.len() != n {
        return Vec::new();
    }

    // Relative flux threshold: the top DRAW_TOP_FRACTION of flowing land.
    let mut fluxes: Vec<f32> = (0..n)
        .filter(|&i| {
            (cells.flow_to[i] as usize) < n
                && cells.elevation_m[i] >= sea_level_m
                && cells.lake_depth_m.get(i).copied().unwrap_or(0.0) <= 0.0
                && cells.water_flux_m3_yr[i] > 0.0
        })
        .map(|i| cells.water_flux_m3_yr[i])
        .collect();
    if fluxes.is_empty() {
        return Vec::new();
    }
    fluxes.sort_unstable_by(f32::total_cmp);
    let cut = ((fluxes.len() as f64) * (1.0 - DRAW_TOP_FRACTION)) as usize;
    let min_flux = fluxes[cut.min(fluxes.len() - 1)].max(ABS_MIN_FLUX_M3_YR);
    let max_flux = fluxes[fluxes.len() - 1].max(min_flux * 8.0);
    let (ln_min, ln_max) = (min_flux.ln(), max_flux.ln());
    let ramp = |f: f32| ((f.max(1.0).ln() - ln_min) / (ln_max - ln_min).max(1e-3)).clamp(0.0, 1.0);

    // Lakes are excluded: the drainage graph legitimately flows THROUGH a
    // lake to its outlet, but the river is the lake there — drawing the
    // chain drew wide ribbons arcing across every lake surface. Breaking at
    // the lake shore makes the outlet cell a fresh source, so the river
    // resumes below the lake by itself.
    let qualifies = |i: usize| {
        (cells.flow_to[i] as usize) < n
            && cells.elevation_m[i] >= sea_level_m
            && cells.lake_depth_m.get(i).copied().unwrap_or(0.0) <= 0.0
            && cells.water_flux_m3_yr[i] >= min_flux
    };

    // Sources: qualifying cells no other qualifying cell drains into.
    let mut has_upstream = vec![false; n];
    for i in 0..n {
        if qualifies(i) {
            has_upstream[cells.flow_to[i] as usize] = true;
        }
    }

    // Walk each chain downstream; a chain ends at the sea, at a cell already
    // claimed by another chain (a confluence — include it so the strips
    // join), or when the flux drops back under the threshold.
    let mut claimed = vec![false; n];
    let mut verts = Vec::new();
    for start in 0..n {
        if !qualifies(start) || has_upstream[start] || claimed[start] {
            continue;
        }
        let mut path: Vec<usize> = vec![start];
        claimed[start] = true;
        let mut cur = start;
        // Sanity bound: consecutive path cells are mesh neighbours, so their
        // chord can never exceed a few cell pitches. Anything longer is
        // corrupt drainage data, and one bad hop drew a strip beelining
        // across an ocean.
        let max_chord2 = {
            let pitch = (4.0 * std::f32::consts::PI / n as f32).sqrt() * 3.0;
            pitch * pitch
        };
        loop {
            let next = cells.flow_to[cur] as usize;
            if next >= n {
                break;
            }
            if (mesh.centers[next] - mesh.centers[cur]).length_squared() > max_chord2 {
                break;
            }
            path.push(next);
            if claimed[next] || !qualifies(next) {
                break; // confluence or mouth: keep the join point, stop.
            }
            claimed[next] = true;
            cur = next;
        }
        if path.len() >= 2 {
            emit_strip(&path, mesh, cells, sea_level_m, &ramp, &mut verts);
        }
    }
    verts
}

/// Catmull-Rom sample of a polyline of unit vectors at parameter `t` in
/// `0..len-1` (chords are tiny, so blending in 3D and renormalising is fine).
fn spline_point(pts: &[Vec3], t: f32) -> Vec3 {
    let last = pts.len() - 1;
    let seg = (t.floor() as usize).min(last - 1);
    let u = t - seg as f32;
    let p0 = pts[seg.saturating_sub(1)];
    let p1 = pts[seg];
    let p2 = pts[seg + 1];
    let p3 = pts[(seg + 2).min(last)];
    let u2 = u * u;
    let u3 = u2 * u;
    (p1 * 2.0
        + (p2 - p0) * u
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * u2
        + (p3 - p0 + (p1 - p2) * 3.0) * u3)
        * 0.5
}

fn emit_strip(
    path: &[usize],
    mesh: &Mesh,
    cells: &ViewCells,
    sea_level_m: f32,
    ramp: &dyn Fn(f32) -> f32,
    verts: &mut Vec<RiverVertex>,
) {
    // Consecutive duplicates give the spline zero-length tangents (NaN after
    // normalisation) — a cycle in corrupt drainage data can repeat a cell.
    // Nodes are filtered TOGETHER with their attributes so the spline and
    // the per-node blends stay in lockstep.
    let mut pts: Vec<Vec3> = Vec::with_capacity(path.len());
    let mut node_elev: Vec<f32> = Vec::with_capacity(path.len());
    let mut node_t: Vec<f32> = Vec::with_capacity(path.len());
    for &c in path {
        let p = mesh.centers[c];
        if pts.last().is_none_or(|l| (*l - p).length_squared() > 1e-12) {
            pts.push(p);
            // Water-surface elevation: mouths meet the sea at the waterline,
            // and a final lake cell meets the LAKE surface (bed + depth)
            // instead of diving under it.
            node_elev.push(
                (cells.elevation_m[c] + cells.lake_depth_m.get(c).copied().unwrap_or(0.0))
                    .max(sea_level_m),
            );
            node_t.push(ramp(cells.water_flux_m3_yr[c]));
        }
    }
    if pts.len() < 2 {
        return;
    }

    let steps = (pts.len() - 1) * SUBDIV;
    let mut prev_pair: Option<(RiverVertex, RiverVertex)> = None;
    let mut prev_p: Option<Vec3> = None;
    // No emitted triangle may span more than about a cell: whatever upstream
    // data or spline degeneracy produces a jump, a strip that fans across
    // half an ocean must be structurally impossible.
    let max_step = (4.0 * std::f32::consts::PI / mesh.n_cells() as f32).sqrt() * 1.5;
    for s in 0..=steps {
        let t = s as f32 / SUBDIV as f32;
        let p = spline_point(&pts, t).normalize_or_zero();
        // Inverted guard on purpose: a NaN from a degenerate spline compares
        // FALSE to everything, so `dist > max_step` let garbage through.
        // Only a provably-finite, provably-short step may continue a strip.
        let ok = p.is_finite()
            && p != Vec3::ZERO
            && prev_p.is_none_or(|pp| (p - pp).length() <= max_step);
        if !ok {
            prev_pair = None;
            prev_p = if p.is_finite() && p != Vec3::ZERO {
                Some(p)
            } else {
                None
            };
            continue;
        }
        prev_p = Some(p);
        // Forward direction along the curve, projected to the tangent plane.
        let ahead = spline_point(&pts, (t + 0.25).min((pts.len() - 1) as f32));
        let behind = spline_point(&pts, (t - 0.25).max(0.0));
        let fwd = {
            let d = ahead - behind;
            (d - p * d.dot(p)).normalize_or_zero()
        };
        if fwd == Vec3::ZERO {
            // A stale pair here would bridge to the next valid sample with a
            // long garbage triangle.
            prev_pair = None;
            continue;
        }
        let side = p.cross(fwd).normalize_or_zero();
        // Linear blend of the node attributes.
        let (i0, frac) = ((t.floor() as usize).min(pts.len() - 2), t.fract());
        let tt = node_t[i0] * (1.0 - frac) + node_t[i0 + 1] * frac;
        let elev = node_elev[i0] * (1.0 - frac) + node_elev[i0 + 1] * frac;
        let half_w = MIN_HALF_WIDTH_KM + (MAX_HALF_WIDTH_KM - MIN_HALF_WIDTH_KM) * tt;
        let off = side * (half_w / iw_mesh::EARTH_RADIUS_KM);
        let color = [RIVER_RGB[0], RIVER_RGB[1], RIVER_RGB[2], 0.5 + 0.4 * tt];
        let l = RiverVertex {
            pos: (p - off).normalize().to_array(),
            elevation_m: elev,
            color,
        };
        let r = RiverVertex {
            pos: (p + off).normalize().to_array(),
            elevation_m: elev,
            color,
        };
        if let Some((pl, pr)) = prev_pair {
            verts.extend_from_slice(&[pl, r, pr, pl, l, r]);
        }
        prev_pair = Some((l, r));
    }
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
        let tag = std::env::var("IW_PHASE_TAG").unwrap_or_else(|_| "phase-recent_past".into());
        let store = iw_store_postcard::FileStore::new(dir).unwrap();
        let planet = iw_core::CheckpointStore::load(&store, &tag).unwrap();
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
