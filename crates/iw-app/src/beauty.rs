//! The beauty view's CPU half (WP11, DESIGN.md §9): the Blue-Marble albedo
//! palette, the per-cell relief gradients the lit shader needs, and the
//! procedural cloud coverage field.
//!
//! This module is the single home for the CPU-side artist knobs. Its GPU
//! counterparts are `shaders/beauty.glsl` (colours, exponents, strengths used
//! inside a shader) and `iw_render_vulkan::beauty` (sun geometry, altitude
//! fades). Nothing here touches the GPU, so all of it is unit tested.
//!
//! Reference target: NASA's Blue Marble — deep saturated ocean blue with
//! turquoise shelves, land coloured by land cover, bare rock above the
//! treeline, white ice with a blue cast at thin margins.

use glam::Vec3;
use iw_core::Biome;
use iw_mesh::{Mesh, EARTH_RADIUS_M};
use iw_render_vulkan::{CellShade, SurfaceKind};

// --- water -----------------------------------------------------------------

/// Depth (m below sea level) at which the abyssal colour is reached.
pub const OCEAN_DEEP_M: f32 = 6_000.0;
/// Shape of the depth ramp. Below 1 the shelf colours are compressed into the
/// genuinely shallow water, as they are on the real thing.
pub const OCEAN_DEPTH_CURVE: f32 = 0.45;
/// Ramp position where the shelf turquoise has fully given way to open blue.
pub const OCEAN_SHELF_END: f32 = 0.30;
/// Shallow shelf water.
pub const OCEAN_SHELF_RGB: [u8; 3] = [0x2c, 0x80, 0x9c];
/// Open ocean.
pub const OCEAN_MID_RGB: [u8; 3] = [0x10, 0x48, 0x86];
/// Abyssal plain.
pub const OCEAN_DEEP_RGB: [u8; 3] = [0x07, 0x1e, 0x4e];
/// Lake depth (m) at which the deep-lake colour is reached.
pub const LAKE_DEEP_M: f32 = 120.0;
/// Lake water is lighter and greener than the sea.
pub const LAKE_SHALLOW_RGB: [u8; 3] = [0x46, 0x92, 0xb4];
/// See [`LAKE_SHALLOW_RGB`].
pub const LAKE_DEEP_RGB: [u8; 3] = [0x1b, 0x53, 0x92];
/// Lake depth (m) below which a cell is not treated as water at all.
pub const LAKE_MIN_M: f32 = 1.0;

// --- ice -------------------------------------------------------------------

/// Ice thickness (m) at which a cell is drawn as fully glaciated.
pub const ICE_FULL_M: f32 = 500.0;
/// Ice thickness (m) below which ice is ignored.
pub const ICE_MIN_M: f32 = 5.0;
/// Thin ice at a glacier's margin: blue, because that is what shadowed ice and
/// the water under it look like from orbit.
pub const ICE_MARGIN_RGB: [u8; 3] = [0xa4, 0xc2, 0xd8];
/// Thick ice: near-white with the faintest blue left in it.
pub const ICE_CORE_RGB: [u8; 3] = [0xf6, 0xfa, 0xff];

// --- land ------------------------------------------------------------------

/// Elevation (m) where bare rock starts to show through the vegetation.
pub const ROCK_START_M: f32 = 2_200.0;
/// Elevation (m) where the rock blend saturates.
pub const ROCK_FULL_M: f32 = 4_200.0;
/// How much rock is allowed to take over at [`ROCK_FULL_M`].
pub const ROCK_MAX_BLEND: f32 = 0.80;
/// Exposed high-altitude rock.
pub const ROCK_RGB: [u8; 3] = [0x8c, 0x82, 0x74];
/// Annual mean temperature (C) at which permanent snow starts to show.
pub const SNOW_START_C: f32 = -2.0;
/// Annual mean temperature (C) at which the land is snow covered.
pub const SNOW_FULL_C: f32 = -9.0;
/// How much snow is allowed to take over at [`SNOW_FULL_C`].
pub const SNOW_MAX_BLEND: f32 = 0.85;
/// Permanent snow (slightly warmer than glacier ice).
pub const SNOW_RGB: [u8; 3] = [0xef, 0xf3, 0xf6];
/// Land tint used before the biome process has classified anything.
pub const RAW_LOW_RGB: [u8; 3] = [0x6b, 0x7a, 0x52];
/// See [`RAW_LOW_RGB`].
pub const RAW_HIGH_RGB: [u8; 3] = [0x9c, 0x8f, 0x78];

// --- clouds ----------------------------------------------------------------

/// Base cloud coverage of a cell with no precipitation and average noise.
pub const CLOUD_BASE: f32 = 0.22;
/// How far the noise field swings the coverage either way.
pub const CLOUD_NOISE_GAIN: f32 = 0.55;
/// How much the precipitation field adds on top.
pub const CLOUD_PRECIP_GAIN: f32 = 0.40;
/// Precipitation (mm/yr) at which the precipitation bias saturates.
pub const CLOUD_PRECIP_FULL_MM_YR: f32 = 2_500.0;
/// Octaves of the coverage noise, and its base frequency in cycles per radius.
pub const CLOUD_OCTAVES: u32 = 4;
/// See [`CLOUD_OCTAVES`].
pub const CLOUD_FREQUENCY: f32 = 3.5;
/// Neighbour smoothing passes applied to the coverage field. Two is enough to
/// take the cell structure out of it without erasing the weather.
pub const CLOUD_SMOOTH_PASSES: u32 = 2;

// --- albedo ----------------------------------------------------------------

/// Everything the beauty palette reads about one cell. A history snapshot only
/// carries some of these; the rest keep their [`Default`].
#[derive(Debug, Clone, Copy)]
pub struct BeautyCell {
    pub elev_m: f32,
    pub sea_level_m: f32,
    pub ice_m: f32,
    pub lake_depth_m: f32,
    pub temperature_c: f32,
    pub biome: Biome,
}

impl Default for BeautyCell {
    fn default() -> Self {
        BeautyCell {
            elev_m: 0.0,
            sea_level_m: 0.0,
            ice_m: 0.0,
            lake_depth_m: 0.0,
            // Neutral: warm enough that nothing gets a snow blend it did not
            // ask for.
            temperature_c: 15.0,
            biome: Biome::Unclassified,
        }
    }
}

fn lerp(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let mut out = [0u8; 3];
    for i in 0..3 {
        out[i] = (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round() as u8;
    }
    out
}

fn rgba(c: [u8; 3]) -> [u8; 4] {
    [c[0], c[1], c[2], 255]
}

/// Smoothstep between two edges, in either direction.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Position on the ocean depth ramp: 0 at the shoreline, 1 at
/// [`OCEAN_DEEP_M`]. Monotone in depth.
pub fn ocean_depth_t(depth_m: f32) -> f32 {
    (depth_m.max(0.0) / OCEAN_DEEP_M)
        .clamp(0.0, 1.0)
        .powf(OCEAN_DEPTH_CURVE)
}

/// Sea colour at a given depth below sea level.
pub fn ocean_color(depth_m: f32) -> [u8; 3] {
    let t = ocean_depth_t(depth_m);
    if t < OCEAN_SHELF_END {
        lerp(OCEAN_SHELF_RGB, OCEAN_MID_RGB, t / OCEAN_SHELF_END)
    } else {
        lerp(
            OCEAN_MID_RGB,
            OCEAN_DEEP_RGB,
            (t - OCEAN_SHELF_END) / (1.0 - OCEAN_SHELF_END),
        )
    }
}

/// Land colour: biome land cover, with bare rock above the treeline and
/// permanent snow where the annual mean is far enough below freezing.
pub fn land_color(cell: &BeautyCell) -> [u8; 3] {
    let rel = cell.elev_m - cell.sea_level_m;
    let base = match cell.biome {
        // The biome process has not run yet (or the cell is flooded in the
        // data but dry here): fall back to a height tint so the early globe is
        // still readable.
        Biome::Unclassified | Biome::Ocean | Biome::Lake => {
            lerp(RAW_LOW_RGB, RAW_HIGH_RGB, (rel / 4_000.0).clamp(0.0, 1.0))
        }
        b => iw_biomes::biome_color(b),
    };
    let rock = smoothstep(ROCK_START_M, ROCK_FULL_M, rel) * ROCK_MAX_BLEND;
    let snow = smoothstep(SNOW_START_C, SNOW_FULL_C, cell.temperature_c) * SNOW_MAX_BLEND;
    lerp(lerp(base, ROCK_RGB, rock), SNOW_RGB, snow)
}

/// The beauty albedo of one cell: ice, then sea, then lake, then land.
///
/// This is albedo only — the lighting, glint and limb are the shader's job, so
/// the same colour is used at noon and at the terminator.
pub fn beauty_color(cell: &BeautyCell) -> [u8; 4] {
    if cell.ice_m >= ICE_MIN_M {
        let t = (cell.ice_m / ICE_FULL_M).clamp(0.0, 1.0).sqrt();
        return rgba(lerp(ICE_MARGIN_RGB, ICE_CORE_RGB, t));
    }
    let rel = cell.elev_m - cell.sea_level_m;
    if rel < 0.0 {
        return rgba(ocean_color(-rel));
    }
    if cell.lake_depth_m >= LAKE_MIN_M {
        let t = (cell.lake_depth_m / LAKE_DEEP_M).clamp(0.0, 1.0).sqrt();
        return rgba(lerp(LAKE_SHALLOW_RGB, LAKE_DEEP_RGB, t));
    }
    rgba(land_color(cell))
}

// --- shading ---------------------------------------------------------------

/// What the shader treats this cell as.
pub fn surface_kind(cell: &BeautyCell) -> SurfaceKind {
    if cell.elev_m < cell.sea_level_m {
        SurfaceKind::Ocean
    } else if cell.lake_depth_m >= LAKE_MIN_M {
        SurfaceKind::Lake
    } else {
        SurfaceKind::Land
    }
}

/// Elevation the beauty view displaces by: the sea surface is flat.
///
/// Bathymetry is in the colour, not in the geometry — a 30x exaggerated ocean
/// floor would put trenches in the planet's silhouette and sink the coastlines
/// into a bowl.
pub fn display_elevation(elev_m: f32, sea_level_m: f32) -> f32 {
    elev_m.max(sea_level_m)
}

/// Least-squares fit of the elevation gradient at `cell`, in that cell's own
/// (east, north) tangent basis, metres per metre.
///
/// The neighbours are projected into the tangent plane at arc-length scale and
/// a plane is fitted through their elevation differences; a degenerate system
/// (a cell with fewer than two independent neighbour directions) yields a zero
/// gradient rather than a division by zero.
pub fn elevation_gradient(mesh: &Mesh, elevation_m: &[f32], cell: u32) -> (f32, f32) {
    let c = mesh.centers[cell as usize];
    let (east, north) = mesh.east_north(cell);
    let e0 = elevation_m[cell as usize];
    let (mut sxx, mut sxy, mut syy, mut sxz, mut syz) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &n in mesh.neighbors_of(cell) {
        let p = mesh.centers[n as usize];
        // Tangential direction and arc length from cell centre to neighbour.
        let tangential = p - c * c.dot(p);
        let len = tangential.length();
        if len < 1e-9 {
            continue;
        }
        let dir = tangential / len;
        let arc_m = c.dot(p).clamp(-1.0, 1.0).acos() * EARTH_RADIUS_M as f32;
        let (x, y) = (
            (dir.dot(east) * arc_m) as f64,
            (dir.dot(north) * arc_m) as f64,
        );
        let dz = (elevation_m[n as usize] - e0) as f64;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxz += x * dz;
        syz += y * dz;
    }
    let det = sxx * syy - sxy * sxy;
    if det.abs() < 1e-6 {
        return (0.0, 0.0);
    }
    let ge = (sxz * syy - syz * sxy) / det;
    let gn = (syz * sxx - sxz * sxy) / det;
    (ge as f32, gn as f32)
}

/// Per-cell shading inputs and the elevation the beauty view displaces by.
///
/// Water cells get a flat surface and a zero gradient, so the sea reads as one
/// smooth sphere and the sun glint is not carved up by the sea floor.
pub fn shading(
    mesh: &Mesh,
    elev_m: &[f32],
    sea_level_m: f32,
    ice_m: &[f32],
    lake_depth_m: &[f32],
) -> (Vec<f32>, Vec<CellShade>) {
    use rayon::prelude::*;
    let n = elev_m.len();
    let display: Vec<f32> = elev_m
        .par_iter()
        .map(|e| display_elevation(*e, sea_level_m))
        .collect();
    let shade = (0..n)
        .into_par_iter()
        .map(|i| {
            let elev = elev_m[i];
            let cell = BeautyCell {
                elev_m: elev,
                sea_level_m,
                ice_m: ice_m.get(i).copied().unwrap_or(0.0),
                lake_depth_m: lake_depth_m.get(i).copied().unwrap_or(0.0),
                ..BeautyCell::default()
            };
            let kind = surface_kind(&cell);
            let (grad_east, grad_north) = if kind == SurfaceKind::Land {
                elevation_gradient(mesh, &display, i as u32)
            } else {
                (0.0, 0.0)
            };
            CellShade {
                grad_east,
                grad_north,
                kind,
                depth_t: if kind == SurfaceKind::Ocean {
                    ocean_depth_t(sea_level_m - elev)
                } else {
                    0.0
                },
                ice_t: (cell.ice_m / ICE_FULL_M).clamp(0.0, 1.0),
            }
        })
        .collect();
    (display, shade)
}

// --- clouds ----------------------------------------------------------------

/// Cloud coverage of one cell from its noise sample and its precipitation.
/// Always in 0..1 and non-decreasing in both inputs.
pub fn cloud_density(noise01: f32, precip_mm_yr: f32) -> f32 {
    let wet = (precip_mm_yr.max(0.0) / CLOUD_PRECIP_FULL_MM_YR)
        .clamp(0.0, 1.0)
        .sqrt();
    let n = noise01.clamp(0.0, 1.0) - 0.5;
    (CLOUD_BASE + CLOUD_NOISE_GAIN * n + CLOUD_PRECIP_GAIN * wet).clamp(0.0, 1.0)
}

/// Integer hash -> 0..1, the CPU counterpart of the shader's `hash13`.
fn hash01(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = seed
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add((x as u32).wrapping_mul(0x85eb_ca6b))
        .wrapping_add((y as u32).wrapping_mul(0xc2b2_ae35))
        .wrapping_add((z as u32).wrapping_mul(0x27d4_eb2f));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x2974_5c99);
    h ^= h >> 15;
    (h >> 8) as f32 / 16_777_216.0
}

/// Trilinear value noise in 0..1.
fn value_noise(p: Vec3, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let f = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let (ix, iy, iz) = (i.x as i32, i.y as i32, i.z as i32);
    let mut acc = 0.0;
    for (dz, wz) in [(0, 1.0 - f.z), (1, f.z)] {
        for (dy, wy) in [(0, 1.0 - f.y), (1, f.y)] {
            for (dx, wx) in [(0, 1.0 - f.x), (1, f.x)] {
                acc += wx * wy * wz * hash01(ix + dx, iy + dy, iz + dz, seed);
            }
        }
    }
    acc
}

/// Normalised fBm of [`value_noise`], in 0..1.
pub fn cloud_noise(dir: Vec3, seed: u32) -> f32 {
    let mut p = dir * CLOUD_FREQUENCY;
    let mut amp = 1.0f32;
    let (mut sum, mut norm) = (0.0f32, 0.0f32);
    for octave in 0..CLOUD_OCTAVES {
        sum += amp * value_noise(p, seed.wrapping_add(octave * 7919));
        norm += amp;
        amp *= 0.55;
        p *= 2.17;
    }
    (sum / norm).clamp(0.0, 1.0)
}

/// The per-cell cloud coverage field: seeded 3D noise on the sphere, biased
/// towards wet cells, then smoothed over the neighbour ring so no cell
/// boundary survives into the picture.
///
/// Static in the planet frame — the deck's motion is a rigid rotation applied
/// by the shader, not an advection of this field.
pub fn cloud_coverage(mesh: &Mesh, precip_mm_yr: &[f32], seed: u64) -> Vec<f32> {
    use rayon::prelude::*;
    let seed32 = (seed ^ (seed >> 32)) as u32;
    let mut field: Vec<f32> = (0..mesh.n_cells())
        .into_par_iter()
        .map(|i| {
            let noise = cloud_noise(mesh.centers[i], seed32);
            cloud_density(noise, precip_mm_yr.get(i).copied().unwrap_or(0.0))
        })
        .collect();
    for _ in 0..CLOUD_SMOOTH_PASSES {
        let src = field;
        field = (0..mesh.n_cells())
            .into_par_iter()
            .map(|i| {
                let ns = mesh.neighbors_of(i as u32);
                if ns.is_empty() {
                    return src[i];
                }
                let mean: f32 = ns.iter().map(|&n| src[n as usize]).sum::<f32>() / ns.len() as f32;
                0.5 * src[i] + 0.5 * mean
            })
            .collect();
    }
    field
}

/// Sample a per-cell field at the cloud shell's vertices.
pub fn sample_at_dirs(mesh: &Mesh, field: &[f32], dirs: &[Vec3]) -> Vec<f32> {
    use rayon::prelude::*;
    dirs.par_iter()
        .map(|d| field.get(mesh.cell_at(*d) as usize).copied().unwrap_or(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ocean_ramp_is_monotone_and_ends_where_it_should() {
        assert_eq!(ocean_depth_t(0.0), 0.0);
        assert_eq!(ocean_depth_t(OCEAN_DEEP_M), 1.0);
        assert_eq!(ocean_depth_t(50_000.0), 1.0, "must clamp");
        assert_eq!(ocean_depth_t(-10.0), 0.0, "must clamp");
        let mut prev_t = -1.0;
        let mut prev_lum = f32::MAX;
        for m in 0..=6_000 {
            let d = m as f32;
            let t = ocean_depth_t(d);
            assert!(t >= prev_t - 1e-6, "depth ramp not monotone at {d} m");
            prev_t = t;
            // Deeper water is never lighter.
            let c = ocean_color(d);
            let lum = 0.30 * c[0] as f32 + 0.59 * c[1] as f32 + 0.11 * c[2] as f32;
            assert!(lum <= prev_lum + 1e-3, "colour brightens at {d} m: {c:?}");
            prev_lum = lum;
        }
        let shelf = ocean_color(10.0);
        let abyss = ocean_color(6_000.0);
        assert!(
            shelf[1] > shelf[0] && shelf[2] > shelf[0],
            "shelf turquoise"
        );
        assert!(abyss[2] > abyss[1] && abyss[2] > abyss[0], "abyss navy");
        assert!(shelf[1] > abyss[1] + 80, "shelf is far lighter than abyss");
    }

    #[test]
    fn albedo_separates_ice_sea_lake_and_land() {
        let sea = beauty_color(&BeautyCell {
            elev_m: -4_000.0,
            ..Default::default()
        });
        let shelf = beauty_color(&BeautyCell {
            elev_m: -30.0,
            ..Default::default()
        });
        let lake = beauty_color(&BeautyCell {
            elev_m: 300.0,
            lake_depth_m: 40.0,
            ..Default::default()
        });
        let land = beauty_color(&BeautyCell {
            elev_m: 300.0,
            biome: Biome::TemperateBroadleaf,
            ..Default::default()
        });
        let ice = beauty_color(&BeautyCell {
            elev_m: 300.0,
            ice_m: 2_000.0,
            biome: Biome::IceSheet,
            ..Default::default()
        });
        let thin_ice = beauty_color(&BeautyCell {
            elev_m: 300.0,
            ice_m: 20.0,
            ..Default::default()
        });
        assert!(sea[2] > sea[1] && sea[2] > sea[0], "deep sea is blue");
        assert!(shelf[1] > sea[1], "shelf is lighter than the abyss");
        assert!(
            land[1] > land[2] && land[1] > land[0],
            "vegetated land is green"
        );
        assert!(lake[2] > lake[0], "lake is blue");
        assert!(
            lake[1] > sea[1] && lake[2] > sea[2],
            "lakes are lighter than the open sea: {lake:?} vs {sea:?}"
        );
        assert!(ice.iter().take(3).all(|c| *c > 200), "thick ice is white");
        assert!(
            thin_ice[2] > thin_ice[0],
            "a thin margin keeps a blue cast: {thin_ice:?}"
        );
        assert!(thin_ice[0] < ice[0], "margins are darker than the ice core");
        for c in [sea, shelf, lake, land, ice, thin_ice] {
            assert_eq!(c[3], 255, "beauty colours are opaque");
        }
    }

    #[test]
    fn high_ground_turns_to_rock_and_cold_ground_to_snow() {
        let low = land_color(&BeautyCell {
            elev_m: 200.0,
            biome: Biome::TemperateConifer,
            ..Default::default()
        });
        let high = land_color(&BeautyCell {
            elev_m: 4_500.0,
            biome: Biome::TemperateConifer,
            ..Default::default()
        });
        // Rock is browner and lighter than conifer forest.
        assert!(high[0] > low[0] + 30, "{high:?} vs {low:?}");
        let warm = land_color(&BeautyCell {
            elev_m: 200.0,
            temperature_c: 10.0,
            biome: Biome::Tundra,
            ..Default::default()
        });
        let cold = land_color(&BeautyCell {
            elev_m: 200.0,
            temperature_c: -20.0,
            biome: Biome::Tundra,
            ..Default::default()
        });
        assert!(cold.iter().all(|c| *c > 200), "deep cold is snow: {cold:?}");
        assert!(cold[0] > warm[0] && cold[2] > warm[2]);
        // The blend is continuous: no step at the ramp ends.
        let just_warm = land_color(&BeautyCell {
            elev_m: 200.0,
            temperature_c: SNOW_START_C + 0.01,
            biome: Biome::Tundra,
            ..Default::default()
        });
        assert_eq!(just_warm, warm);
    }

    /// A field that varies linearly along one axis must come back out of the
    /// fit as a gradient pointing that way, with the right magnitude.
    #[test]
    fn gradient_fit_recovers_a_synthetic_slope() {
        let mesh = Mesh::build(4);
        // f(p) = k * (p . axis); on the sphere its tangential gradient is
        // k * axis_tangential / R (metres per metre).
        let axis = Vec3::new(0.3, -0.5, 0.81).normalize();
        let k = 5_000.0f32;
        let elevation: Vec<f32> = mesh.centers.iter().map(|c| k * c.dot(axis)).collect();
        let mut checked = 0;
        for cell in (0..mesh.n_cells() as u32).step_by(37) {
            let c = mesh.centers[cell as usize];
            let (east, north) = mesh.east_north(cell);
            let tangential = axis - c * c.dot(axis);
            if tangential.length() < 0.2 {
                continue; // near the extremum the gradient is degenerate
            }
            let expected = (
                k * tangential.dot(east) / EARTH_RADIUS_M as f32,
                k * tangential.dot(north) / EARTH_RADIUS_M as f32,
            );
            let (ge, gn) = elevation_gradient(&mesh, &elevation, cell);
            let want = Vec3::new(expected.0, expected.1, 0.0);
            let got = Vec3::new(ge, gn, 0.0);
            let cos = want.normalize().dot(got.normalize());
            assert!(cos > 0.999, "cell {cell}: direction off, cos = {cos}");
            let rel = (got.length() - want.length()).abs() / want.length();
            assert!(rel < 0.02, "cell {cell}: magnitude off by {rel}");
            checked += 1;
        }
        assert!(checked > 20, "only checked {checked} cells");
    }

    #[test]
    fn flat_ground_and_isolated_cells_have_no_gradient() {
        let mesh = Mesh::build(3);
        let flat = vec![1_234.0f32; mesh.n_cells()];
        for cell in [0u32, 5, 42] {
            assert_eq!(elevation_gradient(&mesh, &flat, cell), (0.0, 0.0));
        }
    }

    #[test]
    fn shading_flattens_water_and_keeps_land_relief() {
        let mesh = Mesh::build(3);
        let n = mesh.n_cells();
        let axis = Vec3::Z;
        // Northern hemisphere land rising towards the pole, southern ocean.
        let elev: Vec<f32> = mesh
            .centers
            .iter()
            .map(|c| if c.z > 0.0 { 4_000.0 * c.z } else { -5_000.0 })
            .collect();
        let ice = vec![0.0; n];
        let lake = vec![0.0; n];
        let (display, shade) = shading(&mesh, &elev, 0.0, &ice, &lake);
        assert_eq!(display.len(), n);
        for i in 0..n {
            if elev[i] < 0.0 {
                assert_eq!(display[i], 0.0, "sea surface is flat");
                assert_eq!(shade[i].kind, SurfaceKind::Ocean);
                assert_eq!((shade[i].grad_east, shade[i].grad_north), (0.0, 0.0));
                assert!(shade[i].depth_t > 0.0);
            } else {
                assert_eq!(display[i], elev[i]);
                assert_eq!(shade[i].kind, SurfaceKind::Land);
            }
        }
        // Somewhere on the northern slope the gradient must point north.
        let sloped = (0..n)
            .filter(|&i| mesh.centers[i].z > 0.3 && mesh.centers[i].z < 0.9)
            .max_by(|a, b| {
                let ga = shade[*a].grad_north.abs();
                let gb = shade[*b].grad_north.abs();
                ga.partial_cmp(&gb).unwrap()
            })
            .unwrap();
        assert!(
            shade[sloped].grad_north > 0.0,
            "elevation rises towards the pole: {:?}",
            shade[sloped]
        );
        let _ = axis;
    }

    #[test]
    fn cloud_density_stays_in_range_and_follows_precipitation() {
        for n in [-1.0f32, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0] {
            for p in [-100.0f32, 0.0, 250.0, 1_000.0, 2_500.0, 9_000.0] {
                let d = cloud_density(n, p);
                assert!((0.0..=1.0).contains(&d), "n={n} p={p} -> {d}");
            }
        }
        // Non-decreasing in precipitation, strictly increasing over the ramp.
        let dry = cloud_density(0.5, 0.0);
        let wet = cloud_density(0.5, 2_500.0);
        assert!(wet > dry + 0.2, "{wet} vs {dry}");
        let mut prev = -1.0;
        for mm in (0..3_000).step_by(50) {
            let d = cloud_density(0.5, mm as f32);
            assert!(d >= prev - 1e-6);
            prev = d;
        }
        // And non-decreasing in the noise input.
        let mut prev = -1.0;
        for i in 0..=100 {
            let d = cloud_density(i as f32 / 100.0, 500.0);
            assert!(d >= prev - 1e-6);
            prev = d;
        }
    }

    #[test]
    fn cloud_field_is_deterministic_smooth_and_precipitation_biased() {
        let mesh = Mesh::build(3);
        let n = mesh.n_cells();
        let dry = vec![0.0f32; n];
        let wet = vec![3_000.0f32; n];
        let a = cloud_coverage(&mesh, &dry, 7);
        let b = cloud_coverage(&mesh, &dry, 7);
        assert_eq!(a, b, "same seed must give the same weather");
        let c = cloud_coverage(&mesh, &dry, 8);
        assert!(a != c, "a different seed must give different weather");
        let w = cloud_coverage(&mesh, &wet, 7);
        let mean = |v: &Vec<f32>| v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean(&w) > mean(&a) + 0.2, "wet planets are cloudier");
        for v in a.iter().chain(w.iter()) {
            assert!((0.0..=1.0).contains(v), "coverage out of range: {v}");
        }
        // The noise must actually vary across the sphere.
        let spread =
            a.iter().cloned().fold(f32::MIN, f32::max) - a.iter().cloned().fold(f32::MAX, f32::min);
        assert!(spread > 0.15, "cloud field is too flat: {spread}");
    }

    #[test]
    fn shell_sampling_reads_the_owning_cell() {
        let mesh = Mesh::build(3);
        let field: Vec<f32> = (0..mesh.n_cells()).map(|i| i as f32).collect();
        let dirs: Vec<Vec3> = mesh.centers.iter().step_by(11).copied().collect();
        let got = sample_at_dirs(&mesh, &field, &dirs);
        for (k, v) in got.iter().enumerate() {
            assert_eq!(*v, (k * 11) as f32);
        }
    }
}
