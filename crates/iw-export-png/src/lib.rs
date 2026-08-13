//! Rasterized map export for Infinite Worlds (DESIGN.md §12): equirectangular
//! and Mercator PNGs of elevation, biome and top-rock-type.
//!
//! Every pixel is resolved independently through [`iw_mesh::Mesh::cell_at`]
//! (2x2 sub-pixel supersampling, colors averaged), so output is deterministic
//! and does not depend on thread count or scheduling.

#![warn(missing_docs)]

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};
use std::path::Path;

use anyhow::Context;
use glam::Vec3;
use image::RgbImage;
use iw_core::{Biome, MapExporter, Planet, RockType};
use iw_mesh::Mesh;
use rayon::prelude::*;

/// Which per-cell field a map colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapKind {
    /// Hypsometric tint keyed off `elevation_m` relative to `sea_level_m`.
    Elevation,
    /// Fixed palette keyed off `biome`.
    Biome,
    /// Fixed categorical palette keyed off the topmost rock in each column.
    TopRock,
    /// Log-scaled precipitation ramp (diagnostic layer for climate tuning);
    /// ocean cells are darkened so the land pattern reads.
    Precip,
}

/// Sub-pixel sample offsets (fraction of a pixel) for 2x2 supersampling.
const SUBPIXEL_OFFSETS: [(f32, f32); 4] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];

/// Render `what` as an equirectangular (2:1) projection, `width` x
/// `width / 2` pixels, and write it as a PNG to `path`.
pub fn export_equirect(
    planet: &Planet,
    mesh: &Mesh,
    what: MapKind,
    path: &Path,
    width: u32,
) -> anyhow::Result<()> {
    let height = (width / 2).max(1);
    let wf = width as f32;
    let hf = height as f32;
    let img = render(planet, mesh, what, width, height, move |px, py| {
        let lon = (px / wf) * TAU - PI;
        let lat = FRAC_PI_2 - (py / hf) * PI;
        (lat, lon)
    });
    img.save(path)
        .with_context(|| format!("writing {}", path.display()))
}

/// Render `what` as a Mercator projection clamped to ±85° latitude
/// (`y = ln(tan(pi/4 + lat/2))`), square-ish (`width` x `width`), and write it
/// as a PNG to `path`.
pub fn export_mercator(
    planet: &Planet,
    mesh: &Mesh,
    what: MapKind,
    path: &Path,
    width: u32,
) -> anyhow::Result<()> {
    let height = width;
    let wf = width as f32;
    let hf = height as f32;
    let lat_clamp = 85f32.to_radians();
    let y_max = (FRAC_PI_4 + lat_clamp / 2.0).tan().ln();
    let img = render(planet, mesh, what, width, height, move |px, py| {
        let lon = (px / wf) * TAU - PI;
        let t = py / hf;
        let y_merc = y_max - t * (2.0 * y_max);
        let lat = (2.0 * y_merc.exp().atan() - FRAC_PI_2).clamp(-lat_clamp, lat_clamp);
        (lat, lon)
    });
    img.save(path)
        .with_context(|| format!("writing {}", path.display()))
}

/// Shared rasterizer: for every pixel, average [`SUBPIXEL_OFFSETS`] samples
/// resolved through `to_latlon(pixel_x, pixel_y) -> (lat_rad, lon_rad)`.
fn render<F>(
    planet: &Planet,
    mesh: &Mesh,
    what: MapKind,
    width: u32,
    height: u32,
    to_latlon: F,
) -> RgbImage
where
    F: Fn(f32, f32) -> (f32, f32) + Sync,
{
    let stride = width as usize * 3;
    let mut buf = vec![0u8; stride * height as usize];
    buf.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
        for x in 0..width {
            let mut acc = [0u32; 3];
            for &(sx, sy) in &SUBPIXEL_OFFSETS {
                let (lat, lon) = to_latlon(x as f32 + sx, y as f32 + sy);
                let dir = latlon_to_dir(lat, lon);
                let cell = mesh.cell_at(dir);
                let c = color_for(what, planet, cell);
                acc[0] += c[0] as u32;
                acc[1] += c[1] as u32;
                acc[2] += c[2] as u32;
            }
            let n = SUBPIXEL_OFFSETS.len() as u32;
            let idx = x as usize * 3;
            row[idx] = (acc[0] / n) as u8;
            row[idx + 1] = (acc[1] / n) as u8;
            row[idx + 2] = (acc[2] / n) as u8;
        }
    });
    RgbImage::from_raw(width, height, buf).expect("buffer sized to width * height * 3")
}

#[inline]
fn latlon_to_dir(lat: f32, lon: f32) -> Vec3 {
    Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
}

fn color_for(what: MapKind, planet: &Planet, cell: u32) -> [u8; 3] {
    match what {
        MapKind::Elevation => {
            elevation_color(planet.elevation_m[cell as usize], planet.sea_level_m)
        }
        MapKind::Biome => biome_color(planet.biome[cell as usize]),
        MapKind::TopRock => rock_color(planet.columns.top_rock(cell)),
        MapKind::Precip => precip_color(
            planet.precip_mm_yr[cell as usize],
            planet.elevation_m[cell as usize] >= planet.sea_level_m,
        ),
    }
}

/// Diagnostic precipitation ramp: log scale from 30 mm/yr (deep desert,
/// near-black red) through yellow (~250, the desert threshold) and green
/// (~750, Earth's land mean) to blue (3000+). Ocean is shown dimmed.
pub fn precip_color(mm_yr: f32, is_land: bool) -> [u8; 3] {
    let t = ((mm_yr.max(1.0) / 30.0).ln() / (3000.0f32 / 30.0).ln()).clamp(0.0, 1.0);
    let c = if t < 0.33 {
        lerp_color([0x40, 0x10, 0x08], [0xe8, 0xd4, 0x30], t / 0.33)
    } else if t < 0.66 {
        lerp_color([0xe8, 0xd4, 0x30], [0x2f, 0x9e, 0x44], (t - 0.33) / 0.33)
    } else {
        lerp_color([0x2f, 0x9e, 0x44], [0x1c, 0x4f, 0xd6], (t - 0.66) / 0.34)
    };
    if is_land {
        c
    } else {
        [c[0] / 3, c[1] / 3, c[2] / 3]
    }
}

const OCEAN_DEEP: [u8; 3] = [0x0a, 0x1a, 0x3f];
const OCEAN_COAST: [u8; 3] = [0x2e, 0x6b, 0xb8];
const LAND_LOW: [u8; 3] = [0x2d, 0x6a, 0x34];
const LAND_MID: [u8; 3] = [0xc9, 0xb0, 0x77];
const LAND_HIGH: [u8; 3] = [0x8b, 0x6b, 0x47];
const LAND_PEAK: [u8; 3] = [0xff, 0xff, 0xff];

/// Depth below sea level, in meters, at which the ocean ramp bottoms out at
/// [`OCEAN_DEEP`].
const OCEAN_RAMP_DEPTH_M: f32 = 6000.0;
/// Height above sea level, in meters, at which the land ramp reaches
/// [`LAND_PEAK`] (the notional snow line).
const LAND_RAMP_HEIGHT_M: f32 = 4500.0;

/// Position along the hypsometric ramp for `elev_m` relative to
/// `sea_level_m`: `-1.0` at `OCEAN_RAMP_DEPTH_M` or deeper, `0.0` at the
/// coastline, `1.0` at `LAND_RAMP_HEIGHT_M` or higher. Monotonically
/// non-decreasing in `elev_m` for a fixed `sea_level_m`.
pub fn hypsometric_t(elev_m: f32, sea_level_m: f32) -> f32 {
    let rel = elev_m - sea_level_m;
    if rel <= 0.0 {
        (rel / OCEAN_RAMP_DEPTH_M).clamp(-1.0, 0.0)
    } else {
        (rel / LAND_RAMP_HEIGHT_M).clamp(0.0, 1.0)
    }
}

/// Hypsometric tint: dark navy abyssal ocean -> coastal blue -> green
/// lowlands -> tan -> brown highlands -> white above the snow line.
pub fn elevation_color(elev_m: f32, sea_level_m: f32) -> [u8; 3] {
    let t = hypsometric_t(elev_m, sea_level_m);
    if t <= 0.0 {
        lerp_color(OCEAN_DEEP, OCEAN_COAST, t + 1.0)
    } else if t < 0.2 {
        lerp_color(LAND_LOW, LAND_MID, t / 0.2)
    } else if t < 0.6 {
        lerp_color(LAND_MID, LAND_HIGH, (t - 0.2) / 0.4)
    } else {
        lerp_color(LAND_HIGH, LAND_PEAK, (t - 0.6) / 0.4)
    }
}

fn lerp_color(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let mut out = [0u8; 3];
    for i in 0..3 {
        out[i] = (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round() as u8;
    }
    out
}

/// Satellite-ish palette for the 14 WWF biomes plus water/ice markers.
/// Self-contained (does not depend on `iw-biomes`, which owns a separate
/// beauty-view palette for the renderer).
pub fn biome_color(biome: Biome) -> [u8; 3] {
    match biome {
        Biome::Unclassified => [0x80, 0x80, 0x80],
        Biome::TropicalMoistBroadleaf => [0x1a, 0x4d, 0x2e],
        Biome::TropicalDryBroadleaf => [0x5c, 0x77, 0x2a],
        Biome::TropicalConifer => [0x2f, 0x5d, 0x3a],
        Biome::TemperateBroadleaf => [0x4c, 0x7a, 0x3d],
        Biome::TemperateConifer => [0x2e, 0x59, 0x39],
        Biome::BorealTaiga => [0x35, 0x56, 0x3f],
        Biome::TropicalGrassland => [0xb8, 0xa6, 0x4f],
        Biome::TemperateGrassland => [0xb5, 0xa6, 0x42],
        Biome::FloodedGrassland => [0x6b, 0x9e, 0x78],
        Biome::MontaneGrassland => [0x8f, 0xaa, 0x6e],
        Biome::Tundra => [0x9a, 0xa0, 0x8a],
        Biome::Mediterranean => [0x7d, 0x8c, 0x4a],
        Biome::Desert => [0xd9, 0xc2, 0x7e],
        Biome::Mangrove => [0x3d, 0x6b, 0x52],
        Biome::Ocean => [0x0b, 0x2a, 0x5e],
        Biome::Lake => [0x3f, 0x7f, 0xbf],
        Biome::IceSheet => [0xf0, 0xf5, 0xff],
    }
}

/// Distinct categorical palette for the 18 [`RockType`]s; gray when a column
/// has nothing deposited (bare mantle).
pub fn rock_color(top: Option<RockType>) -> [u8; 3] {
    let Some(rock) = top else {
        return [0x70, 0x70, 0x70];
    };
    match rock {
        RockType::Basalt => [0x4a, 0x4a, 0x4a],
        RockType::Gabbro => [0x3a, 0x4a, 0x3a],
        RockType::Granite => [0xe8, 0xa0, 0xa0],
        RockType::Diorite => [0xa0, 0x88, 0x8c],
        RockType::Andesite => [0x8c, 0x8c, 0x7a],
        RockType::Rhyolite => [0xd9, 0xb3, 0x8c],
        RockType::Tuff => [0xe0, 0xd9, 0xa0],
        RockType::Sandstone => [0xd9, 0xb2, 0x6b],
        RockType::Shale => [0x5c, 0x62, 0x70],
        RockType::Limestone => [0xe8, 0xe0, 0xc0],
        RockType::Conglomerate => [0xb3, 0x70, 0x3d],
        RockType::Evaporite => [0xf5, 0xf0, 0xe6],
        RockType::Slate => [0x6b, 0x7b, 0x8c],
        RockType::Schist => [0x8c, 0xa0, 0x90],
        RockType::Gneiss => [0xa8, 0x9b, 0xa0],
        RockType::Marble => [0xf0, 0xf0, 0xf0],
        RockType::Quartzite => [0xe8, 0xd8, 0xd0],
        RockType::Amphibolite => [0x2e, 0x3b, 0x2e],
    }
}

/// Equirectangular width used by [`PngExporter::export`].
const DEFAULT_EXPORT_WIDTH: u32 = 2048;

/// [`MapExporter`] that writes fixed-name equirectangular PNGs
/// (`elevation.png`, `biome.png`, `toprock.png`) into the given directory.
/// Use [`export_equirect`] / [`export_mercator`] directly for other
/// projections, kinds or sizes.
pub struct PngExporter;

impl MapExporter for PngExporter {
    fn export(&self, planet: &Planet, mesh: &Mesh, out_dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("creating {}", out_dir.display()))?;
        export_equirect(
            planet,
            mesh,
            MapKind::Elevation,
            &out_dir.join("elevation.png"),
            DEFAULT_EXPORT_WIDTH,
        )?;
        export_equirect(
            planet,
            mesh,
            MapKind::Biome,
            &out_dir.join("biome.png"),
            DEFAULT_EXPORT_WIDTH,
        )?;
        export_equirect(
            planet,
            mesh,
            MapKind::TopRock,
            &out_dir.join("toprock.png"),
            DEFAULT_EXPORT_WIDTH,
        )?;
        export_equirect(
            planet,
            mesh,
            MapKind::Precip,
            &out_dir.join("precip.png"),
            DEFAULT_EXPORT_WIDTH,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iw_core::PlanetConfig;

    fn synthetic_planet(mesh: &Mesh) -> Planet {
        let config = PlanetConfig {
            subdivision_level: mesh.level,
            ..PlanetConfig::default()
        };
        let mut planet = Planet::new(config, mesh.n_cells());
        planet.sea_level_m = 0.0;
        for i in 0..mesh.n_cells() {
            let lat = mesh.latlon[i][0];
            planet.elevation_m[i] = lat.sin() * 3000.0;
            planet.biome[i] = if planet.elevation_m[i] < 0.0 {
                Biome::Ocean
            } else {
                Biome::Desert
            };
            let rock = if i % 2 == 0 {
                RockType::Granite
            } else {
                RockType::Basalt
            };
            planet.columns.deposit(i as u32, rock, 10.0, 0.0);
        }
        planet
    }

    #[test]
    fn hypsometric_t_is_monotonic() {
        let sea = 0.0;
        let mut prev = f32::NEG_INFINITY;
        let mut e = -8000.0f32;
        while e <= 6000.0 {
            let t = hypsometric_t(e, sea);
            assert!(t >= prev, "t went backwards at elev {e}: {t} < {prev}");
            prev = t;
            e += 137.0;
        }
    }

    #[test]
    fn elevation_color_key_points() {
        assert_eq!(elevation_color(-10_000.0, 0.0), OCEAN_DEEP);
        assert_eq!(elevation_color(0.0, 0.0), OCEAN_COAST);
        assert_eq!(elevation_color(5_000.0, 0.0), LAND_PEAK);
    }

    #[test]
    fn equirect_dimensions_and_determinism() {
        let mesh = Mesh::build(3);
        let planet = synthetic_planet(&mesh);
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("a.png");
        let p2 = dir.path().join("b.png");
        export_equirect(&planet, &mesh, MapKind::Elevation, &p1, 64).unwrap();
        export_equirect(&planet, &mesh, MapKind::Elevation, &p2, 64).unwrap();

        let img1 = image::open(&p1).unwrap().to_rgb8();
        assert_eq!(img1.width(), 64);
        assert_eq!(img1.height(), 32);

        let b1 = std::fs::read(&p1).unwrap();
        let b2 = std::fs::read(&p2).unwrap();
        assert_eq!(b1, b2, "equirect export must be deterministic");
    }

    #[test]
    fn mercator_polar_cells_do_not_panic() {
        let mesh = Mesh::build(3);
        let planet = synthetic_planet(&mesh);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("merc.png");
        export_mercator(&planet, &mesh, MapKind::Biome, &p, 64).unwrap();
        let img = image::open(&p).unwrap().to_rgb8();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn map_exporter_writes_all_three() {
        let mesh = Mesh::build(3);
        let planet = synthetic_planet(&mesh);
        let dir = tempfile::tempdir().unwrap();
        PngExporter.export(&planet, &mesh, dir.path()).unwrap();
        assert!(dir.path().join("elevation.png").exists());
        assert!(dir.path().join("biome.png").exists());
        assert!(dir.path().join("toprock.png").exists());
    }
}
