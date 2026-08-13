//! Data layers: which per-cell field the globe is coloured by, and the pure
//! functions that turn that field into an RGBA8 colour.
//!
//! Everything here is CPU-side and GPU-free so it can be unit tested. The
//! renderer only ever sees the finished `[u8; 4]` array (see
//! `iw_render_vulkan::Renderer::update_cells`).
//!
//! Palette conventions:
//! * sequential physical fields (temperature, precipitation, ice, crust age)
//!   use a two- or three-stop ramp with fixed, documented end points, so the
//!   same value is the same colour in every planet and every frame;
//! * categorical fields (biome, top rock, crust type) reuse the palettes the
//!   PNG exporter and `iw-biomes` already own, so maps and globe agree;
//! * `Plates` derives a hue per plate id from the golden ratio, which keeps
//!   consecutive ids far apart on the colour wheel.

use iw_core::{Biome, CrustType, RockType, ViewCells};

/// Ice thickness (m) at which a cell is drawn as fully glaciated.
pub const ICE_FULL_M: f32 = 500.0;
/// Ice thickness (m) below which ice is ignored by the beauty layer.
pub const ICE_MIN_M: f32 = 5.0;
/// Ocean depth (m below sea level) at which the abyssal colour is reached.
pub const OCEAN_DEEP_M: f32 = 6_000.0;
/// Oceanic crust age (Myr) at which the "old crust" colour is reached.
pub const CRUST_AGE_MAX_MYR: f32 = 200.0;
/// Crust thickness (m) at which the "thick crust" lightness is reached.
pub const CRUST_THICK_MAX_M: f32 = 70_000.0;
/// Temperature ramp end points, degrees C.
pub const TEMP_MIN_C: f32 = -40.0;
/// See [`TEMP_MIN_C`].
pub const TEMP_MAX_C: f32 = 40.0;
/// Precipitation ramp end points, mm/yr.
pub const PRECIP_MAX_MM_YR: f32 = 4_000.0;
/// Unassigned plate marker: Phase-1 ocean has no plate yet.
pub const NO_PLATE: u16 = u16::MAX;

/// The switchable data layers (DESIGN.md §9). Order is the display order and
/// the number-key order (`1`..`9`, `0`; [`Layer::WaterFlux`] has no digit and
/// is bound to `-`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Placeholder for the WP11 beauty view: biome albedo on land, depth-shaded
    /// blues at sea, white ice.
    Beauty,
    /// Hypsometric tint (the PNG exporter's palette).
    Elevation,
    /// WWF biome palette.
    Biomes,
    /// One hue per plate id, plus velocity arrows drawn by the UI.
    Plates,
    /// Oceanic crust age.
    CrustAge,
    /// Crust type as hue, crust thickness as lightness.
    CrustType,
    /// Topmost rock of each stratigraphic column.
    TopRock,
    /// Annual mean surface temperature.
    Temperature,
    /// Annual precipitation.
    Precipitation,
    /// Ice sheet thickness.
    Ice,
    /// River discharge, log-scaled, over the beauty base.
    WaterFlux,
}

impl Layer {
    /// Every layer, in display order.
    pub const ALL: [Layer; 11] = [
        Layer::Beauty,
        Layer::Elevation,
        Layer::Biomes,
        Layer::Plates,
        Layer::CrustAge,
        Layer::CrustType,
        Layer::TopRock,
        Layer::Temperature,
        Layer::Precipitation,
        Layer::Ice,
        Layer::WaterFlux,
    ];

    /// Menu label.
    pub fn name(self) -> &'static str {
        match self {
            Layer::Beauty => "Beauty (placeholder)",
            Layer::Elevation => "Elevation",
            Layer::Biomes => "Biomes",
            Layer::Plates => "Plates + velocity",
            Layer::CrustAge => "Crust age",
            Layer::CrustType => "Crust type / thickness",
            Layer::TopRock => "Top rock",
            Layer::Temperature => "Temperature",
            Layer::Precipitation => "Precipitation",
            Layer::Ice => "Ice thickness",
            Layer::WaterFlux => "Water flux",
        }
    }

    /// One-line description of what the colours mean.
    pub fn legend(self) -> &'static str {
        match self {
            Layer::Beauty => "biome albedo on land, depth-shaded ocean, white ice",
            Layer::Elevation => "hypsometric: navy abyss -> green lowland -> white peak",
            Layer::Biomes => "WWF biome palette",
            Layer::Plates => "one hue per plate; grey = unassigned",
            Layer::CrustAge => "oceanic crust: red 0 Myr -> blue 200 Myr; grey = continental",
            Layer::CrustType => "blue = oceanic, orange = continental; lighter = thicker",
            Layer::TopRock => "categorical: topmost stratum of each column",
            Layer::Temperature => "-40 C blue -> +40 C red",
            Layer::Precipitation => "dry tan -> wet blue-green (0..4000 mm/yr)",
            Layer::Ice => "ice thickness, 0..500 m+ white",
            Layer::WaterFlux => "log discharge over the beauty base",
        }
    }

    /// Position in [`Layer::ALL`].
    pub fn index(self) -> usize {
        Layer::ALL.iter().position(|l| *l == self).unwrap_or(0)
    }

    /// Layer at `index`, wrapping out-of-range indices to [`Layer::Beauty`].
    pub fn from_index(index: usize) -> Layer {
        Layer::ALL.get(index).copied().unwrap_or(Layer::Beauty)
    }

    /// The next layer in display order (wraps).
    pub fn next(self) -> Layer {
        Layer::from_index((self.index() + 1) % Layer::ALL.len())
    }

    /// Whether a thin history snapshot (elevation, biome, plate id, ice) holds
    /// enough to draw this layer. Layers that fail this fall back to
    /// [`Layer::Elevation`] while scrubbing; see [`snapshot_layer`].
    pub fn in_thin_snapshot(self) -> bool {
        matches!(
            self,
            Layer::Beauty | Layer::Elevation | Layer::Biomes | Layer::Plates | Layer::Ice
        )
    }
}

/// Which layer a thin history snapshot can actually be drawn as: the requested
/// one, or [`Layer::Elevation`] when the snapshot lacks the data. The bool is
/// true when a substitution happened, so the UI can say so.
pub fn snapshot_layer(requested: Layer) -> (Layer, bool) {
    if requested.in_thin_snapshot() {
        (requested, false)
    } else {
        (Layer::Elevation, true)
    }
}

/// View-dependent normalisation, recomputed when a new snapshot arrives so a
/// planet whose rivers are still tiny is not drawn black.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerScales {
    /// log10 of the discharge (m^3/yr) where the flux ramp starts.
    pub flux_log_min: f32,
    /// log10 of the discharge (m^3/yr) where the flux ramp saturates.
    pub flux_log_max: f32,
}

impl Default for LayerScales {
    fn default() -> Self {
        LayerScales {
            flux_log_min: 9.0,
            flux_log_max: 13.0,
        }
    }
}

impl LayerScales {
    /// Derive the flux ramp from the data: the top of the ramp is the largest
    /// discharge present, the bottom is four decades below it. Both are
    /// quantised to whole decades so small changes between snapshots do not
    /// make the rivers shimmer.
    pub fn from_cells(cells: &ViewCells) -> LayerScales {
        let max = cells
            .water_flux_m3_yr
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        if max <= 0.0 {
            return LayerScales::default();
        }
        let log_max = max.log10().ceil().clamp(4.0, 20.0);
        LayerScales {
            flux_log_min: log_max - 4.0,
            flux_log_max: log_max,
        }
    }
}

// --- ramps -----------------------------------------------------------------

fn lerp(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let mut out = [0u8; 3];
    for i in 0..3 {
        out[i] = (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round() as u8;
    }
    out
}

#[inline]
fn rgba(c: [u8; 3]) -> [u8; 4] {
    [c[0], c[1], c[2], 255]
}

/// HSV -> RGB with s, v in 0..1 and hue in turns (0..1, wrapping).
pub fn hsv(hue_turns: f32, s: f32, v: f32) -> [u8; 3] {
    let h = (hue_turns.rem_euclid(1.0)) * 6.0;
    let i = h.floor();
    let f = h - i;
    let (s, v) = (s.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    ]
}

/// Beauty placeholder: biome albedo on land, depth-shaded blue at sea, white
/// where there is ice. WP11 replaces this with the lit Blue-Marble shader.
pub fn beauty_color(elev_m: f32, sea_level_m: f32, ice_m: f32, biome: Biome) -> [u8; 4] {
    if ice_m >= ICE_MIN_M {
        let t = (ice_m / ICE_FULL_M).clamp(0.0, 1.0);
        return rgba(lerp([0xd0, 0xe0, 0xe8], [0xf4, 0xfa, 0xff], t));
    }
    let rel = elev_m - sea_level_m;
    if rel < 0.0 {
        // Deep water dark navy, shelf turquoise.
        let t = (-rel / OCEAN_DEEP_M).clamp(0.0, 1.0);
        return rgba(lerp([0x2e, 0x86, 0xa8], [0x08, 0x1d, 0x4a], t));
    }
    match biome {
        Biome::Ocean | Biome::Lake => rgba([0x1d, 0x4e, 0x89]),
        // Before the biome process has classified anything, fall back to a
        // land tint keyed off height so the early globe is still readable.
        Biome::Unclassified => rgba(lerp(
            [0x6b, 0x7a, 0x52],
            [0x9c, 0x8f, 0x78],
            (rel / 4_000.0).clamp(0.0, 1.0),
        )),
        b => rgba(iw_biomes::biome_color(b)),
    }
}

/// Hypsometric elevation tint, shared with the PNG exporter.
pub fn elevation_color(elev_m: f32, sea_level_m: f32) -> [u8; 4] {
    rgba(iw_export_png::elevation_color(elev_m, sea_level_m))
}

/// Biome palette (the satellite-look one owned by `iw-biomes`).
pub fn biome_color(biome: Biome) -> [u8; 4] {
    rgba(iw_biomes::biome_color(biome))
}

/// One hue per plate, spaced by the golden ratio so consecutive ids are far
/// apart on the wheel. [`NO_PLATE`] (Phase-1 ocean) is neutral grey — the
/// caller must never use it to index `Planet::plates`.
pub fn plate_color(plate_id: u16) -> [u8; 4] {
    if plate_id == NO_PLATE {
        return rgba([0x55, 0x58, 0x5c]);
    }
    const GOLDEN: f32 = 0.618_034;
    let hue = (plate_id as f32 * GOLDEN).fract();
    // Vary saturation/value a little on a different cycle so neighbouring ids
    // differ even for a viewer with reduced colour vision.
    let s = 0.55 + 0.30 * ((plate_id % 3) as f32 / 2.0);
    let v = 0.95 - 0.20 * ((plate_id % 2) as f32);
    rgba(hsv(hue, s, v))
}

/// Oceanic crust age: hot red at the ridge through to cold blue abyssal
/// floor. Continental crust has no meaningful age here and is drawn grey.
pub fn crust_age_color(age_myr: f32, crust: CrustType) -> [u8; 4] {
    if crust == CrustType::Continental {
        return rgba([0x77, 0x72, 0x6a]);
    }
    let t = (age_myr / CRUST_AGE_MAX_MYR).clamp(0.0, 1.0);
    if t < 0.5 {
        rgba(lerp([0xd6, 0x2b, 0x1f], [0xe8, 0xd0, 0x54], t / 0.5))
    } else {
        rgba(lerp(
            [0xe8, 0xd0, 0x54],
            [0x14, 0x35, 0x8c],
            (t - 0.5) / 0.5,
        ))
    }
}

/// Crust type as hue (oceanic blue, continental orange), thickness as
/// lightness (thin dark, thick bright).
pub fn crust_type_color(crust: CrustType, thickness_m: f32) -> [u8; 4] {
    let t = (thickness_m / CRUST_THICK_MAX_M).clamp(0.0, 1.0);
    match crust {
        CrustType::Oceanic => rgba(lerp([0x0a, 0x1c, 0x3a], [0x6f, 0xb4, 0xe8], t.sqrt())),
        CrustType::Continental => rgba(lerp([0x3a, 0x24, 0x0a], [0xf2, 0xc9, 0x84], t.sqrt())),
    }
}

/// Categorical top-rock palette, shared with the PNG exporter.
pub fn rock_color(top: Option<RockType>) -> [u8; 4] {
    rgba(iw_export_png::rock_color(top))
}

/// Temperature ramp: [`TEMP_MIN_C`] blue, 0 C pale, [`TEMP_MAX_C`] red.
pub fn temperature_color(t_c: f32) -> [u8; 4] {
    let t = ((t_c - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C)).clamp(0.0, 1.0);
    if t < 0.5 {
        rgba(lerp([0x18, 0x33, 0xa0], [0xef, 0xef, 0xe0], t / 0.5))
    } else {
        rgba(lerp(
            [0xef, 0xef, 0xe0],
            [0xb3, 0x18, 0x14],
            (t - 0.5) / 0.5,
        ))
    }
}

/// Precipitation ramp: dry tan through to wet blue-green. Square-rooted so the
/// interesting 0..500 mm/yr range is not squashed into one shade.
pub fn precip_color(mm_yr: f32) -> [u8; 4] {
    let t = (mm_yr / PRECIP_MAX_MM_YR).clamp(0.0, 1.0).sqrt();
    if t < 0.5 {
        rgba(lerp([0xd9, 0xc0, 0x8a], [0x86, 0xbf, 0x8e], t / 0.5))
    } else {
        rgba(lerp(
            [0x86, 0xbf, 0x8e],
            [0x0d, 0x4f, 0x6b],
            (t - 0.5) / 0.5,
        ))
    }
}

/// Ice thickness over a dark base: bare ground stays dark, ice ramps to white.
pub fn ice_color(ice_m: f32, elev_m: f32, sea_level_m: f32) -> [u8; 4] {
    if ice_m < ICE_MIN_M {
        let base = if elev_m < sea_level_m {
            [0x12, 0x1c, 0x2e]
        } else {
            [0x33, 0x33, 0x30]
        };
        return rgba(base);
    }
    let t = (ice_m / (ICE_FULL_M * 4.0)).clamp(0.0, 1.0).sqrt();
    rgba(lerp([0x9d, 0xc4, 0xdc], [0xff, 0xff, 0xff], t))
}

/// Log position of a discharge on the flux ramp, or `None` below the ramp's
/// threshold (where the beauty base shows through). Monotone in `flux`.
pub fn flux_t(flux_m3_yr: f32, scales: LayerScales) -> Option<f32> {
    // NaN and non-positive discharge both mean "nothing to draw".
    if flux_m3_yr.is_nan() || flux_m3_yr <= 0.0 {
        return None;
    }
    let l = flux_m3_yr.log10();
    if l <= scales.flux_log_min {
        return None;
    }
    let span = (scales.flux_log_max - scales.flux_log_min).max(1e-6);
    Some(((l - scales.flux_log_min) / span).clamp(0.0, 1.0))
}

/// River discharge painted over the beauty base.
pub fn water_flux_color(
    flux_m3_yr: f32,
    elev_m: f32,
    sea_level_m: f32,
    ice_m: f32,
    biome: Biome,
    scales: LayerScales,
) -> [u8; 4] {
    let base = beauty_color(elev_m, sea_level_m, ice_m, biome);
    let Some(t) = flux_t(flux_m3_yr, scales) else {
        return base;
    };
    // Small streams blend in; big rivers reach full cyan-white.
    let river = lerp([0x2c, 0x7f, 0xd4], [0xd8, 0xf4, 0xff], t);
    let mix = 0.35 + 0.65 * t;
    rgba(lerp([base[0], base[1], base[2]], river, mix))
}

/// Colour one cell for `layer`.
pub fn cell_color(
    layer: Layer,
    cells: &ViewCells,
    sea_level_m: f32,
    cell: usize,
    scales: LayerScales,
) -> [u8; 4] {
    let elev = cells.elevation_m[cell];
    match layer {
        Layer::Beauty => beauty_color(
            elev,
            sea_level_m,
            cells.ice_thickness_m[cell],
            cells.biome[cell],
        ),
        Layer::Elevation => elevation_color(elev, sea_level_m),
        Layer::Biomes => biome_color(cells.biome[cell]),
        Layer::Plates => plate_color(cells.plate_id[cell]),
        Layer::CrustAge => crust_age_color(cells.crust_age_myr[cell], cells.crust_type[cell]),
        Layer::CrustType => crust_type_color(cells.crust_type[cell], cells.crust_thickness_m[cell]),
        Layer::TopRock => rock_color(cells.top_rock[cell]),
        Layer::Temperature => temperature_color(cells.temperature_c[cell]),
        Layer::Precipitation => precip_color(cells.precip_mm_yr[cell]),
        Layer::Ice => ice_color(cells.ice_thickness_m[cell], elev, sea_level_m),
        Layer::WaterFlux => water_flux_color(
            cells.water_flux_m3_yr[cell],
            elev,
            sea_level_m,
            cells.ice_thickness_m[cell],
            cells.biome[cell],
            scales,
        ),
    }
}

/// Colour every cell of a snapshot for `layer`.
pub fn cell_colors(
    layer: Layer,
    cells: &ViewCells,
    sea_level_m: f32,
    scales: LayerScales,
) -> Vec<[u8; 4]> {
    use rayon::prelude::*;
    let n = cells.elevation_m.len();
    (0..n)
        .into_par_iter()
        .map(|i| cell_color(layer, cells, sea_level_m, i, scales))
        .collect()
}

/// Colour every cell of a thin history snapshot, substituting the layer when
/// the snapshot lacks the data (see [`snapshot_layer`]).
pub fn snapshot_colors(
    layer: Layer,
    snap: &iw_store_postcard::HistorySnapshot,
) -> (Vec<[u8; 4]>, bool) {
    use rayon::prelude::*;
    let (layer, substituted) = snapshot_layer(layer);
    let sea = snap.sea_level_m;
    let colors = (0..snap.elevation_m.len())
        .into_par_iter()
        .map(|i| {
            let elev = snap.elevation_m[i];
            match layer {
                Layer::Beauty => beauty_color(elev, sea, snap.ice_thickness_m[i], snap.biome[i]),
                Layer::Biomes => biome_color(snap.biome[i]),
                Layer::Plates => plate_color(snap.plate_id[i]),
                Layer::Ice => ice_color(snap.ice_thickness_m[i], elev, sea),
                // Elevation, and every layer that fell back to it.
                _ => elevation_color(elev, sea),
            }
        })
        .collect();
    (colors, substituted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(n: usize) -> ViewCells {
        ViewCells {
            elevation_m: vec![0.0; n],
            sediment_m: vec![0.0; n],
            biome: vec![Biome::Ocean; n],
            plate_id: vec![0; n],
            crust_type: vec![CrustType::Oceanic; n],
            crust_age_myr: vec![0.0; n],
            crust_thickness_m: vec![7_000.0; n],
            temperature_c: vec![15.0; n],
            precip_mm_yr: vec![1_000.0; n],
            ice_thickness_m: vec![0.0; n],
            water_flux_m3_yr: vec![0.0; n],
            lake_depth_m: vec![0.0; n],
            top_rock: vec![None; n],
            plate_velocity_m_yr: vec![glam::Vec3::ZERO; n],
            tectonic_flags: vec![0; n],
        }
    }

    #[test]
    fn every_layer_is_deterministic_and_opaque() {
        let mut c = cells(8);
        for (i, e) in c.elevation_m.iter_mut().enumerate() {
            *e = i as f32 * 500.0 - 2000.0;
        }
        c.plate_id[3] = NO_PLATE;
        c.crust_type[4] = CrustType::Continental;
        c.top_rock[5] = Some(RockType::Granite);
        c.water_flux_m3_yr[6] = 1.0e12;
        c.ice_thickness_m[7] = 900.0;
        let scales = LayerScales::from_cells(&c);
        for layer in Layer::ALL {
            for i in 0..8 {
                let a = cell_color(layer, &c, 0.0, i, scales);
                let b = cell_color(layer, &c, 0.0, i, scales);
                assert_eq!(a, b, "{layer:?} cell {i} is not deterministic");
                assert_eq!(a[3], 255, "{layer:?} cell {i} must be opaque");
            }
        }
    }

    #[test]
    fn adjacent_plate_ids_get_distinct_hues() {
        // Every consecutive pair must differ by a visible margin, and the
        // unassigned marker must never collide with a real plate.
        let unassigned = plate_color(NO_PLATE);
        for id in 0..64u16 {
            let a = plate_color(id);
            let b = plate_color(id + 1);
            let d: i32 = (0..3)
                .map(|k| (a[k] as i32 - b[k] as i32).abs())
                .sum::<i32>();
            assert!(d > 60, "plates {id} and {} too close: {a:?} {b:?}", id + 1);
            let du: i32 = (0..3)
                .map(|k| (a[k] as i32 - unassigned[k] as i32).abs())
                .sum();
            assert!(du > 40, "plate {id} looks unassigned: {a:?}");
        }
    }

    #[test]
    fn plate_colors_are_stable_across_calls() {
        assert_eq!(plate_color(7), plate_color(7));
        assert_eq!(plate_color(NO_PLATE), plate_color(NO_PLATE));
    }

    #[test]
    fn flux_mapping_is_monotone_and_thresholded() {
        let s = LayerScales {
            flux_log_min: 9.0,
            flux_log_max: 13.0,
        };
        assert_eq!(flux_t(0.0, s), None);
        assert_eq!(flux_t(1.0e9, s), None, "at the threshold nothing shows");
        assert_eq!(flux_t(1.0e8, s), None);
        let mut prev = 0.0;
        let mut seen = 0;
        for e in 90..=140 {
            let f = 10f32.powf(e as f32 / 10.0);
            if let Some(t) = flux_t(f, s) {
                assert!(t >= prev - 1e-6, "flux ramp not monotone at 1e{e}");
                assert!((0.0..=1.0).contains(&t));
                prev = t;
                seen += 1;
            }
        }
        assert!(seen > 20);
        assert!((flux_t(1.0e13, s).unwrap() - 1.0).abs() < 1e-5);
        // Saturates rather than overflowing.
        assert!((flux_t(1.0e20, s).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn flux_scale_follows_the_data() {
        let mut c = cells(4);
        c.water_flux_m3_yr[2] = 5.0e11;
        let s = LayerScales::from_cells(&c);
        assert_eq!(s.flux_log_max, 12.0);
        assert_eq!(s.flux_log_min, 8.0);
        // The biggest river present is always visible.
        assert!(flux_t(5.0e11, s).is_some());
        // An all-dry planet keeps the default ramp instead of dividing by zero.
        let dry = LayerScales::from_cells(&cells(4));
        assert_eq!(dry, LayerScales::default());
    }

    #[test]
    fn temperature_and_precipitation_ramps_are_monotone_in_brightness() {
        // Cold end must be bluer than the hot end; hot end redder.
        let cold = temperature_color(-40.0);
        let hot = temperature_color(40.0);
        assert!(cold[2] > cold[0], "cold end should be blue: {cold:?}");
        assert!(hot[0] > hot[2], "hot end should be red: {hot:?}");
        assert_eq!(temperature_color(-100.0), cold, "must clamp");
        assert_eq!(temperature_color(100.0), hot, "must clamp");

        let dry = precip_color(0.0);
        let wet = precip_color(4_000.0);
        assert!(dry[0] > dry[2], "dry end should be tan: {dry:?}");
        assert!(wet[2] > wet[0], "wet end should be blue-green: {wet:?}");
        assert_eq!(precip_color(9_999.0), wet, "must clamp");
    }

    #[test]
    fn beauty_separates_land_sea_and_ice() {
        let sea = beauty_color(-3000.0, 0.0, 0.0, Biome::Ocean);
        let shelf = beauty_color(-50.0, 0.0, 0.0, Biome::Ocean);
        let land = beauty_color(500.0, 0.0, 0.0, Biome::TemperateBroadleaf);
        let ice = beauty_color(500.0, 0.0, 2_000.0, Biome::IceSheet);
        assert!(sea[2] > sea[1] && sea[2] > sea[0], "deep sea is blue");
        assert!(shelf[1] > sea[1], "shelf is lighter than the abyss");
        assert!(land[1] > land[2], "vegetated land is green");
        assert!(ice.iter().take(3).all(|c| *c > 200), "ice is near-white");
        // Sea level moves the shoreline, not just the palette.
        assert_eq!(beauty_color(100.0, 500.0, 0.0, Biome::Ocean)[3], 255);
        assert!(beauty_color(100.0, 500.0, 0.0, Biome::Ocean)[2] > 100);
    }

    #[test]
    fn crust_layers_separate_type_age_and_thickness() {
        let young = crust_age_color(0.0, CrustType::Oceanic);
        let old = crust_age_color(200.0, CrustType::Oceanic);
        assert!(young[0] > young[2], "young crust is red");
        assert!(old[2] > old[0], "old crust is blue");
        assert_eq!(
            crust_age_color(50.0, CrustType::Continental),
            crust_age_color(150.0, CrustType::Continental),
            "continental age is meaningless and must not be coloured by it"
        );

        let thin = crust_type_color(CrustType::Continental, 20_000.0);
        let thick = crust_type_color(CrustType::Continental, 70_000.0);
        assert!(
            thick[0] > thin[0] && thick[1] > thin[1],
            "thicker = lighter"
        );
        let ocean = crust_type_color(CrustType::Oceanic, 7_000.0);
        assert!(ocean[2] >= ocean[0], "oceanic hue is blue");
    }

    #[test]
    fn thin_snapshots_fall_back_to_elevation() {
        for layer in Layer::ALL {
            let (used, substituted) = snapshot_layer(layer);
            if layer.in_thin_snapshot() {
                assert_eq!(used, layer);
                assert!(!substituted);
            } else {
                assert_eq!(used, Layer::Elevation, "{layer:?} has no snapshot data");
                assert!(substituted);
            }
        }
        // The five layers a thin snapshot really can draw.
        assert_eq!(
            Layer::ALL.iter().filter(|l| l.in_thin_snapshot()).count(),
            5
        );
    }

    #[test]
    fn snapshot_colors_match_the_live_path_where_data_exists() {
        let snap = iw_store_postcard::HistorySnapshot {
            version: 1,
            time_myr: 10.0,
            sea_level_m: -100.0,
            phase: iw_core::Phase::Drift,
            elevation_m: vec![-2000.0, 300.0],
            biome: vec![Biome::Ocean, Biome::Desert],
            plate_id: vec![NO_PLATE, 4],
            ice_thickness_m: vec![0.0, 0.0],
        };
        let (colors, substituted) = snapshot_colors(Layer::Biomes, &snap);
        assert!(!substituted);
        assert_eq!(colors[1], biome_color(Biome::Desert));
        let (colors, substituted) = snapshot_colors(Layer::Temperature, &snap);
        assert!(substituted, "temperature is not in a thin snapshot");
        assert_eq!(colors[0], elevation_color(-2000.0, -100.0));
        let (colors, _) = snapshot_colors(Layer::Plates, &snap);
        assert_eq!(colors[0], plate_color(NO_PLATE));
        assert_eq!(colors[1], plate_color(4));
    }

    #[test]
    fn layer_indices_round_trip_and_wrap() {
        for (i, layer) in Layer::ALL.iter().enumerate() {
            assert_eq!(layer.index(), i);
            assert_eq!(Layer::from_index(i), *layer);
        }
        assert_eq!(Layer::from_index(999), Layer::Beauty);
        assert_eq!(Layer::WaterFlux.next(), Layer::Beauty);
        assert_eq!(Layer::Beauty.next(), Layer::Elevation);
    }

    #[test]
    fn cell_colors_matches_per_cell_colouring() {
        let mut c = cells(64);
        for (i, e) in c.elevation_m.iter_mut().enumerate() {
            *e = (i as f32 - 32.0) * 200.0;
        }
        let scales = LayerScales::from_cells(&c);
        let bulk = cell_colors(Layer::Elevation, &c, 0.0, scales);
        assert_eq!(bulk.len(), 64);
        for (i, got) in bulk.iter().enumerate() {
            assert_eq!(*got, cell_color(Layer::Elevation, &c, 0.0, i, scales));
        }
    }
}
