//! Golden summary statistics: a compact JSON fingerprint of a generated
//! planet, used by `iw-headless` itself (printed + written to
//! `summary.json`) and by WP12's determinism / golden-planet tests.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Context;
use iw_core::{Biome, Planet, PlanetConfig};
use iw_mesh::Mesh;
use serde::Serialize;

/// Golden stats for a generated (or resumed) planet.
#[derive(Debug, Serialize)]
pub struct Summary {
    /// RNG seed the planet was generated with.
    pub seed: u64,
    /// Goldberg subdivision level.
    pub level: u8,
    /// Full config the planet was generated (or resumed) with.
    pub config: PlanetConfig,
    /// Area-weighted fraction of the surface at or above sea level.
    pub land_fraction: f64,
    /// Sea level, meters.
    pub sea_level_m: f32,
    /// Minimum elevation, meters.
    pub elevation_min_m: f32,
    /// Maximum elevation, meters.
    pub elevation_max_m: f32,
    /// Area-weighted 25th percentile elevation, meters.
    pub elevation_q1_m: f32,
    /// Area-weighted 50th percentile (median) elevation, meters.
    pub elevation_median_m: f32,
    /// Area-weighted 75th percentile elevation, meters.
    pub elevation_q3_m: f32,
    /// Number of distinct plate ids present.
    pub plate_count: usize,
    /// Area fraction of the surface in each biome, keyed by
    /// [`Biome::name`].
    pub biome_fractions: BTreeMap<String, f64>,
    /// Area fraction of the surface with each rock type on top of its
    /// column, keyed by `{:?}` of [`iw_core::RockType`] (or `"None"` for
    /// bare cells with nothing deposited).
    pub rock_fractions: BTreeMap<String, f64>,
    /// Wall-clock time the simulation run took, seconds.
    pub runtime_secs: f64,
}

/// Compute golden stats for `planet` on `mesh`.
pub fn compute_summary(planet: &Planet, mesh: &Mesh, runtime_secs: f64) -> Summary {
    let n = planet.n_cells();
    let areas = &mesh.areas_km2;
    let total_area: f64 = areas.iter().map(|a| *a as f64).sum();

    let mut land_area = 0.0f64;
    let mut elevation_min = f32::INFINITY;
    let mut elevation_max = f32::NEG_INFINITY;
    for (&e, &area) in planet.elevation_m.iter().zip(areas.iter()) {
        if e >= planet.sea_level_m {
            land_area += area as f64;
        }
        elevation_min = elevation_min.min(e);
        elevation_max = elevation_max.max(e);
    }
    let land_fraction = if total_area > 0.0 {
        land_area / total_area
    } else {
        0.0
    };

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| planet.elevation_m[a].total_cmp(&planet.elevation_m[b]));
    let quartile = |q: f64| -> f32 {
        if order.is_empty() {
            return 0.0;
        }
        let target = q * total_area;
        let mut cum = 0.0f64;
        for &i in &order {
            cum += areas[i] as f64;
            if cum >= target {
                return planet.elevation_m[i];
            }
        }
        planet.elevation_m[*order.last().unwrap()]
    };

    let plate_count = planet
        .plate_id
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    let mut biome_area: HashMap<Biome, f64> = HashMap::new();
    let mut rock_area: HashMap<String, f64> = HashMap::new();
    for (i, (&biome, &area)) in planet.biome.iter().zip(areas.iter()).enumerate() {
        let area = area as f64;
        *biome_area.entry(biome).or_insert(0.0) += area;
        let rock_key = match planet.columns.top_rock(i as u32) {
            Some(r) => format!("{r:?}"),
            None => "None".to_string(),
        };
        *rock_area.entry(rock_key).or_insert(0.0) += area;
    }
    let biome_fractions: BTreeMap<String, f64> = biome_area
        .into_iter()
        .map(|(b, a)| (b.name().to_string(), safe_div(a, total_area)))
        .collect();
    let rock_fractions: BTreeMap<String, f64> = rock_area
        .into_iter()
        .map(|(k, a)| (k, safe_div(a, total_area)))
        .collect();

    Summary {
        seed: planet.config.seed,
        level: planet.config.subdivision_level,
        config: planet.config.clone(),
        land_fraction,
        sea_level_m: planet.sea_level_m,
        elevation_min_m: elevation_min,
        elevation_max_m: elevation_max,
        elevation_q1_m: quartile(0.25),
        elevation_median_m: quartile(0.5),
        elevation_q3_m: quartile(0.75),
        plate_count,
        biome_fractions,
        rock_fractions,
        runtime_secs,
    }
}

fn safe_div(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        a / b
    } else {
        0.0
    }
}

/// Write `summary.json` into `out_dir` and echo it to stdout.
pub fn write_summary(out_dir: &Path, summary: &Summary) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(summary).context("serializing summary")?;
    std::fs::write(out_dir.join("summary.json"), &json)
        .with_context(|| format!("writing {}", out_dir.join("summary.json").display()))?;
    println!("{json}");
    Ok(())
}
