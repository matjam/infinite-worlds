use std::sync::Arc;

use glam::Vec3;
use iw_mesh::Mesh;

use crate::{Biome, CrustType, Phase, Planet, RockType};

/// Immutable per-cell snapshot data shared with observers via `Arc`.
pub struct ViewCells {
    pub elevation_m: Vec<f32>,
    pub sediment_m: Vec<f32>,
    pub biome: Vec<Biome>,
    pub plate_id: Vec<u16>,
    pub crust_type: Vec<CrustType>,
    pub crust_age_myr: Vec<f32>,
    pub crust_thickness_m: Vec<f32>,
    pub temperature_c: Vec<f32>,
    pub precip_mm_yr: Vec<f32>,
    pub ice_thickness_m: Vec<f32>,
    pub water_flux_m3_yr: Vec<f32>,
    pub lake_depth_m: Vec<f32>,
    /// Topmost rock of each column (None = bare mantle/none deposited).
    pub top_rock: Vec<Option<RockType>>,
    /// Plate surface velocity per cell, m/yr (for the plate arrows layer).
    pub plate_velocity_m_yr: Vec<Vec3>,
    /// Bitwise OR of [`crate::planet::cell_flags`] constants.
    pub tectonic_flags: Vec<u8>,
}

/// A consistent, cheap-to-clone snapshot of the planet for rendering and
/// inspection. Published by the sim loop at safe points (never mid-step).
#[derive(Clone)]
pub struct PlanetView {
    /// Monotonically increasing snapshot version.
    pub version: u64,
    pub phase: Phase,
    pub time_myr: f64,
    pub sea_level_m: f32,
    pub cells: Arc<ViewCells>,
}

impl PlanetView {
    pub fn capture(version: u64, planet: &Planet, mesh: &Mesh) -> PlanetView {
        let n = planet.n_cells();
        let mut top_rock = Vec::with_capacity(n);
        for c in 0..n as u32 {
            top_rock.push(planet.columns.top_rock(c));
        }
        let mut plate_velocity_m_yr = vec![Vec3::ZERO; n];
        for (i, v) in plate_velocity_m_yr.iter_mut().enumerate() {
            if let Some(plate) = planet.plates.get(planet.plate_id[i] as usize) {
                *v = plate.velocity_m_yr(mesh.centers[i]);
            }
        }
        PlanetView {
            version,
            phase: planet.phase,
            time_myr: planet.time_myr,
            sea_level_m: planet.sea_level_m,
            cells: Arc::new(ViewCells {
                elevation_m: planet.elevation_m.clone(),
                sediment_m: planet.sediment_m.clone(),
                biome: planet.biome.clone(),
                plate_id: planet.plate_id.clone(),
                crust_type: planet.crust_type.clone(),
                crust_age_myr: planet.crust_age_myr.clone(),
                crust_thickness_m: planet.crust_thickness_m.clone(),
                temperature_c: planet.temperature_c.clone(),
                precip_mm_yr: planet.precip_mm_yr.clone(),
                ice_thickness_m: planet.ice_thickness_m.clone(),
                water_flux_m3_yr: planet.water_flux_m3_yr.clone(),
                lake_depth_m: planet.lake_depth_m.clone(),
                top_rock,
                plate_velocity_m_yr,
                tectonic_flags: planet.tectonic_flags.clone(),
            }),
        }
    }
}
