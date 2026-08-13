use glam::{DQuat, DVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::{Biome, Phase, PlanetConfig, StrataColumns};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CrustType {
    Oceanic,
    Continental,
}

/// A rigid plate rotating about an Euler pole (DESIGN.md §5 Phase 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plate {
    /// Unit rotation axis.
    pub euler_pole: DVec3,
    /// Angular speed about the pole, radians per Myr. Sign = right-hand rule.
    pub omega_rad_myr: f64,
    /// True once a continent-continent collision has welded this plate to
    /// another; welded plates share motion until rifting separates them.
    pub welded_to: Option<u16>,
    /// Rotation accumulated since this plate's fields last advected (drift v2).
    /// Composed every step; consumed (reset to identity) by the remap pass once
    /// the implied surface displacement reaches a cell pitch. Serialized so
    /// checkpoint resume reproduces the remap schedule exactly.
    pub accum: DQuat,
}

impl Plate {
    /// Surface velocity (m/yr) of a point at unit direction `r`.
    pub fn velocity_m_yr(&self, r: Vec3) -> Vec3 {
        let w = self.euler_pole * self.omega_rad_myr; // rad/Myr
        let v = w.cross(r.as_dvec3()); // (rad/Myr) * unit
                                       // radians/Myr at Earth radius -> meters/year
        (v * iw_mesh::EARTH_RADIUS_M / 1.0e6).as_vec3()
    }

    /// Fold this step's rotation into the pending advection rotation.
    pub fn accumulate(&mut self, dt_myr: f64) {
        let angle = self.omega_rad_myr * dt_myr;
        if angle.abs() > 0.0 {
            self.accum = (DQuat::from_axis_angle(self.euler_pole, angle) * self.accum).normalize();
        }
    }

    /// Surface displacement (meters) at the sphere's "equator" of the pending
    /// rotation — the upper bound over the plate's cells.
    pub fn pending_displacement_m(&self) -> f64 {
        // Rotation angle of a unit quaternion is 2*acos(|w|), in [0, pi].
        2.0 * self.accum.w.abs().clamp(0.0, 1.0).acos() * iw_mesh::EARTH_RADIUS_M
    }
}

/// Per-cell tectonic state bits, rewritten by tectonics each step and consumed
/// by geology (igneous emplacement, metamorphism bonuses) and the renderer.
pub mod cell_flags {
    /// Oceanic cell currently being consumed at a convergent boundary.
    pub const SUBDUCTING: u8 = 1 << 0;
    /// Volcanic arc cell (overriding side of a subduction zone).
    pub const ARC: u8 = 1 << 1;
    /// Active continent-continent collision zone.
    pub const COLLISION: u8 = 1 << 2;
    /// Active rift / spreading ridge.
    pub const RIFT: u8 = 1 << 3;
    /// Currently over a mantle plume.
    pub const HOTSPOT: u8 = 1 << 4;
    /// Transform boundary cell.
    pub const TRANSFORM: u8 = 1 << 5;
    /// Ancient collision seam (weak zone; rifts prefer to nucleate here).
    pub const SUTURE: u8 = 1 << 6;
}

/// A fixed mantle plume (DESIGN.md §5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hotspot {
    /// Unit direction of the plume.
    pub pos: Vec3,
    /// Relative strength, ~0.5..2.0.
    pub strength: f32,
}

/// Complete simulation state, struct-of-arrays keyed by cell id.
///
/// Ownership convention (who writes what — see IMPLEMENTATION_PLAN.md §3):
/// tectonics: `plate_id`, `plates`, `crust_*`, `hotspots`, columns (create/destroy);
/// geology: `elevation_m`, `sea_level_m`, columns (metamorphism/intrusion);
/// climate: `temperature_c`, `precip_mm_yr`, `wind`;
/// surface: `sediment_m`, `ice_thickness_m`, `water_flux_m3_yr`, `lake_depth_m`,
/// columns (erosion/deposition); biomes: `biome`.
#[derive(Clone, Serialize, Deserialize)]
pub struct Planet {
    pub config: PlanetConfig,
    pub phase: Phase,
    /// Elapsed simulation time in Myr since the start of CrustalFormation.
    pub time_myr: f64,
    /// Monotonic step counter across all phases (drives per-step RNG streams).
    pub step_index: u64,
    /// Sea surface height relative to the reference geoid, meters.
    pub sea_level_m: f32,

    // --- per-cell fields, all len == n_cells ---
    pub plate_id: Vec<u16>,
    pub crust_type: Vec<CrustType>,
    pub crust_thickness_m: Vec<f32>,
    pub crust_density_kg_m3: Vec<f32>,
    /// Age of oceanic crust; meaningless (kept 0) for continental cells.
    pub crust_age_myr: Vec<f32>,
    /// Solid surface height relative to the reference geoid, meters.
    /// Written only by the isostasy pass in iw-geology.
    pub elevation_m: Vec<f32>,
    /// Loose regolith/alluvium depth above bedrock columns.
    pub sediment_m: Vec<f32>,
    pub temperature_c: Vec<f32>,
    pub precip_mm_yr: Vec<f32>,
    /// Per-cell wind vector in the local tangent plane (world coords), m/s.
    pub wind_m_s: Vec<Vec3>,
    pub ice_thickness_m: Vec<f32>,
    pub water_flux_m3_yr: Vec<f32>,
    /// Depth of standing water above ground for lake cells (0 elsewhere).
    pub lake_depth_m: Vec<f32>,
    pub biome: Vec<Biome>,
    /// Bitwise OR of [`cell_flags`] constants; tectonics rewrites each step.
    pub tectonic_flags: Vec<u8>,
    /// River routing (docs/voronoi-v2.md): the neighbor-slot index this cell's
    /// water exits through ([`FLOW_NONE`] for ocean, pits, and dry cells).
    /// Written by the surface process; discharge is `water_flux_m3_yr`.
    pub flow_to: Vec<u8>,
    pub columns: StrataColumns,

    // --- entities ---
    pub plates: Vec<Plate>,
    pub hotspots: Vec<Hotspot>,

    /// The Voronoi generators this planet's mesh was built from, in f64 —
    /// simulation state, because the tessellation is terrain-driven and
    /// re-tessellates between eras. Checkpoint resume rebuilds the mesh from
    /// these bit-exactly (the hull is deterministic). Empty only for legacy
    /// Goldberg meshes in unit tests.
    pub mesh_generators: Vec<glam::DVec3>,
}

/// Sentinel for [`Planet::flow_to`]: no outflow edge.
pub const FLOW_NONE: u8 = u8::MAX;

impl Planet {
    /// Fresh all-ocean proto-planet: thin young basaltic crust everywhere.
    pub fn new(config: PlanetConfig, n_cells: usize) -> Planet {
        Planet {
            phase: Phase::CrustalFormation,
            time_myr: 0.0,
            step_index: 0,
            sea_level_m: 0.0,
            plate_id: vec![0; n_cells],
            crust_type: vec![CrustType::Oceanic; n_cells],
            crust_thickness_m: vec![7_000.0; n_cells],
            crust_density_kg_m3: vec![3_000.0; n_cells],
            crust_age_myr: vec![0.0; n_cells],
            elevation_m: vec![0.0; n_cells],
            sediment_m: vec![0.0; n_cells],
            temperature_c: vec![15.0; n_cells],
            precip_mm_yr: vec![1_000.0; n_cells],
            wind_m_s: vec![Vec3::ZERO; n_cells],
            ice_thickness_m: vec![0.0; n_cells],
            water_flux_m3_yr: vec![0.0; n_cells],
            lake_depth_m: vec![0.0; n_cells],
            biome: vec![Biome::Unclassified; n_cells],
            tectonic_flags: vec![0; n_cells],
            flow_to: vec![FLOW_NONE; n_cells],
            columns: StrataColumns::new(n_cells),
            plates: Vec::new(),
            hotspots: Vec::new(),
            mesh_generators: Vec::new(),
            config,
        }
    }

    #[inline]
    pub fn n_cells(&self) -> usize {
        self.plate_id.len()
    }

    /// Ground (or ice-free rock) height above sea level; negative = submerged.
    #[inline]
    pub fn elevation_above_sea_m(&self, cell: u32) -> f32 {
        self.elevation_m[cell as usize] - self.sea_level_m
    }

    #[inline]
    pub fn is_ocean(&self, cell: u32) -> bool {
        self.elevation_m[cell as usize] < self.sea_level_m
            && self.lake_depth_m[cell as usize] == 0.0
    }
}
