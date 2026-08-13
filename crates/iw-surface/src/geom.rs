//! Mesh-derived geometry the surface passes need in SI units.
//!
//! Everything here is a pure function of the [`Mesh`], so caching it inside
//! [`crate::SurfaceProcess`] carries no simulation state: a fresh process
//! instance rebuilds a bit-identical table.

use iw_mesh::Mesh;

/// Per-cell areas and per-edge distances in metres.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// Cell count this table was built for.
    pub n_cells: usize,
    /// Great-circle centre distance for every CSR neighbour slot, metres.
    /// Indexed exactly like [`Mesh::neighbors`].
    pub dist_m: Vec<f32>,
    /// Cell area in m^2 (`areas_km2 * 1e6`), as `f64` for volume accounting.
    pub area_m2: Vec<f64>,
    /// Mean centre-to-centre spacing over the whole mesh, metres. Used to
    /// scale grid-resolution-dependent rates (hillslope diffusivity).
    pub mean_pitch_m: f32,
}

impl Geometry {
    /// Build the table. O(n) great-circle evaluations, done once per mesh.
    pub fn build(mesh: &Mesh) -> Geometry {
        let n = mesh.n_cells();
        let area_m2: Vec<f64> = mesh.areas_km2.iter().map(|a| *a as f64 * 1.0e6).collect();

        let mut dist_m = vec![0.0f32; mesh.neighbors.len()];
        let mut sum = 0.0f64;
        for c in 0..n {
            let a = mesh.neighbor_offsets[c] as usize;
            let b = mesh.neighbor_offsets[c + 1] as usize;
            let ci = mesh.centers[c];
            for (slot, out) in dist_m[a..b].iter_mut().enumerate() {
                let d =
                    iw_mesh::great_circle_km(ci, mesh.centers[mesh.neighbors[a + slot] as usize])
                        * 1000.0;
                *out = d;
                sum += d as f64;
            }
        }
        let mean_pitch_m = (sum / mesh.neighbors.len().max(1) as f64) as f32;

        Geometry {
            n_cells: n,
            dist_m,
            area_m2,
            mean_pitch_m,
        }
    }

    /// Distance from `cell` to its `k`-th CSR neighbour, metres.
    #[inline]
    pub fn dist(&self, mesh: &Mesh, cell: u32, k: usize) -> f32 {
        self.dist_m[mesh.neighbor_offsets[cell as usize] as usize + k]
    }
}
