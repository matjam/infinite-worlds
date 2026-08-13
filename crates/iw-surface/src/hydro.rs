//! Drainage: priority-flood depression filling, single-direction flow routing
//! and discharge accumulation.
//!
//! # Algorithm
//!
//! 1. **Priority flood** (Barnes et al. 2014). Every cell below sea level is an
//!    outlet and enters a min-heap at its own elevation. Popping the lowest
//!    unresolved cell and raising each unvisited neighbour to
//!    `max(elevation, popped_level + eps)` produces a *filled* surface that is
//!    strictly monotone along every drainage path — no flats, no pits, one pass,
//!    `O(n log n)`. Standing water is `filled - elevation` on land.
//! 2. **Flow direction**: steepest descent on the filled surface, ties broken by
//!    the lowest cell id. Because the fill is strictly monotone, every land cell
//!    has a downhill neighbour, so the flow graph is a forest rooted in the
//!    ocean.
//! 3. **Accumulation**: the heap pops cells in non-decreasing filled order, so
//!    walking that order backwards visits every cell before its receiver — a
//!    topological order for free. Discharge is local runoff plus everything
//!    upstream, floored at zero so endorheic basins can dry out.
//!
//! The whole pass is serial (a heap and a chain of dependent adds); it is the
//! step's dominant cost and is budgeted as such.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use iw_core::Planet;
use iw_mesh::Mesh;

use crate::geom::Geometry;

/// Sentinel in [`Hydrology::downstream`] for "drains nowhere" (ocean cells and
/// the global outlet).
pub const NO_DOWNSTREAM: u32 = u32::MAX;

/// Vertical increment stacked on each flood step so the filled surface is
/// strictly increasing away from the outlet. A millimetre is far below any
/// elevation the model resolves and keeps lake surfaces flat to within
/// `1 mm * cells across the lake`.
pub const FILL_EPSILON_M: f32 = 1.0e-3;

/// Standing water shallower than this is fill epsilon noise, not a lake.
pub const LAKE_MIN_DEPTH_M: f32 = 0.05;

/// Evaporative demand of an open water surface, m/yr.
pub const LAKE_EVAP_M_YR: f32 = 0.5;
/// Evapotranspiration lost before precipitation becomes runoff on land, m/yr.
pub const LAND_ET_M_YR: f32 = 0.5;

/// Rock volume precipitated per unit volume of water evaporated in a closed
/// basin, m^3/m^3. Rivers carry ~200 mg/L of dissolved solids; at an evaporite
/// density of 2200 kg/m^3 that is ~1e-4 m^3 of rock per m^3 of water.
pub const SALT_YIELD_M3_PER_M3: f64 = 1.0e-4;

/// Result of one drainage solve. All vectors are `n_cells` long.
#[derive(Debug, Default, Clone)]
pub struct Hydrology {
    /// Depression-filled surface height, metres on the geoid datum.
    pub filled_m: Vec<f32>,
    /// Cells in priority-flood pop order (non-decreasing `filled_m`).
    /// Iterate in reverse for an upstream-first topological order.
    pub order: Vec<u32>,
    /// Steepest-descent receiver on the filled surface, or [`NO_DOWNSTREAM`].
    pub downstream: Vec<u32>,
    /// Surface slope from a cell to its receiver, dimensionless (0 for sinks).
    pub slope: Vec<f32>,
    /// Discharge leaving the cell, m^3/yr (after local losses, floored at 0).
    pub discharge_m3_yr: Vec<f64>,
    /// Discharge arriving from upstream, m^3/yr.
    pub inflow_m3_yr: Vec<f64>,
    /// True where the fill left standing fresh water above the ground.
    pub is_lake: Vec<bool>,
    /// Water evaporated in closed (outflow-free) basin cells, m^3/yr. Drives
    /// evaporite precipitation; zero everywhere else.
    pub closed_evap_m3_yr: Vec<f64>,
    heap: BinaryHeap<HeapItem>,
    visited: Vec<bool>,
}

/// Min-heap entry ordered by fill level, then by cell id, so the pop sequence
/// is a total order that depends only on the state, never on insertion order.
#[derive(Debug, Clone, Copy)]
struct HeapItem {
    key: f32,
    cell: u32,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap: invert both keys so the smallest level
        // and then the smallest id come out first.
        other
            .key
            .total_cmp(&self.key)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapItem {}

impl Hydrology {
    /// Empty solver; buffers grow on first use.
    pub fn new() -> Hydrology {
        Hydrology::default()
    }

    fn resize(&mut self, n: usize) {
        self.filled_m.clear();
        self.filled_m.resize(n, 0.0);
        self.downstream.clear();
        self.downstream.resize(n, NO_DOWNSTREAM);
        self.slope.clear();
        self.slope.resize(n, 0.0);
        self.discharge_m3_yr.clear();
        self.discharge_m3_yr.resize(n, 0.0);
        self.inflow_m3_yr.clear();
        self.inflow_m3_yr.resize(n, 0.0);
        self.is_lake.clear();
        self.is_lake.resize(n, false);
        self.closed_evap_m3_yr.clear();
        self.closed_evap_m3_yr.resize(n, 0.0);
        self.visited.clear();
        self.visited.resize(n, false);
        self.order.clear();
        self.order.reserve(n);
        self.heap.clear();
    }

    /// Fill depressions, route flow and accumulate discharge.
    ///
    /// `melt_m_yr` is meltwater released by the glacial pass this step (may be
    /// empty, meaning none). Writes `planet.lake_depth_m` and
    /// `planet.water_flux_m3_yr`.
    pub fn solve(&mut self, planet: &mut Planet, mesh: &Mesh, geom: &Geometry, melt_m_yr: &[f32]) {
        let n = planet.n_cells();
        self.resize(n);
        self.fill(planet, mesh);
        self.route(planet, mesh, geom);
        self.accumulate(planet, geom, melt_m_yr);
    }

    /// Priority flood from every sub-sea-level cell.
    ///
    /// "Below sea level" is the project-wide definition of ocean
    /// ([`Planet::is_ocean`]) and the same one iw-geology's hypsometric solve
    /// uses, so a landlocked basin whose floor lies below sea level counts as
    /// sea here too. A planet with no such cell (a dry world) falls back to its
    /// single lowest cell as the global outlet.
    fn fill(&mut self, planet: &Planet, mesh: &Mesh) {
        let n = planet.n_cells();
        let sea = planet.sea_level_m;
        for i in 0..n {
            if planet.elevation_m[i] < sea {
                self.filled_m[i] = planet.elevation_m[i];
                self.visited[i] = true;
                self.heap.push(HeapItem {
                    key: planet.elevation_m[i],
                    cell: i as u32,
                });
            }
        }
        if self.heap.is_empty() {
            let mut lowest = 0usize;
            for i in 1..n {
                if planet.elevation_m[i] < planet.elevation_m[lowest] {
                    lowest = i;
                }
            }
            self.filled_m[lowest] = planet.elevation_m[lowest];
            self.visited[lowest] = true;
            self.heap.push(HeapItem {
                key: planet.elevation_m[lowest],
                cell: lowest as u32,
            });
        }

        while let Some(it) = self.heap.pop() {
            self.order.push(it.cell);
            for &m in mesh.neighbors_of(it.cell) {
                let j = m as usize;
                if self.visited[j] {
                    continue;
                }
                self.visited[j] = true;
                let level = planet.elevation_m[j].max(it.key + FILL_EPSILON_M);
                self.filled_m[j] = level;
                self.heap.push(HeapItem {
                    key: level,
                    cell: m,
                });
            }
        }
        debug_assert_eq!(self.order.len(), n, "flood must reach every cell");
    }

    /// Steepest descent on the filled surface; also marks lakes.
    fn route(&mut self, planet: &mut Planet, mesh: &Mesh, geom: &Geometry) {
        let sea = planet.sea_level_m;
        for cell in 0..planet.n_cells() as u32 {
            let i = cell as usize;
            let depth = self.filled_m[i] - planet.elevation_m[i];
            let land = planet.elevation_m[i] >= sea;
            let lake = land && depth > LAKE_MIN_DEPTH_M;
            self.is_lake[i] = lake;
            planet.lake_depth_m[i] = if lake { depth } else { 0.0 };

            if !land {
                self.downstream[i] = NO_DOWNSTREAM;
                self.slope[i] = 0.0;
                continue;
            }
            let here = self.filled_m[i];
            let base = mesh.neighbor_offsets[i] as usize;
            let mut best = NO_DOWNSTREAM;
            let mut best_level = here;
            let mut best_slope = 0.0f32;
            for (k, &m) in mesh.neighbors_of(cell).iter().enumerate() {
                let lv = self.filled_m[m as usize];
                if lv >= here {
                    continue;
                }
                // Steepest descent; equal levels cannot occur along a fill
                // path, but ties still resolve to the lowest id.
                let better =
                    best == NO_DOWNSTREAM || lv < best_level || (lv == best_level && m < best);
                if better {
                    best_level = lv;
                    best = m;
                    best_slope = (here - lv) / geom.dist_m[base + k].max(1.0);
                }
            }
            self.downstream[i] = best;
            self.slope[i] = best_slope;
        }
    }

    /// Local runoff plus upstream discharge, walked in topological order.
    fn accumulate(&mut self, planet: &mut Planet, geom: &Geometry, melt_m_yr: &[f32]) {
        let n = planet.n_cells();
        let sea = planet.sea_level_m;
        for idx in (0..n).rev() {
            let cell = self.order[idx];
            let i = cell as usize;
            let area = geom.area_m2[i];
            let precip_m = (planet.precip_mm_yr[i] as f64 / 1000.0).max(0.0);
            let melt_m = melt_m_yr.get(i).copied().unwrap_or(0.0) as f64;

            let inflow = self.discharge_m3_yr[i];
            self.inflow_m3_yr[i] = inflow;

            if planet.elevation_m[i] < sea {
                // Ocean: the run ends here.
                self.discharge_m3_yr[i] = inflow;
                planet.water_flux_m3_yr[i] = 0.0;
                continue;
            }

            let local = if self.is_lake[i] {
                (precip_m - LAKE_EVAP_M_YR as f64 + melt_m) * area
            } else if planet.ice_thickness_m[i] > 0.0 {
                // Snow that fell as ice is already booked by the glacial pass.
                melt_m * area
            } else {
                ((precip_m - LAND_ET_M_YR as f64).max(0.0) + melt_m) * area
            };

            let out = (inflow + local).max(0.0);
            self.discharge_m3_yr[i] = out;
            planet.water_flux_m3_yr[i] = out as f32;

            // A lake with no outflow is a terminal basin: everything that
            // arrives evaporates and leaves its dissolved load behind.
            if self.is_lake[i] && out <= 0.0 {
                let supplied = inflow + (precip_m + melt_m) * area;
                self.closed_evap_m3_yr[i] = supplied.min(LAKE_EVAP_M_YR as f64 * area).max(0.0);
                planet.lake_depth_m[i] = 0.0;
                self.is_lake[i] = false;
            } else {
                self.closed_evap_m3_yr[i] = 0.0;
            }

            let d = self.downstream[i];
            if d != NO_DOWNSTREAM {
                self.discharge_m3_yr[d as usize] += out;
            }
        }
    }

    /// Publish the routing into `planet.flow_to` as neighbor-slot indices
    /// (docs/voronoi-v2.md §4): a cell's river exits through the edge shared
    /// with its downstream neighbor. Renderers thread the polyline through
    /// those edges; [`iw_core::planet::FLOW_NONE`] marks ocean/sinks.
    pub(crate) fn publish_flow_edges(&self, planet: &mut Planet, mesh: &Mesh) {
        use iw_core::planet::FLOW_NONE;
        for i in 0..planet.n_cells() {
            let d = self.downstream[i];
            planet.flow_to[i] = if d == NO_DOWNSTREAM {
                FLOW_NONE
            } else {
                mesh.neighbors_of(i as u32)
                    .iter()
                    .position(|&m| m == d)
                    .map(|slot| slot as u8)
                    .unwrap_or(FLOW_NONE)
            };
        }
    }
}
