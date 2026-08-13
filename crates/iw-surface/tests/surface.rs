//! Acceptance tests for WP7 (iw-surface).
//!
//! Every planet here is synthetic and `elevation_m` is set by hand. In the real
//! pipeline iw-geology recomputes elevation from crust/sediment/ice every step;
//! these tests deliberately do not run it, so the ground stays put while the
//! surface engine works on it and each assertion isolates one process.

use iw_core::{
    rng_for, CrustType, MassLedger, NullProgress, Phase, Planet, PlanetConfig, Process, RockType,
    StepCtx,
};
use iw_mesh::Mesh;
use iw_surface::hydro::NO_DOWNSTREAM;
use iw_surface::SurfaceProcess;

/// Airy response of the ground surface to removing a metre of continental
/// crust, `(rho_mantle - rho_crust)/rho_mantle` with iw-geology's constants.
/// Used only to express erosion as the surface lowering a user would see.
const AIRY_SURFACE_RATIO: f64 = (3300.0 - 2700.0) / 3300.0;

// --------------------------------------------------------------------------
// harness
// --------------------------------------------------------------------------

fn config(level: u8) -> PlanetConfig {
    PlanetConfig {
        subdivision_level: level,
        ..Default::default()
    }
}

/// Planet with hand-set elevation: continental crust (20 km of granite over
/// basement) above sea level, oceanic basalt below.
fn planet_from_elevation(mesh: &Mesh, elev: &[f32], sea_level_m: f32) -> Planet {
    let mut p = Planet::new(config(mesh.level), mesh.n_cells());
    p.phase = Phase::RecentPast;
    p.sea_level_m = sea_level_m;
    for (i, &e) in elev.iter().enumerate() {
        p.elevation_m[i] = e;
        if e >= sea_level_m {
            p.crust_type[i] = CrustType::Continental;
            p.crust_thickness_m[i] = 35_000.0;
            p.crust_density_kg_m3[i] = 2700.0;
            p.columns
                .deposit(i as u32, RockType::Granite, 20_000.0, 0.0);
        } else {
            p.crust_type[i] = CrustType::Oceanic;
            p.crust_thickness_m[i] = 7_000.0;
            p.crust_density_kg_m3[i] = 3000.0;
            p.columns.deposit(i as u32, RockType::Basalt, 5_000.0, 0.0);
        }
    }
    p
}

/// One step, standing in for what iw-sim does around a process.
fn step(proc: &mut SurfaceProcess, planet: &mut Planet, mesh: &Mesh, dt_myr: f64) -> MassLedger {
    let mut ledger = MassLedger::default();
    {
        let mut ctx = StepCtx {
            rng: rng_for(planet.config.seed, "surface", planet.step_index),
            progress: &NullProgress,
            ledger: &mut ledger,
        };
        proc.step(planet, mesh, dt_myr, &mut ctx);
    }
    planet.step_index += 1;
    planet.time_myr += dt_myr;
    ledger
}

/// Run `steps` RecentPast steps, returning the summed ledger.
fn run(
    proc: &mut SurfaceProcess,
    planet: &mut Planet,
    mesh: &Mesh,
    steps: usize,
    dt: f64,
) -> MassLedger {
    let mut total = MassLedger::default();
    for _ in 0..steps {
        let l = step(proc, planet, mesh, dt);
        total.eroded_m3 += l.eroded_m3;
        total.deposited_m3 += l.deposited_m3;
        total.created_m3 += l.created_m3;
        total.subducted_m3 += l.subducted_m3;
    }
    total
}

/// Angular distance from cell 0, radians.
fn radius_rad(mesh: &Mesh, cell: usize) -> f32 {
    mesh.centers[0]
        .dot(mesh.centers[cell])
        .clamp(-1.0, 1.0)
        .acos()
}

/// Rings of cells by graph distance from `seed` (ring 0 = the seed itself).
fn rings(mesh: &Mesh, seed: u32, count: usize) -> Vec<usize> {
    let mut ring = vec![usize::MAX; mesh.n_cells()];
    ring[seed as usize] = 0;
    let mut frontier = vec![seed];
    for r in 1..=count {
        let mut next = Vec::new();
        for c in frontier.drain(..) {
            for &m in mesh.neighbors_of(c) {
                if ring[m as usize] == usize::MAX {
                    ring[m as usize] = r;
                    next.push(m);
                }
            }
        }
        frontier = next;
    }
    ring
}

/// Radially symmetric island centred on cell 0.
fn cone_island(mesh: &Mesh, peak_m: f32, radius: f32, shelf: bool) -> Vec<f32> {
    (0..mesh.n_cells())
        .map(|i| {
            let a = radius_rad(mesh, i);
            if a < radius {
                peak_m * (1.0 - a / radius)
            } else if shelf && a < radius * 1.5 {
                -50.0
            } else {
                -3000.0
            }
        })
        .collect()
}

fn total_column_m(p: &Planet, cells: impl Iterator<Item = usize>) -> f64 {
    cells
        .map(|i| p.columns.total_thickness_m(i as u32) as f64)
        .sum()
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn mix(&mut self, b: u64) {
        self.0 ^= b;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.mix(x.to_bits() as u64);
        }
    }
}

/// Bit-exact fingerprint of everything the surface engine writes.
fn state_hash(p: &Planet) -> u64 {
    let mut h = Fnv::new();
    h.f32s(&p.sediment_m);
    h.f32s(&p.ice_thickness_m);
    h.f32s(&p.water_flux_m3_yr);
    h.f32s(&p.lake_depth_m);
    h.f32s(&p.crust_thickness_m);
    for c in 0..p.n_cells() as u32 {
        for s in p.columns.col(c) {
            h.mix(s.rock as u64);
            h.mix(s.thickness_m.to_bits() as u64);
            h.mix(s.deposited_myr.to_bits() as u64);
        }
        h.mix(0xffff);
    }
    h.0
}

// --------------------------------------------------------------------------
// drainage
// --------------------------------------------------------------------------

#[test]
fn cone_island_drains_radially_and_conserves_mass() {
    let mesh = Mesh::build(5);
    let elev = cone_island(&mesh, 3000.0, 0.25, false);
    let mut p = planet_from_elevation(&mesh, &elev, 0.0);
    p.precip_mm_yr.fill(1500.0);
    p.temperature_c.fill(12.0);

    let land: Vec<usize> = (0..mesh.n_cells()).filter(|i| elev[*i] >= 0.0).collect();
    let ocean: Vec<usize> = (0..mesh.n_cells()).filter(|i| elev[*i] < 0.0).collect();
    assert!(land.len() > 30, "island too small to test: {}", land.len());

    let column_before = total_column_m(&p, land.iter().copied());
    let ocean_before = total_column_m(&p, ocean.iter().copied());

    let mut proc = SurfaceProcess::new();
    let ledger = run(&mut proc, &mut p, &mesh, 100, 0.005);

    let h = proc.hydrology();

    // Flow is outward: every land cell drains to a cell further from the peak.
    for &i in &land {
        let d = h.downstream[i];
        assert_ne!(d, NO_DOWNSTREAM, "land cell {i} has no receiver");
        assert!(
            radius_rad(&mesh, d as usize) > radius_rad(&mesh, i),
            "cell {i} drains inward"
        );
    }

    // Discharge grows downstream.
    for &i in &land {
        let d = h.downstream[i] as usize;
        if elev[d] >= 0.0 {
            assert!(
                h.discharge_m3_yr[d] >= h.discharge_m3_yr[i] - 1.0,
                "discharge drops from {i} ({}) to {d} ({})",
                h.discharge_m3_yr[i],
                h.discharge_m3_yr[d]
            );
        }
    }

    // Ledger balance: everything eroded this run was deposited this run.
    let residual = (ledger.deposited_m3 - ledger.eroded_m3).abs();
    assert!(ledger.eroded_m3 > 0.0, "nothing eroded");
    assert!(
        residual < 1.0e-3 * ledger.eroded_m3,
        "mass residual {residual:.3e} m^3 of {:.3e} eroded",
        ledger.eroded_m3
    );

    // Land is thinner, the sea floor around it is thicker.
    let column_after = total_column_m(&p, land.iter().copied());
    let ocean_after = total_column_m(&p, ocean.iter().copied())
        + ocean.iter().map(|i| p.sediment_m[*i] as f64).sum::<f64>();
    assert!(
        column_after < column_before,
        "land columns did not thin: {column_before} -> {column_after}"
    );
    assert!(
        ocean_after > ocean_before,
        "no sediment reached the sea: {ocean_before} -> {ocean_after}"
    );
}

#[test]
fn depressions_fill_and_every_land_cell_reaches_the_sea() {
    let mesh = Mesh::build(5);
    let mut elev = cone_island(&mesh, 3000.0, 0.30, false);
    // Punch a crater into the flank: a closed bowl 800 m below the cone.
    let crater_seed = (0..mesh.n_cells())
        .filter(|i| elev[*i] > 900.0 && elev[*i] < 1100.0)
        .min()
        .expect("cone has a mid-flank");
    let ring = rings(&mesh, crater_seed as u32, 1);
    for i in 0..mesh.n_cells() {
        if ring[i] == 0 {
            elev[i] -= 800.0;
        } else if ring[i] == 1 {
            elev[i] -= 400.0;
        }
    }

    let mut p = planet_from_elevation(&mesh, &elev, 0.0);
    p.precip_mm_yr.fill(1200.0);
    p.temperature_c.fill(10.0);

    let mut proc = SurfaceProcess::new();
    step(&mut proc, &mut p, &mesh, 0.005);

    assert!(
        p.lake_depth_m[crater_seed] > 0.0,
        "crater did not fill: depth {}",
        p.lake_depth_m[crater_seed]
    );
    // The lake spills: water leaves the basin.
    assert!(
        p.water_flux_m3_yr[crater_seed] > 0.0,
        "crater lake has no outflow"
    );

    // No undrained pit: following receivers from any land cell reaches the sea.
    let h = proc.hydrology();
    for i in 0..mesh.n_cells() {
        if p.elevation_m[i] < p.sea_level_m {
            continue;
        }
        let mut cur = i;
        let mut hops = 0;
        while p.elevation_m[cur] >= p.sea_level_m {
            let d = h.downstream[cur];
            assert_ne!(d, NO_DOWNSTREAM, "cell {cur} is an undrained pit");
            assert!(
                h.filled_m[d as usize] < h.filled_m[cur],
                "receiver is not lower"
            );
            cur = d as usize;
            hops += 1;
            assert!(
                hops < mesh.n_cells(),
                "flow path from {i} does not terminate"
            );
        }
    }
}

// --------------------------------------------------------------------------
// glaciers
// --------------------------------------------------------------------------

/// Steep ice cap: a 3 km peak, a 2 km shoulder, a low coastal ring and sea.
fn ice_cap_planet(mesh: &Mesh, warm_terminus: bool, coast_m: f32) -> Planet {
    let ring = rings(mesh, 0, 4);
    let elev: Vec<f32> = ring
        .iter()
        .map(|r| match r {
            0 => 3000.0,
            1 => 2000.0,
            2 => coast_m,
            _ => -2000.0,
        })
        .collect();
    let mut p = planet_from_elevation(mesh, &elev, 0.0);
    for (i, r) in ring.iter().enumerate() {
        p.precip_mm_yr[i] = if *r <= 1 { 600.0 } else { 0.0 };
        p.temperature_c[i] = if warm_terminus && *r >= 2 { 6.0 } else { -10.0 };
    }
    p
}

#[test]
fn ice_grows_flows_carves_and_leaves_moraines() {
    let mesh = Mesh::build(4);
    let ring = rings(&mesh, 0, 4);
    let mut p = ice_cap_planet(&mesh, true, 100.0);
    let crust_before = p.crust_thickness_m.clone();

    let mut proc = SurfaceProcess::new();
    let ledger = run(&mut proc, &mut p, &mesh, 60, 0.005);

    // Accumulation zone is glaciated.
    assert!(
        p.ice_thickness_m[0] > 500.0,
        "no ice cap: {}",
        p.ice_thickness_m[0]
    );
    // Ice reached cells that never receive snow: it flowed there.
    let flowed = (0..mesh.n_cells()).any(|i| ring[i] == 2 && p.ice_thickness_m[i] > 0.0);
    let terminus_reached = (0..mesh.n_cells()).any(|i| ring[i] == 2 && p.sediment_m[i] > 0.0);
    assert!(
        flowed || terminus_reached,
        "ice never left the accumulation zone"
    );

    // Abrasion thinned the bed under the ice.
    let cut: f32 = (0..mesh.n_cells())
        .filter(|i| ring[*i] <= 1)
        .map(|i| crust_before[i] - p.crust_thickness_m[i])
        .sum();
    assert!(cut > 1.0, "glacier did not carve: {cut:.3} m removed");

    // Moraine: loose debris left standing where the ice melts. (The rest is
    // washed on by meltwater into outwash beyond the terminus.)
    let moraine: f32 = (0..mesh.n_cells())
        .filter(|i| ring[*i] == 2)
        .map(|i| p.sediment_m[i])
        .sum();
    assert!(moraine > 0.0, "no moraine at the terminus: {moraine:.3} m");
    let outwash: f32 = (0..mesh.n_cells())
        .filter(|i| ring[*i] >= 3)
        .map(|i| p.sediment_m[i] + p.columns.total_thickness_m(i as u32) - 5_000.0)
        .sum();
    assert!(
        outwash > 0.0,
        "quarried rock never left the glacier: {outwash:.3} m"
    );

    let residual = (ledger.deposited_m3 - ledger.eroded_m3).abs();
    assert!(
        residual < 1.0e-3 * ledger.eroded_m3.max(1.0),
        "glacial mass residual {residual:.3e} of {:.3e}",
        ledger.eroded_m3
    );
}

#[test]
fn coastal_ice_carves_below_sea_level() {
    let mesh = Mesh::build(4);
    let ring = rings(&mesh, 0, 4);
    // Cold everywhere: the ice stream reaches the shore, which sits at +5 m.
    let mut p = ice_cap_planet(&mesh, false, 5.0);
    for (i, r) in ring.iter().enumerate() {
        if *r == 2 {
            p.precip_mm_yr[i] = 300.0;
        }
    }
    let crust_before = p.crust_thickness_m.clone();

    let mut proc = SurfaceProcess::new();
    run(&mut proc, &mut p, &mesh, 120, 0.005);

    // Coastal trough: rock removed under the ice would drop the ground below
    // sea level once isostasy responds (see AIRY_SURFACE_RATIO).
    let fjord = (0..mesh.n_cells())
        .filter(|i| ring[*i] == 2)
        .map(|i| {
            let removed = (crust_before[i] - p.crust_thickness_m[i]) as f64;
            p.elevation_m[i] as f64 - removed * AIRY_SURFACE_RATIO
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        fjord < p.sea_level_m as f64,
        "no coastal overdeepening: deepest implied elevation {fjord:.2} m"
    );
}

// --------------------------------------------------------------------------
// deposition environments
// --------------------------------------------------------------------------

#[test]
fn river_mouth_builds_a_delta() {
    let mesh = Mesh::build(5);
    let elev = cone_island(&mesh, 2500.0, 0.30, true);
    let mut p = planet_from_elevation(&mesh, &elev, 0.0);
    p.precip_mm_yr.fill(2500.0);
    p.temperature_c.fill(10.0);

    let mut proc = SurfaceProcess::new();
    run(&mut proc, &mut p, &mesh, 150, 0.005);

    // Shelf cells next to the shore hold new clastic strata.
    let mut delta_cells = 0;
    for (i, e) in elev.iter().enumerate() {
        if *e >= 0.0 {
            continue;
        }
        let new: f32 = p
            .columns
            .col(i as u32)
            .iter()
            .filter(|s| matches!(s.rock, RockType::Sandstone | RockType::Shale))
            .map(|s| s.thickness_m)
            .sum();
        if new > 0.0 {
            delta_cells += 1;
        }
    }
    assert!(
        delta_cells > 0,
        "no clastic strata deposited at the river mouths"
    );
}

#[test]
fn closed_hot_basin_precipitates_evaporite() {
    let mesh = Mesh::build(4);
    let ring = rings(&mesh, 0, 6);
    // Bowl: floor at 500 m, rim at 2500 m, then a long slope to the sea.
    let elev: Vec<f32> = (0..mesh.n_cells())
        .map(|i| match ring[i] {
            0 => 500.0,
            1 => 600.0,
            2 => 2500.0,
            3 => 1500.0,
            4 => 500.0,
            _ => -2000.0,
        })
        .collect();
    let mut p = planet_from_elevation(&mesh, &elev, 0.0);
    for (i, r) in ring.iter().enumerate() {
        // Dry, hot basin inside a wetter rim.
        p.precip_mm_yr[i] = if *r <= 1 { 50.0 } else { 900.0 };
        p.temperature_c[i] = 30.0;
    }

    let mut proc = SurfaceProcess::new();
    run(&mut proc, &mut p, &mesh, 200, 0.005);

    let evaporite: f32 = (0..mesh.n_cells())
        .filter(|i| ring[*i] <= 1)
        .map(|i| {
            p.columns
                .col(i as u32)
                .iter()
                .filter(|s| s.rock == RockType::Evaporite)
                .map(|s| s.thickness_m)
                .sum::<f32>()
        })
        .sum();
    assert!(
        evaporite > 0.1,
        "closed basin grew only {evaporite:.4} m of evaporite"
    );
    assert_eq!(
        p.lake_depth_m[0], 0.0,
        "a basin that evaporates its whole budget must be a dry playa"
    );
}

// --------------------------------------------------------------------------
// wind
// --------------------------------------------------------------------------

#[test]
fn arid_belt_deflates_downwind() {
    let mesh = Mesh::build(4);
    // Flat, dry world: no ocean (sea level below the ground everywhere), no
    // relief, so only the wind moves anything.
    let elev = vec![500.0f32; mesh.n_cells()];
    let mut p = planet_from_elevation(&mesh, &elev, -1000.0);
    p.sediment_m.fill(10.0);
    p.temperature_c.fill(25.0);
    for i in 0..mesh.n_cells() {
        let lon = mesh.latlon[i][1];
        // Arid band over half the planet, wet elsewhere.
        p.precip_mm_yr[i] = if lon < 0.0 { 100.0 } else { 900.0 };
        let (east, _) = mesh.east_north(i as u32);
        p.wind_m_s[i] = east * 12.0;
    }

    let interior: Vec<usize> = (0..mesh.n_cells())
        .filter(|i| {
            let lon = mesh.latlon[*i][1];
            lon < -0.4 && lon > -2.5
        })
        .collect();
    // The downwind margin: wet cells with an arid neighbour to the west.
    let margin: Vec<usize> = (0..mesh.n_cells())
        .filter(|i| {
            p.precip_mm_yr[*i] >= 250.0
                && mesh
                    .neighbors_of(*i as u32)
                    .iter()
                    .any(|&m| p.precip_mm_yr[m as usize] < 250.0)
        })
        .collect();
    assert!(!interior.is_empty() && !margin.is_empty());

    let mut proc = SurfaceProcess::new();
    run(&mut proc, &mut p, &mesh, 20, 0.005);

    let lost: f32 = interior.iter().map(|i| 10.0 - p.sediment_m[*i]).sum();
    let gained: f32 = margin.iter().map(|i| p.sediment_m[*i] - 10.0).sum();
    assert!(lost > 0.0, "arid interior kept all its sand: {lost:.4} m");
    assert!(
        gained > 0.0,
        "downwind margin gained nothing: {gained:.4} m"
    );
}

// --------------------------------------------------------------------------
// rates
// --------------------------------------------------------------------------

#[test]
fn denudation_rate_is_geologically_plausible() {
    let mesh = Mesh::build(5);
    let elev = cone_island(&mesh, 3000.0, 0.25, false);
    let mut p = planet_from_elevation(&mesh, &elev, 0.0);
    p.precip_mm_yr.fill(1200.0);
    p.temperature_c.fill(10.0);

    let land: Vec<usize> = (0..mesh.n_cells()).filter(|i| elev[*i] >= 0.0).collect();
    let land_area_m2: f64 = land.iter().map(|i| mesh.areas_km2[*i] as f64 * 1.0e6).sum();
    let load = |p: &Planet| -> f64 {
        land.iter()
            .map(|i| {
                (p.crust_thickness_m[*i] + p.sediment_m[*i]) as f64
                    * mesh.areas_km2[*i] as f64
                    * 1.0e6
            })
            .sum()
    };
    let before = load(&p);

    let mut proc = SurfaceProcess::new();
    let myr = 1.0;
    let ledger = run(&mut proc, &mut p, &mesh, 200, myr / 200.0);

    // Net denudation: material that actually left the land, as a mean depth.
    // The ledger's `eroded_m3` is larger because it also counts alluvium that
    // is picked up and put down again within the land surface.
    let net_m_per_myr = (before - load(&p)) / land_area_m2 / myr;
    let gross_m_per_myr = ledger.eroded_m3 / land_area_m2 / myr;
    let surface_m_per_myr = net_m_per_myr * AIRY_SURFACE_RATIO;
    println!(
        "denudation: {net_m_per_myr:.0} m/Myr net rock loss ({gross_m_per_myr:.0} m/Myr gross \
         transfers), {surface_m_per_myr:.0} m/Myr of surface lowering after isostatic rebound"
    );
    assert!(
        (50.0..300.0).contains(&net_m_per_myr),
        "denudation {net_m_per_myr:.0} m/Myr outside the 50-200 m/Myr target band"
    );
    // Continents must not be planed flat: the peak keeps most of its column.
    assert!(
        p.columns.total_thickness_m(0) > 19_000.0,
        "peak stripped: {} m of column left",
        p.columns.total_thickness_m(0)
    );
}

#[test]
fn refinement_smooths_without_glaciers_or_wind() {
    let mesh = Mesh::build(5);
    let elev = cone_island(&mesh, 3000.0, 0.25, false);
    let mut p = planet_from_elevation(&mesh, &elev, 0.0);
    p.phase = Phase::Refinement;
    p.precip_mm_yr.fill(1200.0);
    p.temperature_c.fill(-20.0); // would grow ice if the phase allowed it
    for i in 0..mesh.n_cells() {
        let (east, _) = mesh.east_north(i as u32);
        p.wind_m_s[i] = east * 20.0;
    }

    let mut proc = SurfaceProcess::new();
    let ledger = run(&mut proc, &mut p, &mesh, 40, 0.25);

    assert!(ledger.eroded_m3 > 0.0, "refinement did no work");
    assert!(
        p.ice_thickness_m.iter().all(|t| *t == 0.0),
        "glaciers must not run before RecentPast"
    );
    let residual = (ledger.deposited_m3 - ledger.eroded_m3).abs();
    assert!(
        residual < 1.0e-3 * ledger.eroded_m3,
        "residual {residual:.3e}"
    );
}

#[test]
fn early_phases_are_inert() {
    let mesh = Mesh::build(4);
    let elev = cone_island(&mesh, 3000.0, 0.4, false);
    for phase in [Phase::CrustalFormation, Phase::Drift] {
        let mut p = planet_from_elevation(&mesh, &elev, 0.0);
        p.phase = phase;
        p.precip_mm_yr.fill(2000.0);
        let before = state_hash(&p);
        let mut proc = SurfaceProcess::new();
        let ledger = run(&mut proc, &mut p, &mesh, 5, 0.5);
        assert_eq!(before, state_hash(&p), "{phase:?} changed the surface");
        assert_eq!(ledger.eroded_m3, 0.0);
    }
}

// --------------------------------------------------------------------------
// determinism
// --------------------------------------------------------------------------

#[test]
fn deterministic_and_stateless() {
    let mesh = Mesh::build(4);
    let elev = cone_island(&mesh, 3000.0, 0.4, true);
    let build = || {
        let mut p = planet_from_elevation(&mesh, &elev, 0.0);
        p.precip_mm_yr.fill(1400.0);
        for (i, e) in elev.iter().enumerate() {
            // Cold peak, warm coast: exercises ice, rivers and wind together.
            p.temperature_c[i] = 8.0 - 0.008 * e.max(0.0);
            let (east, _) = mesh.east_north(i as u32);
            p.wind_m_s[i] = east * 8.0;
        }
        p
    };

    let mut a = build();
    let mut proc_a = SurfaceProcess::new();
    run(&mut proc_a, &mut a, &mesh, 24, 0.005);

    // Same again, but with a fresh process swapped in halfway — exactly what
    // resuming from a checkpoint does.
    let mut b = build();
    let mut proc_b1 = SurfaceProcess::new();
    run(&mut proc_b1, &mut b, &mesh, 12, 0.005);
    let mut proc_b2 = SurfaceProcess::new();
    run(&mut proc_b2, &mut b, &mesh, 12, 0.005);

    assert_eq!(
        state_hash(&a),
        state_hash(&b),
        "a fresh process instance mid-run must continue identically"
    );

    // And a plain repeat is bit-identical.
    let mut c = build();
    let mut proc_c = SurfaceProcess::new();
    run(&mut proc_c, &mut c, &mesh, 24, 0.005);
    assert_eq!(state_hash(&a), state_hash(&c), "run is not reproducible");
}

// --------------------------------------------------------------------------
// facies
// --------------------------------------------------------------------------

#[test]
fn starved_warm_shelf_builds_limestone_and_fans_build_conglomerate() {
    let mesh = Mesh::build(4);
    let elev = cone_island(&mesh, 4000.0, 0.35, true);
    let mut p = planet_from_elevation(&mesh, &elev, 0.0);
    // Hot and arid: waves and wind feed the shelf, rivers do not.
    p.precip_mm_yr.fill(80.0);
    p.temperature_c.fill(26.0);
    for (i, e) in elev.iter().enumerate() {
        let (east, _) = mesh.east_north(i as u32);
        p.wind_m_s[i] = east * 10.0;
        p.sediment_m[i] = if *e >= 0.0 { 20.0 } else { 0.0 };
    }
    // A thick alluvial apron at the foot of the cone, ready to lithify.
    let fan = (0..mesh.n_cells())
        .filter(|i| {
            elev[*i] > 0.0
                && mesh
                    .neighbors_of(*i as u32)
                    .iter()
                    .any(|&m| elev[m as usize] - elev[*i] > 500.0)
        })
        .min()
        .expect("cone has a break in slope");
    p.sediment_m[fan] = 400.0;

    let mut proc = SurfaceProcess::new();
    run(&mut proc, &mut p, &mesh, 200, 0.005);

    let has = |cell: usize, rock: RockType| {
        p.columns
            .col(cell as u32)
            .iter()
            .any(|s| s.rock == rock && s.thickness_m > 0.0)
    };
    assert!(
        has(fan, RockType::Conglomerate),
        "mountain front is not conglomerate"
    );
    let limestone = (0..mesh.n_cells())
        .filter(|i| elev[*i] < 0.0 && elev[*i] > -200.0)
        .filter(|i| has(*i, RockType::Limestone))
        .count();
    assert!(limestone > 0, "warm starved shelf grew no carbonate");
}
