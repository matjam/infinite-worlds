//! Shared harness: drive `TectonicsProcess` directly, without `iw-sim`.

#![allow(dead_code)]

use std::sync::Arc;

use iw_core::planet::cell_flags;
use iw_core::{
    rng_for, CrustType, MassLedger, NullProgress, Phase, Planet, PlanetConfig, Process, StepCtx,
};
use iw_mesh::Mesh;
use iw_tectonics::TectonicsProcess;

/// A planet plus the bits `Process::step` needs, driven by hand so the tests
/// stay independent of the orchestrator.
pub struct Harness {
    pub planet: Planet,
    pub mesh: Arc<Mesh>,
    pub ledger: MassLedger,
    /// Ledger totals accumulated over the whole run.
    pub total: MassLedger,
    progress: NullProgress,
}

pub fn test_config(level: u8) -> PlanetConfig {
    let mut c = PlanetConfig {
        subdivision_level: level,
        seed: 42,
        ..PlanetConfig::default()
    };
    c.sanitize();
    c
}

impl Harness {
    pub fn new(config: PlanetConfig, mesh: Arc<Mesh>) -> Harness {
        let planet = Planet::new(config, mesh.n_cells());
        Harness {
            planet,
            mesh,
            ledger: MassLedger::default(),
            total: MassLedger::default(),
            progress: NullProgress,
        }
    }

    pub fn level(level: u8) -> Harness {
        let mesh = Arc::new(Mesh::build(level));
        Harness::new(test_config(level), mesh)
    }

    /// Run `n` steps of `phase` through `process`, exactly as `iw-sim` would.
    pub fn run(&mut self, process: &mut TectonicsProcess, phase: Phase, n: u64) {
        self.planet.phase = phase;
        let dt = self.planet.config.dt_myr(phase);
        for _ in 0..n {
            self.ledger.reset();
            let mut ctx = StepCtx {
                rng: rng_for(
                    self.planet.config.seed,
                    process.name(),
                    self.planet.step_index,
                ),
                progress: &self.progress,
                ledger: &mut self.ledger,
            };
            process.step(&mut self.planet, &self.mesh, dt, &mut ctx);
            self.total.created_m3 += self.ledger.created_m3;
            self.total.subducted_m3 += self.ledger.subducted_m3;
            self.total.deposited_m3 += self.ledger.deposited_m3;
            self.total.eroded_m3 += self.ledger.eroded_m3;
            self.planet.step_index += 1;
            self.planet.time_myr += dt;
        }
    }

    /// Run the whole of Phase 1 at its configured length.
    pub fn run_crustal_formation(&mut self, process: &mut TectonicsProcess) {
        let steps = self.planet.config.steps_in(Phase::CrustalFormation);
        self.run(process, Phase::CrustalFormation, steps);
    }

    pub fn continental_cells(&self) -> usize {
        self.planet
            .crust_type
            .iter()
            .filter(|t| **t == CrustType::Continental)
            .count()
    }

    pub fn flagged(&self, bit: u8) -> usize {
        self.planet
            .tectonic_flags
            .iter()
            .filter(|f| **f & bit != 0)
            .count()
    }

    /// Surface speed of each cell's plate, cm/yr.
    pub fn speeds_cm_yr(&self) -> Vec<f32> {
        (0..self.planet.n_cells())
            .filter_map(|c| {
                let p = self.planet.plates.get(self.planet.plate_id[c] as usize)?;
                Some(p.velocity_m_yr(self.mesh.centers[c]).length() * 100.0)
            })
            .collect()
    }

    /// A cheap order-sensitive digest of everything tectonics owns.
    pub fn digest(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, p) in self.planet.plate_id.iter().enumerate() {
            out.extend_from_slice(&p.to_le_bytes());
            out.extend_from_slice(&self.planet.crust_thickness_m[i].to_le_bytes());
            out.extend_from_slice(&self.planet.crust_density_kg_m3[i].to_le_bytes());
            out.extend_from_slice(&self.planet.crust_age_myr[i].to_le_bytes());
            out.push(self.planet.tectonic_flags[i]);
            out.push(self.planet.crust_type[i] as u8);
            out.push(self.planet.columns.col(i as u32).len() as u8);
        }
        for p in &self.planet.plates {
            out.extend_from_slice(&p.omega_rad_myr.to_le_bytes());
            out.extend_from_slice(&p.euler_pole.x.to_le_bytes());
            out.extend_from_slice(&p.euler_pole.y.to_le_bytes());
            out.extend_from_slice(&p.euler_pole.z.to_le_bytes());
        }
        out
    }
}

/// Every plate is non-empty and forms one connected region on the cell graph.
pub fn plates_are_partition(planet: &Planet, mesh: &Mesh) -> Result<(), String> {
    let np = planet.plates.len();
    if np == 0 {
        return Err("no plates".into());
    }
    let mut seen = vec![0usize; np];
    for (c, p) in planet.plate_id.iter().enumerate() {
        if *p as usize >= np {
            return Err(format!("cell {c} has out-of-range plate {p}"));
        }
        seen[*p as usize] += 1;
    }
    for (p, n) in seen.iter().enumerate() {
        if *n == 0 {
            return Err(format!("plate {p} is empty"));
        }
    }
    // Flood fill from one cell of each plate; every member must be reached.
    let mut visited = vec![false; planet.n_cells()];
    let mut reached = vec![0usize; np];
    for p in 0..np {
        let Some(start) = (0..planet.n_cells()).find(|c| planet.plate_id[*c] as usize == p) else {
            continue;
        };
        let mut stack = vec![start as u32];
        visited[start] = true;
        while let Some(x) = stack.pop() {
            reached[p] += 1;
            for &m in mesh.neighbors_of(x) {
                if !visited[m as usize] && planet.plate_id[m as usize] as usize == p {
                    visited[m as usize] = true;
                    stack.push(m);
                }
            }
        }
        if reached[p] != seen[p] {
            return Err(format!(
                "plate {p} is not contiguous: {} of {} cells reachable",
                reached[p], seen[p]
            ));
        }
    }
    Ok(())
}

pub const ALL_FLAGS: [(&str, u8); 7] = [
    ("SUBDUCTING", cell_flags::SUBDUCTING),
    ("ARC", cell_flags::ARC),
    ("COLLISION", cell_flags::COLLISION),
    ("RIFT", cell_flags::RIFT),
    ("HOTSPOT", cell_flags::HOTSPOT),
    ("TRANSFORM", cell_flags::TRANSFORM),
    ("SUTURE", cell_flags::SUTURE),
];
