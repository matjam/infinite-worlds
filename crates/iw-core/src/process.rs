use iw_mesh::Mesh;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use crate::ports::ProgressSink;
use crate::Planet;

/// Volume accounting for crust and sediment, m^3 per step. The classic failure
/// mode of erosion sims is silently creating or destroying rock; every process
/// reports its moves here and `iw-sim` debug-asserts the ledger balances.
#[derive(Debug, Default, Clone, Copy)]
pub struct MassLedger {
    pub deposited_m3: f64,
    pub eroded_m3: f64,
    pub subducted_m3: f64,
    pub created_m3: f64,
}

impl MassLedger {
    pub fn reset(&mut self) {
        *self = MassLedger::default();
    }

    /// Net material that appeared from / vanished into nowhere. Sources
    /// (ridge crust creation, volcanism) count as `created`; sinks
    /// (subduction) as `subducted`. Erosion must equal deposition once
    /// transport settles, so the residual is deposited - eroded.
    pub fn residual_m3(&self) -> f64 {
        self.deposited_m3 - self.eroded_m3
    }
}

/// Per-step context handed to every process.
pub struct StepCtx<'a> {
    /// Deterministic RNG stream unique to (seed, process, step). Derived
    /// statelessly so checkpoint resume reproduces bit-identical results.
    pub rng: Pcg64Mcg,
    pub progress: &'a dyn ProgressSink,
    pub ledger: &'a mut MassLedger,
}

/// One simulated subsystem (tectonics, geology, climate, surface, biomes).
/// Processes run in a fixed order each step and communicate only through
/// `Planet` fields (see field-ownership table on [`Planet`]).
pub trait Process: Send {
    fn name(&self) -> &'static str;

    /// Advance the planet by `dt_myr`. Must be deterministic given
    /// (planet state, mesh, dt, ctx.rng).
    fn step(&mut self, planet: &mut Planet, mesh: &Mesh, dt_myr: f64, ctx: &mut StepCtx);
}

/// Stable stream derivation: FNV-1a over the process name, mixed with seed and
/// step. Do not replace with `DefaultHasher` (not stable across releases).
pub fn rng_for(seed: u64, process_name: &str, step_index: u64) -> Pcg64Mcg {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in process_name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mixed = seed
        .wrapping_add(h.rotate_left(17))
        .wrapping_add(step_index.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    Pcg64Mcg::seed_from_u64(mixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn streams_differ_and_reproduce() {
        let mut a1 = rng_for(42, "tectonics", 7);
        let mut a2 = rng_for(42, "tectonics", 7);
        let mut b = rng_for(42, "geology", 7);
        let mut c = rng_for(42, "tectonics", 8);
        let x = a1.next_u64();
        assert_eq!(x, a2.next_u64());
        assert_ne!(x, b.next_u64());
        assert_ne!(x, c.next_u64());
    }
}
