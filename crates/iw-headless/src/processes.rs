//! Wires up the real process crates in the fixed order the sim requires
//! (tectonics, geology, climate, surface, biomes — see
//! `IMPLEMENTATION_PLAN.md` §3 and `Planet`'s field-ownership table).

use std::collections::HashSet;

use iw_core::Process;

/// Build the full process pipeline, skipping any name present in `skip`
/// (case-insensitive: tectonics, geology, climate, surface, biomes).
///
/// `skip` exists as an escape hatch for a process crate that isn't ready yet;
/// normal runs pass an empty slice.
pub fn build_processes(skip: &[String]) -> Vec<Box<dyn Process>> {
    let skip: HashSet<String> = skip.iter().map(|s| s.to_ascii_lowercase()).collect();
    let mut processes: Vec<Box<dyn Process>> = Vec::new();
    if !skip.contains("tectonics") {
        processes.push(Box::new(iw_tectonics::TectonicsProcess::default()));
    }
    if !skip.contains("geology") {
        processes.push(Box::new(iw_geology::GeologyProcess::default()));
    }
    if !skip.contains("climate") {
        processes.push(Box::new(iw_climate::ClimateProcess::default()));
    }
    if !skip.contains("surface") {
        processes.push(Box::new(iw_surface::SurfaceProcess::default()));
    }
    if !skip.contains("biomes") {
        // `BiomeProcess` is a stateless unit struct; construct it directly
        // rather than through `Default` (clippy::default_constructed_unit_structs).
        processes.push(Box::new(iw_biomes::BiomeProcess));
    }
    processes
}
