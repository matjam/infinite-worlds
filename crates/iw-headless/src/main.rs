//! Headless world generation, resume and map export for Infinite Worlds.
//!
//! `iw-headless gen --seed 42 --level 6 --out /tmp/planet42` runs the full
//! phase schedule against the real process crates, checkpointing at each
//! phase boundary, capturing history snapshots for the time scrubber, and
//! writing a `summary.json` golden fingerprint. This binary is also the
//! integration test WP12 builds on: identical `(seed, config)` must produce
//! an identical summary.

mod cli;
mod processes;
mod progress;
mod summary;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use iw_core::{CheckpointStore, Phase, PlanetConfig, Process, ProgressSink};
use iw_mesh::Mesh;
use iw_sim::Simulation;
use iw_store_postcard::{FileStore, HistoryStore};

use cli::{parse_durations, parse_phase, Cli, Command, GenArgs, ResumeArgs};
use processes::build_processes;
use progress::CliProgress;
use summary::{compute_summary, write_summary};

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Gen(args) => run_gen(args),
        Command::Resume(args) => run_resume(args),
    }
}

fn build_config(args: &GenArgs) -> anyhow::Result<PlanetConfig> {
    let mut config = PlanetConfig {
        seed: args.seed,
        subdivision_level: args.level,
        water_budget: args.water,
        ..PlanetConfig::default()
    };
    if let Some(d) = &args.durations {
        config.phase_durations_myr = parse_durations(d)?;
    }
    if let Some(cap) = args.history_cap {
        config.history_cap_bytes = cap;
    }
    if let Some(only) = &args.phases_only {
        let keep = only
            .split(',')
            .map(parse_phase)
            .collect::<anyhow::Result<Vec<Phase>>>()?;
        for (i, phase) in Phase::ALL.iter().enumerate() {
            if !keep.contains(phase) {
                config.phase_durations_myr[i] = 0.0;
            }
        }
    }
    config.sanitize();
    Ok(config)
}

fn run_gen(args: GenArgs) -> anyhow::Result<()> {
    let config = build_config(&args)?;
    let mesh = Arc::new(Mesh::build(config.subdivision_level));
    let store = Arc::new(FileStore::new(args.out.clone())?);
    let history = HistoryStore::new(args.out.clone(), config.history_cap_bytes)?;
    let progress: Arc<dyn ProgressSink> = Arc::new(CliProgress::default());
    let processes = build_processes(&args.skip);

    let mut sim = Simulation::new(config, Arc::clone(&mesh), processes, store, progress);

    let start = Instant::now();
    run_with_history(&mut sim, &history)?;
    let runtime_secs = start.elapsed().as_secs_f64();

    finish(sim.planet(), &mesh, runtime_secs, &args.out, args.maps)
}

fn run_resume(args: ResumeArgs) -> anyhow::Result<()> {
    let phase = parse_phase(&args.phase)?;
    let Some(prev) = Phase::ALL.get(phase.index().wrapping_sub(1)).copied() else {
        anyhow::bail!("cannot resume into {phase:?}: it is the first phase; use `gen` instead");
    };

    let store = Arc::new(FileStore::new(args.from.clone())?);
    let boundary_tag = iw_sim::phase_tag(prev);
    let boundary = store
        .load(&boundary_tag)
        .map_err(|e| anyhow::anyhow!("loading checkpoint {boundary_tag}: {e:#}"))?;
    let config = boundary.config.clone();
    let mesh = Arc::new(Mesh::build(config.subdivision_level));
    let history = HistoryStore::new(args.from.clone(), config.history_cap_bytes)?;
    let progress: Arc<dyn ProgressSink> = Arc::new(CliProgress::default());
    let processes: Vec<Box<dyn Process>> = build_processes(&[]);

    let mut sim = Simulation::new(
        config.clone(),
        Arc::clone(&mesh),
        processes,
        store,
        progress,
    );
    sim.rerun_from_phase(phase, config)?;

    let start = Instant::now();
    run_with_history(&mut sim, &history)?;
    let runtime_secs = start.elapsed().as_secs_f64();

    finish(sim.planet(), &mesh, runtime_secs, &args.from, args.maps)
}

/// Step to completion, pushing a history snapshot every time the simulation
/// publishes a new `PlanetView` (the publish throttle inside `Simulation`
/// already bounds how often that happens; `HistoryStore`'s own cap keeps
/// total size bounded regardless).
fn run_with_history(sim: &mut Simulation, history: &HistoryStore) -> anyhow::Result<()> {
    let mut last_version = 0u64;
    while sim.step_once() {
        if let Some(view) = sim.latest_view() {
            if view.version != last_version {
                history.push(&view)?;
                last_version = view.version;
            }
        }
    }
    // Final state is always worth keeping even if it fell inside the
    // publish throttle window.
    if let Some(view) = sim.latest_view() {
        if view.version != last_version {
            history.push(&view)?;
        }
    }
    Ok(())
}

fn finish(
    planet: &iw_core::Planet,
    mesh: &Mesh,
    runtime_secs: f64,
    out_dir: &Path,
    maps: bool,
) -> anyhow::Result<()> {
    let summary = compute_summary(planet, mesh, runtime_secs);
    write_summary(out_dir, &summary)?;
    if maps {
        let exporter = iw_export_png::PngExporter;
        iw_core::MapExporter::export(&exporter, planet, mesh, out_dir)?;
    }
    Ok(())
}
