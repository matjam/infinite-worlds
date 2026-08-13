//! Command-line surface: argument parsing and small textual parsers
//! (durations, phase names) that turn CLI strings into [`iw_core`] types.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use iw_core::Phase;

/// Headless world generation and export for Infinite Worlds.
#[derive(Parser, Debug)]
#[command(name = "iw-headless")]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Generate a new planet end-to-end: run the full phase schedule, write
    /// checkpoints and a `summary.json`.
    Gen(GenArgs),
    /// Resume a previous run from the checkpoint written at a phase
    /// boundary, and run it to completion.
    Resume(ResumeArgs),
}

/// Arguments for `gen`.
#[derive(Args, Debug)]
pub struct GenArgs {
    /// RNG seed.
    #[arg(long)]
    pub seed: u64,
    /// Goldberg subdivision level (clamped to 4..=10).
    #[arg(long)]
    pub level: u8,
    /// Output directory for checkpoints, history and (optionally) maps.
    #[arg(long)]
    pub out: PathBuf,
    /// Surface water budget, in Earth-ocean-volume units.
    #[arg(long, default_value_t = 1.0)]
    pub water: f64,
    /// Comma-separated phase durations in Myr: crust,drift,refine,recent.
    /// Defaults to [`iw_core::PlanetConfig::default`]'s schedule.
    #[arg(long)]
    pub durations: Option<String>,
    /// Only run these phases (others get a zero duration and are skipped).
    /// Comma-separated: crust, drift, refine, recent.
    #[arg(long = "phases-only")]
    pub phases_only: Option<String>,
    /// Export elevation/biome/top-rock equirectangular PNGs after the run.
    #[arg(long)]
    pub maps: bool,
    /// Disk budget for history snapshots, bytes. Defaults to
    /// [`iw_core::PlanetConfig::default`]'s 2 GiB.
    #[arg(long = "history-cap")]
    pub history_cap: Option<u64>,
    /// Process names to skip: tectonics, geology, climate, surface, biomes.
    /// For working around a process crate that isn't ready yet.
    #[arg(long, value_delimiter = ',')]
    pub skip: Vec<String>,
}

/// Arguments for `resume`.
#[derive(Args, Debug)]
pub struct ResumeArgs {
    /// Planet directory written by a previous `gen` or `resume` run.
    #[arg(long)]
    pub from: PathBuf,
    /// Phase to resume into: crust, drift, refine, recent. Loads the
    /// checkpoint written when the *previous* phase completed.
    #[arg(long)]
    pub phase: String,
    /// Export elevation/biome/top-rock equirectangular PNGs after the run.
    #[arg(long)]
    pub maps: bool,
}

/// Parse a phase name token (case-insensitive, several spellings accepted).
pub fn parse_phase(token: &str) -> anyhow::Result<Phase> {
    match token.trim().to_ascii_lowercase().as_str() {
        "crust" | "crustal" | "crustalformation" | "crustal_formation" => {
            Ok(Phase::CrustalFormation)
        }
        "drift" => Ok(Phase::Drift),
        "refine" | "refinement" => Ok(Phase::Refinement),
        "recent" | "recentpast" | "recent_past" => Ok(Phase::RecentPast),
        other => {
            anyhow::bail!("unknown phase '{other}'; expected one of: crust, drift, refine, recent")
        }
    }
}

/// Parse `--durations crust,drift,refine,recent` (Myr) into the
/// `[f64; 4]` shape `PlanetConfig::phase_durations_myr` expects.
pub fn parse_durations(s: &str) -> anyhow::Result<[f64; 4]> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("parsing --durations '{s}'"))?;
    anyhow::ensure!(
        parts.len() == 4,
        "--durations needs 4 comma-separated values (crust,drift,refine,recent), got {}",
        parts.len()
    );
    Ok([parts[0], parts[1], parts[2], parts[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_tokens() {
        assert_eq!(parse_phase("crust").unwrap(), Phase::CrustalFormation);
        assert_eq!(parse_phase("Drift").unwrap(), Phase::Drift);
        assert_eq!(parse_phase("refinement").unwrap(), Phase::Refinement);
        assert_eq!(parse_phase("RECENT_PAST").unwrap(), Phase::RecentPast);
        assert!(parse_phase("nonsense").is_err());
    }

    #[test]
    fn durations_parse() {
        assert_eq!(
            parse_durations("200,200,75,2").unwrap(),
            [200.0, 200.0, 75.0, 2.0]
        );
        assert!(parse_durations("1,2,3").is_err());
        assert!(parse_durations("a,b,c,d").is_err());
    }
}
