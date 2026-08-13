//! Shared domain types and ports for Infinite Worlds.
//!
//! This crate is the contract every process crate and adapter codes against.
//! It holds no algorithms beyond small invariant-preserving operations on the
//! data structures themselves (strata stacking, config arithmetic).
//!
//! # What lives here
//!
//! - [`Planet`] — the struct-of-arrays simulation state (one `Vec` per field,
//!   indexed by cell id), plus [`PlanetConfig`] (every user-facing tunable,
//!   see DESIGN.md §9.1) and [`Phase`] (the four simulation eras).
//! - [`StrataColumns`] / [`Stratum`] / [`RockType`] / [`MetamorphicGrade`] —
//!   the per-cell rock record (DESIGN.md §6).
//! - [`Biome`] — the 14 WWF terrestrial biomes plus water/ice markers.
//! - [`Process`] — the trait every simulation subsystem implements, and
//!   [`StepCtx`] / [`MassLedger`] — the per-step context (deterministic RNG
//!   stream, mass-conservation accounting) handed to it.
//! - Ports consumed by adapters: [`CheckpointStore`], [`MapExporter`],
//!   [`ProgressSink`] (+ [`ProgressEvent`]).
//! - [`PlanetView`] — a cheap-to-clone, `Arc`-backed snapshot of the
//!   render-relevant fields, published by `iw-sim` for observers.
//!
//! `iw-mesh` (the Goldberg-sphere geometry, cell adjacency, spherical math)
//! sits below this crate and has no dependency on it.

pub mod biome;
pub mod config;
pub mod noise;
pub mod planet;
pub mod ports;
pub mod process;
pub mod rock;
pub mod view;

pub use biome::Biome;
pub use config::{Phase, PlanetConfig};
pub use planet::{CrustType, Hotspot, Planet, Plate};
pub use ports::{CheckpointStore, MapExporter, NullProgress, ProgressEvent, ProgressSink};
pub use process::{rng_for, MassLedger, Process, StepCtx};
pub use rock::{MetamorphicGrade, RockType, StrataColumns, Stratum, MAX_STRATA};
pub use view::{PlanetView, ViewCells};
