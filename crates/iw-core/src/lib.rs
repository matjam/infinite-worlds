//! Shared domain types and ports for Infinite Worlds.
//!
//! This crate is the contract every process crate and adapter codes against.
//! It holds no algorithms beyond small invariant-preserving operations on the
//! data structures themselves (strata stacking, config arithmetic).

pub mod biome;
pub mod config;
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
pub use rock::{MetamorphicGrade, RockType, Stratum, StrataColumns, MAX_STRATA};
pub use view::{PlanetView, ViewCells};
