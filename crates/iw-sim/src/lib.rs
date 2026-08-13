//! Orchestration for Infinite Worlds: the phase schedule, the process loop,
//! checkpointing, snapshot publishing and progress narration.
//!
//! Two entry points:
//!
//! - [`Simulation`] — synchronous. Owns the [`iw_core::Planet`], the mesh and
//!   the process list; [`Simulation::run_headless`] runs the whole timeline.
//! - [`SimHandle`] — the same simulation on a worker thread, driven by
//!   [`SimCommand`]s and observed through [`SimHandle::latest_view`],
//!   [`SimHandle::status`] and [`SimHandle::drain_events`].
//!
//! The crate depends only on `iw-core` and `iw-mesh`: processes arrive as
//! `Box<dyn Process>`, persistence as `dyn CheckpointStore`, progress as
//! `dyn ProgressSink`, and even mesh construction as an injected closure. See
//! DESIGN.md §3 and §5.
//!
//! ```no_run
//! use std::sync::Arc;
//! use iw_core::{NullProgress, PlanetConfig};
//! use iw_sim::{test_util::MemoryStore, Simulation};
//!
//! let config = PlanetConfig::default();
//! let mesh = Arc::new(iw_mesh::Mesh::build(config.subdivision_level));
//! let mut sim = Simulation::new(
//!     config,
//!     mesh,
//!     vec![], // real processes go here
//!     Arc::new(MemoryStore::new()),
//!     Arc::new(NullProgress),
//! );
//! sim.run_headless();
//! let view = sim.latest_view().expect("a snapshot was published");
//! println!("{} Myr, sea level {} m", view.time_myr, view.sea_level_m);
//! ```

#![warn(missing_docs)]

pub mod handle;
pub mod narrator;
mod retess;
pub mod sim;

#[doc(hidden)]
pub mod test_util;

pub use handle::{MeshBuilder, SimCommand, SimHandle, SimState, SimStatus};
pub use narrator::{Narrator, EMBEDDED_SNARK};
pub use sim::{phase_slug, phase_tag, Simulation};
