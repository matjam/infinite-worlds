//! Vulkan renderer for Infinite Worlds (DESIGN.md §9, §11).
//!
//! The crate is an adapter: it consumes a [`iw_mesh::Mesh`] for geometry and
//! flat per-cell arrays for data, and knows nothing about the simulation.
//!
//! Layout:
//! * [`camera`], [`mercator`], [`cull`] and [`beauty`] — pure math, unit
//!   tested, no GPU.
//! * [`gpu`], [`swapchain`], [`buffer`] — Vulkan bring-up.
//! * [`globe`] — geometry, per-cell storage buffers, globe/sky/cloud pipelines.
//! * [`renderer`] — the frame loop.
//! * [`ui`] — the egui overlay.
//!
//! Conventions used throughout:
//! * **Reverse-Z**: the depth buffer is `D32_SFLOAT`, cleared to `0.0`, and the
//!   depth compare op is `GREATER`. Projections are built accordingly.
//! * **World units are kilometres**; elevations are metres.
//! * `+z` is the north pole, longitude 0 at `+x` increasing toward `+y`.

pub mod beauty;
pub mod buffer;
pub mod camera;
pub mod cull;
pub mod globe;
pub mod gpu;
pub mod mercator;
pub mod renderer;
pub mod swapchain;
pub mod ui;

pub use beauty::{SunMode, SunSettings};
pub use camera::{Camera, MoveInput, ViewMode};
pub use globe::{CellShade, DrawStats, GlobeParams, SurfaceKind, FRAMES_IN_FLIGHT};
pub use renderer::{EguiFrame, Renderer};
pub use ui::{Ui, UiOutput, UiState};

// Re-exported so downstream crates cannot drift to a different version of the
// windowing/UI stack than the renderer was built against.
pub use egui;
pub use egui_winit::winit;
