//! egui overlay: the small always-on HUD (FPS, altitude, exaggeration, view
//! mode) plus a placeholder layer selector that WP10 fills in.

use egui_winit::winit::window::Window;

use crate::camera::ViewMode;

/// Data layers offered by the placeholder selector.
pub const LAYERS: [&str; 8] = [
    "Beauty (placeholder)",
    "Elevation",
    "Biomes",
    "Plates",
    "Crust age",
    "Top rock",
    "Temperature",
    "Precipitation",
];

/// Values the overlay reads and writes each frame.
#[derive(Debug, Clone)]
pub struct UiState {
    pub fps: f32,
    pub frame_ms: f32,
    pub altitude_km: f32,
    /// Vertical exaggeration the user asked for (1..200).
    pub exaggeration: f32,
    /// Exaggeration actually applied after the near-surface ease.
    pub effective_exaggeration: f32,
    pub mode: ViewMode,
    pub layer: usize,
    pub n_cells: usize,
    pub chunks_drawn: usize,
    pub chunks_total: usize,
    pub triangles_drawn: usize,
    pub device_name: String,
    /// Set by the overlay when the mode button is clicked.
    pub mode_toggled: bool,
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            fps: 0.0,
            frame_ms: 0.0,
            altitude_km: 0.0,
            exaggeration: 30.0,
            effective_exaggeration: 30.0,
            mode: ViewMode::Globe,
            layer: 0,
            n_cells: 0,
            chunks_drawn: 0,
            chunks_total: 0,
            triangles_drawn: 0,
            device_name: String::new(),
            mode_toggled: false,
        }
    }
}

/// egui context plus its winit integration.
pub struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
}

/// Everything the renderer needs from one egui frame.
pub struct UiOutput {
    pub textures_delta: egui::TexturesDelta,
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub pixels_per_point: f32,
    /// True when egui wants the pointer, so the camera should ignore it.
    pub wants_pointer: bool,
    /// True when egui wants the keyboard.
    pub wants_keyboard: bool,
}

impl Ui {
    /// Create the overlay for `window`.
    pub fn new(window: &Window) -> Ui {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(2048),
        );
        Ui { ctx, state }
    }

    /// Feed a winit window event to egui. Returns true when egui consumed it.
    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &egui_winit::winit::event::WindowEvent,
    ) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Build one frame of UI from the built-in overlay.
    pub fn run(&mut self, window: &Window, state: &mut UiState) -> UiOutput {
        state.mode_toggled = false;
        self.run_with(window, |ui| overlay(ui.ctx(), state))
    }

    /// Build one frame of UI from a caller-supplied closure.
    ///
    /// The renderer only needs the tessellated output, so an application that
    /// has outgrown [`UiState`] (panels, inspectors, scrubbers) draws whatever
    /// it likes here instead. `build` is handed the root [`egui::Ui`] of the
    /// frame (panels need it; windows only need `ui.ctx()`). egui may call it
    /// more than once per frame, so it takes `FnMut`.
    pub fn run_with(&mut self, window: &Window, mut build: impl FnMut(&mut egui::Ui)) -> UiOutput {
        let input = self.state.take_egui_input(window);
        let full = self.ctx.run_ui(input, |ui| build(ui));
        self.state
            .handle_platform_output(window, full.platform_output);
        let primitives = self.ctx.tessellate(full.shapes, full.pixels_per_point);
        UiOutput {
            textures_delta: full.textures_delta,
            primitives,
            pixels_per_point: full.pixels_per_point,
            wants_pointer: self.ctx.egui_wants_pointer_input(),
            wants_keyboard: self.ctx.egui_wants_keyboard_input(),
        }
    }

    /// The egui context, for callers that need it outside a frame (fonts,
    /// style, pixels-per-point).
    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }
}

fn overlay(ctx: &egui::Context, state: &mut UiState) {
    egui::Window::new("Infinite Worlds")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!("{:.0} FPS  ({:.2} ms)", state.fps, state.frame_ms));
            ui.label(format!("Altitude: {}", format_altitude(state.altitude_km)));
            ui.separator();

            ui.add(
                egui::Slider::new(&mut state.exaggeration, 1.0..=200.0)
                    .logarithmic(true)
                    .text("Vertical exaggeration")
                    .suffix("x"),
            );
            if (state.effective_exaggeration - state.exaggeration).abs() > 0.05 {
                ui.label(format!(
                    "eased to {:.1}x near the surface",
                    state.effective_exaggeration
                ));
            }

            let label = match state.mode {
                ViewMode::Globe => "Switch to Mercator",
                ViewMode::Mercator => "Switch to Globe",
            };
            if ui.button(label).clicked() {
                state.mode_toggled = true;
            }

            egui::ComboBox::from_label("Layer")
                .selected_text(LAYERS[state.layer.min(LAYERS.len() - 1)])
                .show_ui(ui, |ui| {
                    for (i, name) in LAYERS.iter().enumerate() {
                        ui.selectable_value(&mut state.layer, i, *name);
                    }
                });

            ui.separator();
            ui.label(format!(
                "{} cells | chunks {}/{} | {} tris",
                state.n_cells, state.chunks_drawn, state.chunks_total, state.triangles_drawn
            ));
            ui.label(&state.device_name);
            ui.label("WASD move · right-drag look · R recenter · scroll zoom · M mode");
        });
}

/// Human-readable altitude string.
pub fn format_altitude(altitude_km: f32) -> String {
    if altitude_km < 1000.0 {
        format!("{altitude_km:.0} km")
    } else {
        format!("{:.0} km ({:.2} R)", altitude_km, altitude_km / 6371.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn altitude_formats_both_scales() {
        assert_eq!(format_altitude(50.0), "50 km");
        assert!(format_altitude(12742.0).contains("2.00 R"));
    }

    #[test]
    fn layer_index_is_clamped_for_display() {
        let idx = 99usize.min(LAYERS.len() - 1);
        assert_eq!(LAYERS[idx], "Precipitation");
    }
}
