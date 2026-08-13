//! Infinite Worlds — windowed viewer.
//!
//! WP2 scope: bring up the window, Vulkan renderer and camera against a
//! procedurally generated test planet. WP10 replaces the test data with live
//! `PlanetView` snapshots from the simulation.

mod terrain;
mod test_sphere;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use glam::{Vec2, Vec3};
use iw_mesh::{Mesh, EARTH_RADIUS_KM};
use iw_render_vulkan::camera::{effective_exaggeration, ViewMode};
use iw_render_vulkan::globe::GlobeParams;
use iw_render_vulkan::winit::application::ApplicationHandler;
use iw_render_vulkan::winit::dpi::{LogicalSize, PhysicalPosition};
use iw_render_vulkan::winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use iw_render_vulkan::winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use iw_render_vulkan::winit::keyboard::{KeyCode, PhysicalKey};
use iw_render_vulkan::winit::window::{Window, WindowId};
use iw_render_vulkan::{Camera, EguiFrame, MoveInput, Renderer, Ui, UiState};

/// Radians of free-look per pixel of right-button drag.
const LOOK_SENSITIVITY: f32 = 0.0045;

#[derive(Parser, Debug, Clone)]
#[command(name = "iw-app", about = "Infinite Worlds viewer")]
struct Args {
    /// Goldberg subdivision level (cells = 10*4^level + 2).
    #[arg(long, default_value_t = 6)]
    level: u8,
    /// Exit automatically after this many seconds (smoke testing).
    #[arg(long)]
    exit_after_secs: Option<f32>,
    /// Start in Mercator mode.
    #[arg(long)]
    mercator: bool,
    /// Use the built-in icosphere instead of iw-mesh (pipeline bring-up only).
    #[arg(long)]
    test_sphere: bool,
    /// Seed for the procedural test terrain.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Initial camera altitude in km (50..40000).
    #[arg(long, default_value_t = 16_000.0)]
    altitude_km: f32,
    /// Initial free-look pitch in degrees from straight down (0..88).
    #[arg(long, default_value_t = 0.0)]
    pitch_deg: f32,
    /// Disable chunk frustum/horizon culling (for comparison).
    #[arg(long)]
    no_cull: bool,
    /// Re-upload per-cell data at 10 Hz with a moving sea level. Exercises the
    /// streaming path WP10 uses for live simulation snapshots.
    #[arg(long)]
    animate_cells: bool,
}

#[derive(Default)]
struct Keys {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
}

impl Keys {
    fn move_input(&self) -> MoveInput {
        MoveInput {
            north: (self.w as i32 - self.s as i32) as f32,
            east: (self.d as i32 - self.a as i32) as f32,
        }
    }
}

struct App {
    args: Args,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    ui: Option<Ui>,
    ui_state: UiState,
    camera: Camera,
    keys: Keys,
    mesh: Option<Mesh>,
    elevation_m: Vec<f32>,
    colors: Vec<[u8; 4]>,
    start: Instant,
    last_frame: Instant,
    fps_window_start: Instant,
    fps_frames: u32,
    frames_total: u64,
    fps_min: f32,
    fps_max: f32,
    last_cell_update: Instant,
    cell_updates: u64,
    cursor: Option<PhysicalPosition<f64>>,
    right_down: bool,
    left_down: bool,
    error: Option<anyhow::Error>,
}

impl App {
    fn new(args: Args) -> App {
        let now = Instant::now();
        let mut camera = Camera::default();
        camera.altitude_km = args.altitude_km.clamp(
            iw_render_vulkan::camera::MIN_ALTITUDE_KM,
            iw_render_vulkan::camera::MAX_ALTITUDE_KM,
        );
        camera.free_look(0.0, args.pitch_deg.to_radians());
        if args.mercator {
            camera.mode = ViewMode::Mercator;
        }
        App {
            args,
            window: None,
            renderer: None,
            ui: None,
            ui_state: UiState::default(),
            camera,
            keys: Keys::default(),
            mesh: None,
            elevation_m: Vec::new(),
            colors: Vec::new(),
            start: now,
            last_frame: now,
            fps_window_start: now,
            fps_frames: 0,
            frames_total: 0,
            fps_min: f32::MAX,
            fps_max: 0.0,
            last_cell_update: now,
            cell_updates: 0,
            cursor: None,
            right_down: false,
            left_down: false,
            error: None,
        }
    }

    fn build_planet(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let mesh = if self.args.test_sphere {
            log::info!("building icosphere test mesh (level {})", self.args.level);
            test_sphere::build(self.args.level)
        } else {
            log::info!("building Goldberg mesh (level {})", self.args.level);
            Mesh::build(self.args.level)
        };
        log::info!(
            "mesh: {} cells, {} chunks, {:.2?}",
            mesh.n_cells(),
            mesh.chunks.len(),
            t0.elapsed()
        );
        self.elevation_m = terrain::generate_elevation(&mesh.centers, self.args.seed);
        self.colors = terrain::generate_colors(&self.elevation_m);
        let land = self.elevation_m.iter().filter(|e| **e > 0.0).count();
        log::info!(
            "test terrain: {:.0}% land, elevation {:.0}..{:.0} m",
            100.0 * land as f32 / self.elevation_m.len() as f32,
            self.elevation_m.iter().copied().fold(f32::MAX, f32::min),
            self.elevation_m.iter().copied().fold(f32::MIN, f32::max),
        );
        self.mesh = Some(mesh);
        Ok(())
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attrs = Window::default_attributes()
            .with_title("Infinite Worlds")
            .with_inner_size(LogicalSize::new(1600.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs)?);
        let size = window.inner_size();

        let mut renderer = Renderer::new(
            window.as_ref(),
            (size.width.max(1), size.height.max(1)),
            "Infinite Worlds",
        )?;
        self.build_planet()?;
        let mesh = self.mesh.as_ref().expect("mesh built");
        renderer.upload_mesh(mesh)?;
        renderer.update_cells(&self.elevation_m, &self.colors)?;

        self.ui_state.device_name = renderer.device_name().to_string();
        self.ui_state.n_cells = mesh.n_cells();
        self.ui_state.mode = self.camera.mode;
        self.ui = Some(Ui::new(window.as_ref()));
        self.renderer = Some(renderer);
        self.window = Some(window);
        Ok(())
    }

    fn toggle_mode(&mut self) {
        self.camera.mode = match self.camera.mode {
            ViewMode::Globe => {
                // Enter Mercator centred on the current focus point.
                let ll = iw_mesh::latlon_of(self.camera.focus);
                self.camera.mercator_center =
                    Vec2::new(ll[1], iw_render_vulkan::mercator::mercator_y(ll[0]));
                ViewMode::Mercator
            }
            ViewMode::Mercator => {
                // Return the globe focus to whatever the plane was centred on.
                let lat =
                    iw_render_vulkan::mercator::inverse_mercator_y(self.camera.mercator_center.y);
                self.camera.focus =
                    iw_render_vulkan::mercator::dir_from_latlon(lat, self.camera.mercator_center.x);
                ViewMode::Globe
            }
        };
        self.ui_state.mode = self.camera.mode;
    }

    fn on_cursor_move(&mut self, pos: PhysicalPosition<f64>) {
        let prev = self.cursor.replace(pos);
        let Some(prev) = prev else { return };
        let dx = (pos.x - prev.x) as f32;
        let dy = (pos.y - prev.y) as f32;
        if self.right_down {
            self.camera
                .free_look(-dx * LOOK_SENSITIVITY, -dy * LOOK_SENSITIVITY);
        } else if self.left_down {
            match self.camera.mode {
                ViewMode::Globe => {
                    // Arcball drag: sensitivity falls with altitude so the
                    // surface tracks the pointer at any zoom.
                    let scale =
                        (self.camera.altitude_km / EARTH_RADIUS_KM).clamp(0.01, 2.0) * 0.004;
                    let (east, north) = iw_render_vulkan::camera::east_north(self.camera.focus);
                    let tangent = east * (-dx * scale) + north * (dy * scale);
                    let angle = tangent.length();
                    if angle > 0.0 {
                        self.camera.focus = iw_render_vulkan::camera::great_circle_step(
                            self.camera.focus,
                            tangent,
                            angle,
                        );
                    }
                }
                ViewMode::Mercator => {
                    let size = self
                        .window
                        .as_ref()
                        .map(|w| w.inner_size())
                        .unwrap_or_default();
                    let per_px = self.camera.mercator_half_width * 2.0 / size.width.max(1) as f32;
                    self.camera
                        .mercator_pan(Vec2::new(-dx * per_px, dy * per_px));
                }
            }
        }
    }

    fn draw(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        let Some(window) = self.window.clone() else {
            return Ok(());
        };
        if self.renderer.is_none() || self.ui.is_none() {
            return Ok(());
        }

        self.fps_frames += 1;
        let elapsed = (now - self.fps_window_start).as_secs_f32();
        if elapsed >= 0.25 {
            let fps = self.fps_frames as f32 / elapsed;
            self.ui_state.fps = fps;
            self.ui_state.frame_ms = 1000.0 * elapsed / self.fps_frames as f32;
            // Ignore the first window, which includes mesh upload.
            if self.frames_total > 30 {
                self.fps_min = self.fps_min.min(fps);
                self.fps_max = self.fps_max.max(fps);
            }
            self.fps_frames = 0;
            self.fps_window_start = now;
        }
        self.frames_total += 1;

        if self.args.animate_cells && (now - self.last_cell_update).as_secs_f32() >= 0.1 {
            self.last_cell_update = now;
            // Move sea level a few hundred metres, recolour, re-upload. Same
            // shape of work a live simulation snapshot does.
            let offset = 400.0 * (self.start.elapsed().as_secs_f32() * 0.7).sin();
            let colors: Vec<[u8; 4]> = self
                .elevation_m
                .iter()
                .map(|e| terrain::hypsometric(e - offset))
                .collect();
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.update_cells(&self.elevation_m, &colors)?;
            }
            self.cell_updates += 1;
        }

        let ui_out = self
            .ui
            .as_mut()
            .expect("ui")
            .run(window.as_ref(), &mut self.ui_state);
        if self.ui_state.mode_toggled {
            self.toggle_mode();
        }
        let input = if ui_out.wants_keyboard {
            MoveInput::default()
        } else {
            self.keys.move_input()
        };
        self.camera.update(input, dt);

        let exaggeration =
            effective_exaggeration(self.ui_state.exaggeration, self.camera.altitude_km);
        self.ui_state.altitude_km = self.camera.altitude_km;
        self.ui_state.effective_exaggeration = exaggeration;

        let aspect = self.renderer.as_ref().expect("renderer").aspect();
        let params = GlobeParams {
            view_proj: self.camera.view_proj(aspect),
            camera_pos_km: match self.camera.mode {
                ViewMode::Globe => self.camera.eye(),
                // Mercator is orthographic; the camera position is only used
                // for culling, which is disabled in that mode.
                ViewMode::Mercator => Vec3::ZERO,
            },
            exaggeration,
            base_offset_m: 0.0,
            radius_km: EARTH_RADIUS_KM,
            mode: self.camera.mode,
            center_lon_rad: self.camera.mercator_center.x,
            cull: !self.args.no_cull,
            star_seed: 1.0,
            star_brightness: 1.0,
        };

        let mut ui_out = ui_out;
        let renderer = self.renderer.as_mut().expect("renderer");
        renderer.render(
            &params,
            EguiFrame {
                textures_delta: &mut ui_out.textures_delta,
                primitives: &ui_out.primitives,
                pixels_per_point: ui_out.pixels_per_point,
            },
        )?;

        let stats = renderer.stats();
        self.ui_state.chunks_drawn = stats.chunks_drawn;
        self.ui_state.chunks_total = stats.chunks_total;
        self.ui_state.triangles_drawn = stats.triangles_drawn;

        if let Some(limit) = self.args.exit_after_secs {
            if (now - self.start).as_secs_f32() >= limit {
                log::info!(
                    "exit-after-secs reached: {} frames, {:.1} avg FPS (min {:.1}, max {:.1})",
                    self.frames_total,
                    self.frames_total as f32 / (now - self.start).as_secs_f32(),
                    if self.fps_min == f32::MAX {
                        0.0
                    } else {
                        self.fps_min
                    },
                    self.fps_max
                );
                if self.args.animate_cells {
                    log::info!("{} cell-data uploads", self.cell_updates);
                }
                event_loop.exit();
            }
        }
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        if let Err(e) = self.init(event_loop) {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let consumed = match (self.ui.as_mut(), self.window.clone()) {
            (Some(ui), Some(window)) => ui.on_window_event(window.as_ref(), &event),
            _ => false,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.request_resize((size.width.max(1), size.height.max(1)));
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let (Some(r), Some(w)) = (self.renderer.as_mut(), self.window.as_ref()) {
                    let s = w.inner_size();
                    r.request_resize((s.width.max(1), s.height.max(1)));
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if consumed {
                    return;
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys.w = pressed,
                        KeyCode::KeyA => self.keys.a = pressed,
                        KeyCode::KeyS => self.keys.s = pressed,
                        KeyCode::KeyD => self.keys.d = pressed,
                        KeyCode::KeyR if pressed => self.camera.begin_recenter(),
                        KeyCode::KeyM if pressed => self.toggle_mode(),
                        KeyCode::Escape if pressed => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if pressed && consumed {
                    return;
                }
                match button {
                    MouseButton::Right => self.right_down = pressed,
                    MouseButton::Left => self.left_down = pressed,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => self.on_cursor_move(position),
            WindowEvent::CursorLeft { .. } => {
                self.cursor = None;
                self.right_down = false;
                self.left_down = false;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if consumed {
                    return;
                }
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                self.camera.zoom(steps);
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw(event_loop) {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Tear down here, not after `run_app` returns: the Vulkan surface must
        // die before the window it was made from, and egui's clipboard worker
        // must die before winit drops the Wayland connection out from under it.
        if let Some(r) = self.renderer.as_ref() {
            r.wait_idle();
        }
        self.renderer = None;
        self.ui = None;
        self.window = None;
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let event_loop = EventLoop::new().context("creating the winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(args);
    event_loop.run_app(&mut app).context("event loop")?;
    // Drop GPU resources before returning so teardown errors surface here.
    app.renderer = None;
    match app.error.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
