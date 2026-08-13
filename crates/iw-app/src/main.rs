//! Infinite Worlds — the interactive viewer.
//!
//! The simulation runs on its own worker thread ([`iw_sim::SimHandle`]) from
//! the moment the app starts; the window renders whatever the latest published
//! [`iw_core::PlanetView`] holds, so generation is a live spectacle rather than
//! a progress bar. Everything the UI does is non-blocking: snapshots arrive by
//! `arc-swap`, progress events by a drained ring, stratigraphic columns by a
//! request/response channel, and history snapshots are written on a third
//! thread.

mod history_log;
mod inspector;
mod layers;
mod panel;
mod pick;
mod scrubber;
mod store;
mod terrain;
mod test_sphere;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::Receiver;
use glam::{Vec2, Vec3};
use iw_core::{CheckpointStore, NullProgress, Phase, PlanetConfig, PlanetView, Process, Stratum};
use iw_mesh::{Mesh, EARTH_RADIUS_KM};
use iw_render_vulkan::camera::{effective_exaggeration, ViewMode};
use iw_render_vulkan::egui;
use iw_render_vulkan::globe::GlobeParams;
use iw_render_vulkan::winit::application::ApplicationHandler;
use iw_render_vulkan::winit::dpi::{LogicalSize, PhysicalPosition};
use iw_render_vulkan::winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use iw_render_vulkan::winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use iw_render_vulkan::winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use iw_render_vulkan::winit::window::{Window, WindowId};
use iw_render_vulkan::{Camera, EguiFrame, MoveInput, Renderer, Ui};
use iw_sim::{SimHandle, SimState};
use iw_store_postcard::{HistorySnapshot, HistoryStore};

use history_log::EventLog;
use layers::{Layer, LayerScales};
use panel::{phase_name, PanelState};
use scrubber::{ScrubAction, Scrubber};
use store::{planet_dir, HistoryWriter, SwitchableStore};

/// Radians of free-look per pixel of right-button drag.
const LOG_SENSITIVITY: f32 = 0.0045;
/// Pointer travel (pixels) above which a left-click counts as a drag, not a pick.
const CLICK_SLOP_PX: f32 = 5.0;
/// How long the inspector waits for a column before giving up on the request.
const COLUMN_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// Render a procedural icosphere instead of simulating a planet
    /// (renderer bring-up only; no simulation runs in this mode).
    #[arg(long)]
    test_sphere: bool,
    /// Planet seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Planet directory. Defaults to `./planets/<seed>-L<level>`.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Initial camera altitude in km (50..40000).
    #[arg(long, default_value_t = 16_000.0)]
    altitude_km: f32,
    /// Initial free-look pitch in degrees from straight down (0..88).
    #[arg(long, default_value_t = 0.0)]
    pitch_deg: f32,
    /// Disable chunk frustum/horizon culling (for comparison).
    #[arg(long)]
    no_cull: bool,
    /// Start paused instead of generating immediately.
    #[arg(long)]
    paused: bool,
}

/// Build the five real processes in the order the simulation requires. Mirrors
/// `iw-headless`'s `build_processes` (tectonics, geology, climate, surface,
/// biomes — see `Planet`'s field-ownership table).
fn build_processes() -> Vec<Box<dyn Process>> {
    vec![
        Box::new(iw_tectonics::TectonicsProcess::default()),
        Box::new(iw_geology::GeologyProcess::default()),
        Box::new(iw_climate::ClimateProcess::default()),
        Box::new(iw_surface::SurfaceProcess::default()),
        Box::new(iw_biomes::BiomeProcess),
    ]
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

/// What one frame of UI asked for, applied after the egui closure returns (it
/// borrows the app, so it cannot touch the renderer).
#[derive(Default)]
struct FrameActions {
    regenerate: Option<PlanetConfig>,
    rerun: Option<(Phase, PlanetConfig)>,
    scrub: Option<ScrubAction>,
    recolor: bool,
    toggle_mode: bool,
}

/// The live simulation and everything hanging off it.
struct Sim {
    handle: SimHandle,
    store: Arc<SwitchableStore>,
    history: HistoryWriter,
    reader: Option<HistoryStore>,
    /// Filled by the injected mesh builder when the worker rebuilds the mesh.
    rebuilt_mesh: Arc<Mutex<Option<Arc<Mesh>>>>,
    /// Checkpoint tags on disk, refreshed periodically.
    completed: Vec<Phase>,
    last_checkpoint_scan: Instant,
    /// Worker state at the last disk scan, so a run that has gone quiet stops
    /// re-listing history every frame-and-a-bit.
    last_state: Option<SimState>,
}

struct App {
    args: Args,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    ui: Option<Ui>,
    camera: Camera,
    keys: Keys,

    mesh: Option<Arc<Mesh>>,
    sim: Option<Sim>,
    config: PlanetConfig,
    /// Latest snapshot applied to the globe.
    view: Option<PlanetView>,
    applied_version: u64,
    /// History snapshot currently displayed, if scrubbing.
    historic: Option<HistorySnapshot>,

    layer: Layer,
    scales: LayerScales,
    exaggeration: f32,
    panel: PanelState,
    scrubber: Scrubber,
    inspector: inspector::Inspector,
    pending_column: Option<(Receiver<Vec<Stratum>>, Instant)>,
    /// Scrub requested by a key press, applied after the next UI frame.
    pending_scrub: Option<ScrubAction>,
    /// Regenerate requested by a key press (F5), applied after the UI frame.
    pending_regenerate: Option<PlanetConfig>,
    log: EventLog,
    show_help: bool,

    /// Static-terrain fallback for `--test-sphere`.
    static_elevation: Vec<f32>,

    start: Instant,
    last_frame: Instant,
    fps_window_start: Instant,
    fps_frames: u32,
    frames_total: u64,
    fps: f32,
    frame_ms: f32,
    fps_min: f32,
    fps_max: f32,
    fps_sum_during_gen: f64,
    fps_samples_during_gen: u32,
    cell_updates: u64,

    cursor: Option<PhysicalPosition<f64>>,
    left_press: Option<Vec2>,
    left_moved: f32,
    right_down: bool,
    left_down: bool,
    surface_size: (u32, u32),
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
        let mut config = PlanetConfig {
            seed: args.seed,
            subdivision_level: args.level,
            ..PlanetConfig::default()
        };
        config.sanitize();
        App {
            window: None,
            renderer: None,
            ui: None,
            camera,
            keys: Keys::default(),
            mesh: None,
            sim: None,
            panel: PanelState::from_config(&config),
            config,
            view: None,
            applied_version: 0,
            historic: None,
            layer: Layer::Beauty,
            scales: LayerScales::default(),
            exaggeration: 30.0,
            scrubber: Scrubber::default(),
            inspector: inspector::Inspector::default(),
            pending_column: None,
            pending_scrub: None,
            pending_regenerate: None,
            log: EventLog::default(),
            show_help: false,
            static_elevation: Vec::new(),
            start: now,
            last_frame: now,
            fps_window_start: now,
            fps_frames: 0,
            frames_total: 0,
            fps: 0.0,
            frame_ms: 0.0,
            fps_min: f32::MAX,
            fps_max: 0.0,
            fps_sum_during_gen: 0.0,
            fps_samples_during_gen: 0,
            cell_updates: 0,
            cursor: None,
            left_press: None,
            left_moved: 0.0,
            right_down: false,
            left_down: false,
            surface_size: (1, 1),
            args,
            error: None,
        }
    }

    /// Directory this planet's checkpoints and history live in.
    fn planet_dir(&self, config: &PlanetConfig) -> PathBuf {
        match &self.args.out {
            Some(dir) => dir.clone(),
            None => planet_dir(
                &PathBuf::from("planets"),
                config.seed,
                config.subdivision_level,
            ),
        }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attrs = Window::default_attributes()
            .with_title("Infinite Worlds")
            .with_inner_size(LogicalSize::new(1600.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs)?);
        let size = window.inner_size();
        self.surface_size = (size.width.max(1), size.height.max(1));

        let mut renderer = Renderer::new(window.as_ref(), self.surface_size, "Infinite Worlds")?;

        let t0 = Instant::now();
        let mesh = if self.args.test_sphere {
            log::info!("building icosphere test mesh (level {})", self.args.level);
            Arc::new(test_sphere::build(self.args.level))
        } else {
            log::info!(
                "building Goldberg mesh (level {})",
                self.config.subdivision_level
            );
            Arc::new(Mesh::build(self.config.subdivision_level))
        };
        log::info!(
            "mesh: {} cells, {} chunks, {:.2?}",
            mesh.n_cells(),
            mesh.chunks.len(),
            t0.elapsed()
        );
        renderer.upload_mesh(&mesh)?;

        if self.args.test_sphere {
            // No simulation: procedural terrain, as in WP2.
            self.static_elevation = terrain::generate_elevation(&mesh.centers, self.args.seed);
            let colors = terrain::generate_colors(&self.static_elevation);
            renderer.update_cells(&self.static_elevation, &colors)?;
            self.log.push("test sphere: no simulation running".into());
        } else {
            renderer.update_cells(
                &vec![0.0; mesh.n_cells()],
                &vec![[10, 20, 40, 255]; mesh.n_cells()],
            )?;
        }

        self.ui = Some(Ui::new(window.as_ref()));
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.mesh = Some(mesh);

        if !self.args.test_sphere {
            self.spawn_sim()?;
        }
        Ok(())
    }

    /// Start the simulation worker for the current config and point the
    /// checkpoint/history stores at its planet directory.
    fn spawn_sim(&mut self) -> Result<()> {
        let config = self.config.clone();
        let dir = self.planet_dir(&config);
        let store = Arc::new(
            SwitchableStore::new(dir.clone())
                .with_context(|| format!("opening planet directory {}", dir.display()))?,
        );
        let history = HistoryWriter::new(dir.clone(), config.history_cap_bytes);
        let reader = HistoryStore::new(dir.clone(), config.history_cap_bytes).ok();
        let mesh = self.mesh.clone().expect("mesh built before the sim starts");

        let rebuilt_mesh: Arc<Mutex<Option<Arc<Mesh>>>> = Arc::new(Mutex::new(None));
        let slot = Arc::clone(&rebuilt_mesh);
        let mesh_builder: iw_sim::MeshBuilder = Arc::new(move |level: u8| {
            let t0 = Instant::now();
            let m = Arc::new(Mesh::build(level));
            log::info!(
                "rebuilt mesh at level {level}: {} cells in {:.2?}",
                m.n_cells(),
                t0.elapsed()
            );
            *slot.lock().unwrap() = Some(Arc::clone(&m));
            m
        });

        let handle = SimHandle::spawn(
            config.clone(),
            mesh,
            build_processes(),
            store.clone(),
            Arc::new(NullProgress),
            mesh_builder,
        );
        if !self.args.paused {
            handle.start();
        }
        log::info!(
            "simulation: seed {} level {} -> {}",
            config.seed,
            config.subdivision_level,
            dir.display()
        );
        self.log.push(format!(
            "seed {} level {} ({} cells)",
            config.seed,
            config.subdivision_level,
            config.n_cells()
        ));
        self.sim = Some(Sim {
            handle,
            store,
            history,
            reader,
            rebuilt_mesh,
            completed: Vec::new(),
            last_checkpoint_scan: Instant::now() - Duration::from_secs(10),
            last_state: None,
        });
        Ok(())
    }

    /// Apply a new config: either a full regenerate (new planet directory when
    /// the seed or level changed) or a re-run from a phase checkpoint.
    fn apply_config(&mut self, mut config: PlanetConfig, rerun: Option<Phase>) {
        config.sanitize();
        let Some(sim) = self.sim.as_mut() else {
            return;
        };
        match rerun {
            Some(phase) => {
                if config.subdivision_level != self.config.subdivision_level {
                    self.log
                        .push("re-run needs the same subdivision level; regenerate instead".into());
                    return;
                }
                sim.handle.rerun_from_phase(phase, config.clone());
                self.log
                    .push(format!("re-running from {}", phase_name(phase)));
            }
            None => {
                let dir = match &self.args.out {
                    Some(dir) => dir.clone(),
                    None => planet_dir(
                        &PathBuf::from("planets"),
                        config.seed,
                        config.subdivision_level,
                    ),
                };
                if let Err(e) = sim.store.retarget(dir.clone()) {
                    log::error!("planet directory {}: {e:#}", dir.display());
                }
                sim.history.retarget(dir.clone(), config.history_cap_bytes);
                sim.reader = HistoryStore::new(dir.clone(), config.history_cap_bytes).ok();
                sim.handle.regenerate(config.clone());
                log::info!("planet directory: {}", sim.store.dir().display());
                self.log.push(format!(
                    "regenerating: seed {} level {}",
                    config.seed, config.subdivision_level
                ));
            }
        }
        sim.handle.start();
        sim.completed.clear();
        sim.last_checkpoint_scan = Instant::now() - Duration::from_secs(10);
        self.config = config.clone();
        self.panel.set_config(&config);
        self.scrubber = Scrubber::default();
        self.historic = None;
        self.inspector.clear();
        self.applied_version = 0;
    }

    /// Drain worker events into the log and the progress readout.
    fn poll_events(&mut self) {
        let Some(sim) = self.sim.as_ref() else { return };
        for ev in sim.handle.drain_events() {
            self.log.event(ev);
        }
        if let Some(phase) = self.log.last_completed_phase.take() {
            if let Some(sim) = self.sim.as_mut() {
                if !sim.completed.contains(&phase) {
                    sim.completed.push(phase);
                }
            }
        }
    }

    /// Adopt a new snapshot: recolour the globe, archive it, refresh history.
    fn poll_view(&mut self) -> Result<()> {
        let Some(sim) = self.sim.as_ref() else {
            return Ok(());
        };
        // A regenerate at a new level rebuilds the mesh on the worker thread;
        // pick it up before touching any snapshot sized for it.
        let rebuilt = sim.rebuilt_mesh.lock().unwrap().take();
        if let Some(mesh) = rebuilt {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.upload_mesh(&mesh)?;
            }
            self.mesh = Some(mesh);
            self.applied_version = 0;
        }

        let Some(view) = sim.handle.latest_view() else {
            return Ok(());
        };
        if view.version == self.applied_version {
            return Ok(());
        }
        let n = self.mesh.as_ref().map(|m| m.n_cells()).unwrap_or(0);
        if view.cells.elevation_m.len() != n {
            // Snapshot from before a mesh rebuild; skip it.
            return Ok(());
        }
        sim.history.push(&view);
        self.applied_version = view.version;
        self.scales = LayerScales::from_cells(&view.cells);
        self.view = Some(view);
        if self.scrubber.live {
            self.refresh_cells()?;
        }
        Ok(())
    }

    /// Push per-cell elevation and colour for the current layer and source
    /// (live snapshot, or the history snapshot being scrubbed).
    fn refresh_cells(&mut self) -> Result<()> {
        let Some(renderer) = self.renderer.as_mut() else {
            return Ok(());
        };
        if let Some(snap) = self.historic.as_ref() {
            let (colors, substituted) = layers::snapshot_colors(self.layer, snap);
            renderer.update_cells(&snap.elevation_m, &colors)?;
            self.scrubber.layer_substituted = substituted;
            self.cell_updates += 1;
            return Ok(());
        }
        if let Some(view) = self.view.as_ref() {
            let colors =
                layers::cell_colors(self.layer, &view.cells, view.sea_level_m, self.scales);
            renderer.update_cells(&view.cells.elevation_m, &colors)?;
            self.cell_updates += 1;
        } else if self.args.test_sphere && !self.static_elevation.is_empty() {
            let colors = terrain::generate_colors(&self.static_elevation);
            renderer.update_cells(&self.static_elevation, &colors)?;
        }
        Ok(())
    }

    /// Load and display a history snapshot.
    fn show_history(&mut self, version: u64) -> Result<()> {
        let Some(sim) = self.sim.as_ref() else {
            return Ok(());
        };
        let Some(reader) = sim.reader.as_ref() else {
            return Ok(());
        };
        match reader.load(version) {
            Ok(snap) => {
                let n = self.mesh.as_ref().map(|m| m.n_cells()).unwrap_or(0);
                if snap.elevation_m.len() != n {
                    self.log
                        .push(format!("history {version} is from another mesh; ignored"));
                    return Ok(());
                }
                self.scrubber.showing = Some(version);
                self.historic = Some(snap);
                self.refresh_cells()?;
            }
            Err(e) => {
                log::warn!("history {version}: {e:#}");
                self.log.push(format!("history {version} unreadable"));
            }
        }
        Ok(())
    }

    /// Stop scrubbing and go back to the live snapshot.
    fn go_live(&mut self) -> Result<()> {
        self.historic = None;
        self.scrubber.go_live();
        self.refresh_cells()
    }

    /// Periodically re-read the history list and the checkpoint tags.
    fn refresh_disk_state(&mut self) {
        let Some(sim) = self.sim.as_mut() else { return };
        // `HistoryStore::list` opens every snapshot header, so only re-read it
        // when new snapshots can actually have appeared.
        let state = sim.handle.state();
        let changed = sim.last_state != Some(state);
        sim.last_state = Some(state);
        let growing = matches!(state, SimState::Running(_));
        if self.scrubber.due_refresh() && (growing || changed || self.scrubber.entries.is_empty()) {
            let entries = sim
                .reader
                .as_ref()
                .and_then(|r| r.list().ok())
                .unwrap_or_default();
            self.scrubber.set_entries(entries);
        }
        if sim.last_checkpoint_scan.elapsed() >= Duration::from_secs(1) {
            sim.last_checkpoint_scan = Instant::now();
            if let Ok(tags) = sim.store.list() {
                sim.completed = Phase::ALL
                    .iter()
                    .copied()
                    .filter(|p| tags.contains(&iw_sim::phase_tag(*p)))
                    .collect();
            }
        }
    }

    /// Poll an outstanding column request from the inspector.
    fn poll_column(&mut self) {
        if let Some((rx, sent)) = self.pending_column.as_ref() {
            match rx.try_recv() {
                Ok(column) => {
                    self.inspector.set_column(column, self.applied_version);
                    self.pending_column = None;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if sent.elapsed() > COLUMN_TIMEOUT {
                        self.inspector.awaiting_column = false;
                        self.pending_column = None;
                    }
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    self.inspector.awaiting_column = false;
                    self.pending_column = None;
                }
            }
        }
        if self.pending_column.is_none() && self.inspector.needs_column(self.applied_version) {
            if let (Some(sim), Some(cell)) = (self.sim.as_ref(), self.inspector.cell) {
                self.inspector.awaiting_column = true;
                self.pending_column = Some((sim.handle.request_column(cell), Instant::now()));
            }
        }
    }

    /// Left-click: ray through the cursor, sphere intersection, cell lookup.
    fn pick_at(&mut self, pos: Vec2) {
        let (Some(mesh), Some(renderer)) = (self.mesh.as_ref(), self.renderer.as_ref()) else {
            return;
        };
        let ndc = pick::ndc_from_pixel(pos, self.surface_size);
        let Some(dir) =
            pick::surface_direction(&self.camera, renderer.aspect(), ndc, EARTH_RADIUS_KM)
        else {
            self.inspector.clear();
            return;
        };
        let cell = mesh.cell_at(dir);
        self.inspector.select(cell);
    }

    /// Inspect whatever is under the middle of the window (the keyboard route
    /// to the same ray -> cell lookup a left-click uses).
    fn pick_centre(&mut self) {
        let centre = Vec2::new(
            self.surface_size.0 as f32 * 0.5,
            self.surface_size.1 as f32 * 0.5,
        );
        self.pick_at(centre);
    }

    /// Move the scrubber `delta` snapshots and queue the load.
    fn scrub_by(&mut self, delta: i64) {
        if self.scrubber.entries.is_empty() {
            return;
        }
        let last = self.scrubber.entries.len() as i64 - 1;
        let index = (self.scrubber.index as i64 + delta).clamp(0, last) as usize;
        self.scrubber.index = index;
        self.scrubber.live = false;
        if let Some(v) = self.scrubber.selected_version() {
            self.pending_scrub = Some(ScrubAction::Load(v));
        }
    }

    fn toggle_mode(&mut self) {
        self.camera.mode = match self.camera.mode {
            ViewMode::Globe => {
                let ll = iw_mesh::latlon_of(self.camera.focus);
                self.camera.mercator_center =
                    Vec2::new(ll[1], iw_render_vulkan::mercator::mercator_y(ll[0]));
                ViewMode::Mercator
            }
            ViewMode::Mercator => {
                let lat =
                    iw_render_vulkan::mercator::inverse_mercator_y(self.camera.mercator_center.y);
                self.camera.focus =
                    iw_render_vulkan::mercator::dir_from_latlon(lat, self.camera.mercator_center.x);
                ViewMode::Globe
            }
        };
    }

    fn set_layer(&mut self, layer: Layer) {
        if self.layer != layer {
            self.layer = layer;
            if let Err(e) = self.refresh_cells() {
                self.error = Some(e);
            }
        }
    }

    fn toggle_pause(&mut self) {
        let Some(sim) = self.sim.as_ref() else { return };
        match sim.handle.state() {
            SimState::Running(_) => sim.handle.pause(),
            SimState::Idle | SimState::Paused(_) => sim.handle.start(),
            SimState::Done => {}
        }
    }

    fn on_cursor_move(&mut self, pos: PhysicalPosition<f64>) {
        let prev = self.cursor.replace(pos);
        let Some(prev) = prev else { return };
        let dx = (pos.x - prev.x) as f32;
        let dy = (pos.y - prev.y) as f32;
        if self.left_down {
            self.left_moved += dx.abs() + dy.abs();
        }
        if self.right_down {
            self.camera
                .free_look(-dx * LOG_SENSITIVITY, -dy * LOG_SENSITIVITY);
        } else if self.left_down {
            match self.camera.mode {
                ViewMode::Globe => {
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
                    let per_px =
                        self.camera.mercator_half_width * 2.0 / self.surface_size.0.max(1) as f32;
                    self.camera
                        .mercator_pan(Vec2::new(-dx * per_px, dy * per_px));
                }
            }
        }
    }

    // --- UI ----------------------------------------------------------------

    fn build_ui(&mut self, root: &mut egui::Ui, actions: &mut FrameActions) {
        let ctx = root.ctx().clone();
        let ctx = &ctx;
        let scrub = scrubber::show(
            root,
            &mut self.scrubber,
            self.config.total_duration_myr(),
            self.layer.name(),
        );
        if scrub != ScrubAction::None {
            actions.scrub = Some(scrub);
        }
        self.hud(ctx, actions);
        self.progress_window(ctx);
        let running = self.config.clone();
        let completed = self
            .sim
            .as_ref()
            .map(|s| s.completed.clone())
            .unwrap_or_default();
        let action = panel::show(ctx, &mut self.panel, &running, &completed);
        if action.regenerate {
            actions.regenerate = Some(self.panel.to_config());
        }
        if let Some(phase) = action.rerun_from {
            actions.rerun = Some((phase, self.panel.to_config()));
        }
        let mut close_inspector = false;
        if let (Some(mesh), Some(view)) = (self.mesh.as_ref(), self.view.as_ref()) {
            close_inspector =
                !inspector::show(ctx, &self.inspector, mesh, view, self.historic.is_some());
        }
        if close_inspector {
            self.inspector.clear();
        }
        if self.layer == Layer::Plates {
            self.draw_plate_arrows(ctx);
        }
        if self.show_help {
            self.help_window(ctx);
        }
    }

    fn hud(&mut self, ctx: &egui::Context, actions: &mut FrameActions) {
        egui::Window::new("Infinite Worlds")
            .default_pos([12.0, 12.0])
            .default_width(300.0)
            .show(ctx, |ui| {
                ui.label(format!("{:.0} FPS  ({:.2} ms)", self.fps, self.frame_ms));
                ui.label(format!(
                    "Altitude: {}",
                    iw_render_vulkan::ui::format_altitude(self.camera.altitude_km)
                ));
                ui.label(self.status_line());
                ui.separator();

                ui.add(
                    egui::Slider::new(&mut self.exaggeration, 1.0..=200.0)
                        .logarithmic(true)
                        .text("Vertical exaggeration")
                        .suffix("x"),
                );
                let eff = effective_exaggeration(self.exaggeration, self.camera.altitude_km);
                if (eff - self.exaggeration).abs() > 0.05 {
                    ui.label(format!("eased to {eff:.1}x near the surface"));
                }
                let label = match self.camera.mode {
                    ViewMode::Globe => "Switch to Mercator",
                    ViewMode::Mercator => "Switch to Globe",
                };
                if ui.button(label).clicked() {
                    actions.toggle_mode = true;
                }

                let mut layer_index = self.layer.index();
                egui::ComboBox::from_label("Layer")
                    .selected_text(self.layer.name())
                    .show_ui(ui, |ui| {
                        for (i, layer) in Layer::ALL.iter().enumerate() {
                            ui.selectable_value(&mut layer_index, i, layer.name());
                        }
                    });
                if layer_index != self.layer.index() {
                    self.layer = Layer::from_index(layer_index);
                    actions.recolor = true;
                }
                ui.weak(self.layer.legend());

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("New planet (N)").clicked() {
                        self.panel.open = !self.panel.open;
                    }
                    if ui.button("Help (F1)").clicked() {
                        self.show_help = !self.show_help;
                    }
                });
                if self.sim.is_some() {
                    ui.horizontal(|ui| {
                        let running = matches!(
                            self.sim.as_ref().map(|s| s.handle.state()),
                            Some(SimState::Running(_))
                        );
                        if ui
                            .button(if running {
                                "Pause (Space)"
                            } else {
                                "Run (Space)"
                            })
                            .clicked()
                        {
                            self.toggle_pause();
                        }
                        if ui.button("Step (Shift+S)").clicked() {
                            if let Some(sim) = self.sim.as_ref() {
                                sim.handle.step_once();
                            }
                        }
                    });
                }

                ui.separator();
                if let (Some(mesh), Some(renderer)) = (self.mesh.as_ref(), self.renderer.as_ref()) {
                    let stats = renderer.stats();
                    ui.label(format!(
                        "{} cells | chunks {}/{} | {} tris",
                        mesh.n_cells(),
                        stats.chunks_drawn,
                        stats.chunks_total,
                        stats.triangles_drawn
                    ));
                    ui.weak(renderer.device_name());
                }
            });
    }

    /// Phase, progress bar, clocks and the narration log.
    fn progress_window(&mut self, ctx: &egui::Context) {
        let Some(sim) = self.sim.as_ref() else { return };
        let status = sim.handle.status();
        let total_myr = self.config.total_duration_myr();
        egui::Window::new("Generation")
            // Below the HUD in the 1600x900 default window; egui clamps this
            // upward (overlapping the HUD) if the window is much shorter.
            .default_pos([12.0, 430.0])
            .default_width(340.0)
            .show(ctx, |ui| {
                let phase = match status.state {
                    SimState::Running(p) | SimState::Paused(p) => Some(p),
                    _ => None,
                };
                ui.label(match status.state {
                    SimState::Idle => "idle".to_string(),
                    SimState::Running(p) => format!("running: {}", phase_name(p)),
                    SimState::Paused(p) => format!("paused: {}", phase_name(p)),
                    SimState::Done => "complete".to_string(),
                });
                let (step, of) = match (phase, self.log.phase_progress) {
                    (Some(p), Some((lp, step, of))) if lp == p => (step, of),
                    _ => (
                        status.step_index.min(status.total_steps),
                        status.total_steps.max(1),
                    ),
                };
                let frac = step as f32 / of.max(1) as f32;
                ui.add(
                    egui::ProgressBar::new(frac.clamp(0.0, 1.0)).text(format!("step {step}/{of}")),
                );
                ui.label(format!(
                    "{:.2} Myr elapsed  ({:.2} Myr ago)",
                    status.time_myr,
                    (total_myr - status.time_myr).max(0.0)
                ));
                ui.weak(format!(
                    "{} of {} steps overall",
                    status.step_index, status.total_steps
                ));
                ui.separator();
                ui.label(self.log.latest_narration());
                ui.collapsing("Log", |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(160.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for line in self.log.recent(24) {
                                ui.weak(line);
                            }
                        });
                });
            });
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_help;
        egui::Window::new("Keys")
            .open(&mut open)
            .default_pos([700.0, 120.0])
            .show(ctx, |ui| {
                for (key, what) in [
                    ("W A S D", "move north / west / south / east"),
                    ("left drag", "spin the globe (Mercator: pan)"),
                    ("left click", "inspect the cell under the cursor"),
                    ("I", "inspect the cell at the centre of the view"),
                    ("right drag", "free look"),
                    ("scroll", "zoom"),
                    ("R", "recenter the view"),
                    ("G / M", "globe / Mercator toggle"),
                    ("Space", "pause / resume the simulation"),
                    ("Shift+S", "single step (plain S is 'move south')"),
                    ("N", "New planet panel"),
                    ("F5", "roll a new seed and regenerate"),
                    ("1..9, 0", "select data layer"),
                    ("-", "water flux layer"),
                    ("Tab", "next data layer"),
                    (", / .", "scrub one history snapshot back / forward"),
                    ("L", "back to the live simulation"),
                    ("F1", "this help"),
                    ("Esc", "quit"),
                ] {
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{key:<11}"));
                        ui.label(what);
                    });
                }
            });
        self.show_help = open;
    }

    /// Plate velocity arrows.
    ///
    /// Drawn with the egui painter in screen space rather than as a Vulkan line
    /// pipeline: the arrows are a sparse overlay (a few hundred segments), they
    /// need no depth interaction beyond the horizon test done here, and this
    /// keeps the renderer's pipeline set (globe + starfield + egui) unchanged.
    fn draw_plate_arrows(&self, ctx: &egui::Context) {
        let (Some(mesh), Some(view), Some(renderer)) = (
            self.mesh.as_ref(),
            self.view.as_ref(),
            self.renderer.as_ref(),
        ) else {
            return;
        };
        if self.camera.mode != ViewMode::Globe || self.historic.is_some() {
            return;
        }
        let cells = &view.cells;
        if cells.plate_velocity_m_yr.len() != mesh.n_cells() {
            return;
        }
        let view_proj = self.camera.view_proj(renderer.aspect());
        let ppp = ctx.pixels_per_point().max(0.1);
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("plate-arrows"),
        ));
        let eye = self.camera.eye();
        let cos_horizon = EARTH_RADIUS_KM / self.camera.radius_km();
        let eye_dir = eye.normalize();
        // ~500 arrows regardless of subdivision level.
        let stride = (mesh.n_cells() / 500).max(1);
        let stroke = egui::Stroke::new(1.2, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 190));
        for cell in (0..mesh.n_cells()).step_by(stride) {
            if cells.plate_id[cell] == layers::NO_PLATE {
                continue;
            }
            let c = mesh.centers[cell];
            if c.dot(eye_dir) < cos_horizon {
                continue;
            }
            let v = cells.plate_velocity_m_yr[cell];
            let speed_cm_yr = v.length() * 100.0;
            if speed_cm_yr < 0.05 {
                continue;
            }
            // 10 cm/yr reads as ~4 degrees of arc.
            let angle = (speed_cm_yr / 10.0).min(2.0) * 0.07;
            let tip = iw_render_vulkan::camera::great_circle_step(c, v.normalize(), angle);
            let (Some(a), Some(b)) = (
                pick::project_to_pixels(view_proj, c * EARTH_RADIUS_KM, self.surface_size),
                pick::project_to_pixels(view_proj, tip * EARTH_RADIUS_KM, self.surface_size),
            ) else {
                continue;
            };
            let (a, b) = (
                egui::pos2(a.x / ppp, a.y / ppp),
                egui::pos2(b.x / ppp, b.y / ppp),
            );
            if a.distance(b) < 1.5 {
                continue;
            }
            painter.line_segment([a, b], stroke);
            // Arrowhead: two short barbs at the tip.
            let dir = (b - a).normalized();
            let n = egui::vec2(-dir.y, dir.x);
            let head = 4.0f32.min(a.distance(b) * 0.5);
            painter.line_segment([b, b - dir * head + n * head * 0.5], stroke);
            painter.line_segment([b, b - dir * head - n * head * 0.5], stroke);
        }
    }

    fn status_line(&self) -> String {
        match self.sim.as_ref() {
            None => "static test sphere (no simulation)".to_string(),
            Some(sim) => {
                let s = sim.handle.status();
                let phase = match s.state {
                    SimState::Running(p) | SimState::Paused(p) => phase_name(p),
                    SimState::Idle => "idle",
                    SimState::Done => "complete",
                };
                format!("{phase} · {:.1} Myr · {}", s.time_myr, self.layer.name())
            }
        }
    }

    // --- frame -------------------------------------------------------------

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
            self.fps = fps;
            self.frame_ms = 1000.0 * elapsed / self.fps_frames as f32;
            if self.frames_total > 30 {
                self.fps_min = self.fps_min.min(fps);
                self.fps_max = self.fps_max.max(fps);
                if matches!(
                    self.sim.as_ref().map(|s| s.handle.state()),
                    Some(SimState::Running(_))
                ) {
                    self.fps_sum_during_gen += fps as f64;
                    self.fps_samples_during_gen += 1;
                }
            }
            self.fps_frames = 0;
            self.fps_window_start = now;
        }
        self.frames_total += 1;

        self.poll_events();
        self.poll_view()?;
        self.refresh_disk_state();
        self.poll_column();

        let mut actions = FrameActions::default();
        let mut ui = self.ui.take().expect("ui");
        let mut ui_out = ui.run_with(window.as_ref(), |root| self.build_ui(root, &mut actions));
        self.ui = Some(ui);

        if actions.toggle_mode {
            self.toggle_mode();
        }
        if actions.recolor {
            self.refresh_cells()?;
        }
        match actions.scrub.or_else(|| self.pending_scrub.take()) {
            Some(ScrubAction::Load(version)) => self.show_history(version)?,
            Some(ScrubAction::Live) => self.go_live()?,
            _ => {}
        }
        if let Some((phase, config)) = actions.rerun {
            self.apply_config(config, Some(phase));
        } else if let Some(config) = actions
            .regenerate
            .or_else(|| self.pending_regenerate.take())
        {
            self.apply_config(config, None);
        }

        let input = if ui_out.wants_keyboard {
            MoveInput::default()
        } else {
            self.keys.move_input()
        };
        self.camera.update(input, dt);

        let exaggeration = effective_exaggeration(self.exaggeration, self.camera.altitude_km);
        let aspect = self.renderer.as_ref().expect("renderer").aspect();
        let params = GlobeParams {
            view_proj: self.camera.view_proj(aspect),
            camera_pos_km: match self.camera.mode {
                ViewMode::Globe => self.camera.eye(),
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

        let renderer = self.renderer.as_mut().expect("renderer");
        renderer.render(
            &params,
            EguiFrame {
                textures_delta: &mut ui_out.textures_delta,
                primitives: &ui_out.primitives,
                pixels_per_point: ui_out.pixels_per_point,
            },
        )?;

        if let Some(limit) = self.args.exit_after_secs {
            if (now - self.start).as_secs_f32() >= limit {
                let secs = (now - self.start).as_secs_f32();
                log::info!(
                    "exit-after-secs reached: {} frames, {:.1} avg FPS (min {:.1}, max {:.1}), \
                     {} cell uploads",
                    self.frames_total,
                    self.frames_total as f32 / secs,
                    if self.fps_min == f32::MAX {
                        0.0
                    } else {
                        self.fps_min
                    },
                    self.fps_max,
                    self.cell_updates,
                );
                if self.fps_samples_during_gen > 0 {
                    log::info!(
                        "{:.1} avg FPS while generating ({} samples)",
                        self.fps_sum_during_gen / self.fps_samples_during_gen as f64,
                        self.fps_samples_during_gen
                    );
                }
                if let Some(sim) = self.sim.as_ref() {
                    let s = sim.handle.status();
                    log::info!(
                        "simulation reached {:.2} Myr, step {}/{} ({:?})",
                        s.time_myr,
                        s.step_index,
                        s.total_steps,
                        s.state
                    );
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
                self.surface_size = (size.width.max(1), size.height.max(1));
                if let Some(r) = self.renderer.as_mut() {
                    r.request_resize(self.surface_size);
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let (Some(r), Some(w)) = (self.renderer.as_mut(), self.window.as_ref()) {
                    let s = w.inner_size();
                    self.surface_size = (s.width.max(1), s.height.max(1));
                    r.request_resize(self.surface_size);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if consumed {
                    return;
                }
                // Movement is keyed off the *physical* key so WASD stays under
                // the same fingers on any layout; every other shortcut is keyed
                // off the logical key, so it follows the user's layout (and is
                // reachable from synthetic input).
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys.w = pressed,
                        KeyCode::KeyA => self.keys.a = pressed,
                        KeyCode::KeyS => self.keys.s = pressed,
                        KeyCode::KeyD => self.keys.d = pressed,
                        _ => {}
                    }
                }
                if !pressed {
                    return;
                }
                match &event.logical_key {
                    Key::Character(c) => match c.as_str() {
                        "S" => {
                            // Shift+S: single step (plain S is "move south").
                            if let Some(sim) = self.sim.as_ref() {
                                sim.handle.step_once();
                            }
                        }
                        "r" | "R" => self.camera.begin_recenter(),
                        "m" | "M" | "g" | "G" => self.toggle_mode(),
                        "n" | "N" => self.panel.open = !self.panel.open,
                        "i" | "I" => self.pick_centre(),
                        "l" | "L" => self.pending_scrub = Some(ScrubAction::Live),
                        "," | "<" => self.scrub_by(-1),
                        "." | ">" => self.scrub_by(1),
                        "-" | "_" => self.set_layer(Layer::WaterFlux),
                        digit => {
                            if let Some(index) = digit_layer(digit) {
                                self.set_layer(Layer::from_index(index));
                            }
                        }
                    },
                    Key::Named(NamedKey::Space) => self.toggle_pause(),
                    Key::Named(NamedKey::F5) => {
                        // Same as the panel's dice button: new seed, same
                        // settings, regenerate now.
                        self.panel.randomize_seed(panel::entropy());
                        self.pending_regenerate = Some(self.panel.to_config());
                    }
                    Key::Named(NamedKey::F1) => self.show_help = !self.show_help,
                    Key::Named(NamedKey::Tab) => {
                        let next = self.layer.next();
                        self.set_layer(next);
                    }
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if pressed && consumed {
                    return;
                }
                match button {
                    MouseButton::Right => self.right_down = pressed,
                    MouseButton::Left => {
                        self.left_down = pressed;
                        if pressed {
                            self.left_moved = 0.0;
                            self.left_press =
                                self.cursor.map(|c| Vec2::new(c.x as f32, c.y as f32));
                        } else if self.left_moved <= CLICK_SLOP_PX {
                            if let Some(pos) = self.left_press.take() {
                                self.pick_at(pos);
                            }
                        }
                    }
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
        // Stop the worker threads first: the simulation holds nothing the
        // renderer owns, but joining here means no snapshot arrives (or is
        // written) while Vulkan is being torn down.
        if let Some(sim) = self.sim.take() {
            let Sim {
                handle,
                mut history,
                ..
            } = sim;
            handle.shutdown();
            history.shutdown();
        }
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

/// Layer index for a number key, `1`..`9` then `0`. Anything else is `None`.
fn digit_layer(text: &str) -> Option<usize> {
    match text {
        "1" => Some(0),
        "2" => Some(1),
        "3" => Some(2),
        "4" => Some(3),
        "5" => Some(4),
        "6" => Some(5),
        "7" => Some(6),
        "8" => Some(7),
        "9" => Some(8),
        "0" => Some(9),
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_keys_map_to_the_first_ten_layers() {
        assert_eq!(digit_layer("1"), Some(0));
        assert_eq!(digit_layer("9"), Some(8));
        assert_eq!(digit_layer("0"), Some(9));
        assert_eq!(digit_layer("q"), None);
        assert_eq!(digit_layer(""), None);
        for i in 0..10 {
            assert!(Layer::from_index(i).index() == i);
        }
        // The eleventh layer has no digit; it is reachable with `-` and Tab.
        assert_eq!(Layer::ALL.len(), 11);
    }

    #[test]
    fn the_default_planet_directory_is_named_by_seed_and_level() {
        let mut app = App::new(Args::parse_from(["iw-app", "--seed", "7", "--level", "5"]));
        let dir = app.planet_dir(&app.config.clone());
        assert!(dir.ends_with("7-L5"), "{dir:?}");
        app.args.out = Some(PathBuf::from("/tmp/elsewhere"));
        assert_eq!(
            app.planet_dir(&app.config.clone()),
            PathBuf::from("/tmp/elsewhere")
        );
    }

    #[test]
    fn cli_defaults_match_the_smoke_path() {
        let args = Args::parse_from(["iw-app", "--exit-after-secs", "3"]);
        assert_eq!(args.level, 6);
        assert_eq!(args.seed, 42);
        assert_eq!(args.exit_after_secs, Some(3.0));
        assert!(!args.mercator && !args.test_sphere && !args.paused);
    }
}
