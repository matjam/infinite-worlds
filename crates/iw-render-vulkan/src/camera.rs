//! Camera state and the pure math behind it (DESIGN.md §9).
//!
//! One camera, whose behaviour blends with altitude:
//! * the focus point is a unit direction on the sphere; WASD moves it along
//!   great circles (north/west/south/east),
//! * high up, input accelerates an angular velocity that damps exponentially
//!   after release (flick and glide); low down, velocity is direct and scaled
//!   by altitude so travel feels constant,
//! * right-mouse drag adds persistent free-look pitch/yaw offsets, eased back
//!   to zero by the recenter key,
//! * scroll changes altitude multiplicatively,
//! * vertical exaggeration eases toward 1x below `EXAG_EASE_ALT_KM`.
//!
//! Everything here is CPU-side and GPU-free so it can be unit tested.

use glam::{Mat4, Vec2, Vec3};
use iw_mesh::EARTH_RADIUS_KM;

/// Lowest altitude the camera may descend to, km.
pub const MIN_ALTITUDE_KM: f32 = 50.0;
/// Highest altitude the camera may rise to, km.
pub const MAX_ALTITUDE_KM: f32 = 40_000.0;
/// Below this altitude WASD is direct velocity; above it, momentum.
pub const MOMENTUM_LO_KM: f32 = 1_000.0;
/// Above this altitude WASD is pure momentum.
pub const MOMENTUM_HI_KM: f32 = 3_000.0;
/// Half-life of the angular-velocity decay after key release, seconds.
pub const MOMENTUM_HALF_LIFE_S: f32 = 2.0;
/// Angular acceleration applied by a held key in momentum mode, rad/s^2.
pub const MOMENTUM_ACCEL_RAD_S2: f32 = 0.55;
/// Ground speed per unit altitude in direct mode, 1/s (speed_km_s = alt_km * k).
pub const DIRECT_SPEED_PER_ALT: f32 = 1.2;
/// Seconds the recenter key takes to ease free-look offsets back to zero.
pub const RECENTER_SECS: f32 = 0.5;
/// Vertical exaggeration is fully applied at or above this altitude.
pub const EXAG_EASE_ALT_KM: f32 = 500.0;
/// Free-look pitch is clamped to this many radians from nadir.
pub const MAX_PITCH_RAD: f32 = 1.55;
/// Vertical field of view, radians.
pub const FOV_Y_RAD: f32 = 0.9;

/// Which view projection the renderer is drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// 3D globe, perspective, reverse-Z.
    Globe,
    /// Mercator plane, orthographic, 2D pan/zoom.
    Mercator,
}

/// One frame's worth of movement input, already resolved from key state.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveInput {
    /// +1 = north (W), -1 = south (S).
    pub north: f32,
    /// +1 = east (D), -1 = west (A).
    pub east: f32,
}

impl MoveInput {
    /// Length-limited input vector, so diagonals aren't faster.
    pub fn clamped(self) -> Vec2 {
        let v = Vec2::new(self.east, self.north);
        if v.length_squared() > 1.0 {
            v.normalize()
        } else {
            v
        }
    }

    /// True when no direction is held.
    pub fn is_zero(self) -> bool {
        self.north == 0.0 && self.east == 0.0
    }
}

/// Camera state. Mutated by input each frame, read by the renderer.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Unit direction of the point the camera is over.
    pub focus: Vec3,
    /// Height above the reference sphere, km.
    pub altitude_km: f32,
    /// Free-look yaw offset (radians, about the local up).
    pub yaw_rad: f32,
    /// Free-look pitch offset (radians from straight-down).
    pub pitch_rad: f32,
    /// Angular velocity of the focus point: x = east, y = north, rad/s.
    pub angular_velocity: Vec2,
    /// Remaining seconds of the recenter ease, 0 when idle.
    recenter_left_s: f32,
    /// Free-look offsets at the moment recentering started.
    recenter_from: Vec2,
    /// Mercator pan centre: x = longitude (rad), y = mercator y.
    pub mercator_center: Vec2,
    /// Mercator half-width of the viewport in projection units.
    pub mercator_half_width: f32,
    /// Current view mode.
    pub mode: ViewMode,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            focus: Vec3::new(1.0, 0.0, 0.0),
            altitude_km: 18_000.0,
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            angular_velocity: Vec2::ZERO,
            recenter_left_s: 0.0,
            recenter_from: Vec2::ZERO,
            mercator_center: Vec2::ZERO,
            mercator_half_width: std::f32::consts::PI,
            mode: ViewMode::Globe,
        }
    }
}

/// Local tangent basis at a unit direction: (east, north). +z is the pole.
pub fn east_north(dir: Vec3) -> (Vec3, Vec3) {
    let up = dir.normalize();
    let mut east = Vec3::Z.cross(up);
    if east.length_squared() < 1e-12 {
        // At a pole any basis will do; pick a deterministic one.
        east = Vec3::X;
    }
    let east = east.normalize();
    let north = up.cross(east).normalize();
    (east, north)
}

/// Rotate `dir` along the great circle through it in tangent direction
/// `tangent` by `angle_rad`. Result stays a unit vector.
pub fn great_circle_step(dir: Vec3, tangent: Vec3, angle_rad: f32) -> Vec3 {
    if angle_rad == 0.0 || tangent.length_squared() < 1e-18 {
        return dir.normalize();
    }
    let t = tangent.normalize();
    (dir * angle_rad.cos() + t * angle_rad.sin()).normalize()
}

/// Blend factor between direct velocity (0) and momentum (1) at an altitude.
pub fn momentum_blend(altitude_km: f32) -> f32 {
    let t = ((altitude_km - MOMENTUM_LO_KM) / (MOMENTUM_HI_KM - MOMENTUM_LO_KM)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Direct-mode angular rate at an altitude, rad/s. Speed over the ground is
/// proportional to altitude so travel feels the same at every zoom level.
pub fn direct_rate(altitude_km: f32) -> f32 {
    (altitude_km * DIRECT_SPEED_PER_ALT) / EARTH_RADIUS_KM
}

/// Exponential decay factor for `dt` seconds at the momentum half-life.
pub fn damping(dt: f32) -> f32 {
    0.5f32.powf(dt / MOMENTUM_HALF_LIFE_S)
}

/// Vertical exaggeration actually applied at an altitude: eases to 1x on
/// approach so near-surface terrain doesn't look cartoonish.
pub fn effective_exaggeration(user_exaggeration: f32, altitude_km: f32) -> f32 {
    let t = (altitude_km / EXAG_EASE_ALT_KM).clamp(0.0, 1.0);
    1.0 + (user_exaggeration - 1.0) * t
}

/// Smoothstep on [0,1].
fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Camera {
    /// Distance from the planet centre, km.
    #[inline]
    pub fn radius_km(&self) -> f32 {
        EARTH_RADIUS_KM + self.altitude_km
    }

    /// World-space eye position, km.
    #[inline]
    pub fn eye(&self) -> Vec3 {
        self.focus.normalize() * self.radius_km()
    }

    /// Camera basis (forward, up) including free-look offsets. At zero offsets
    /// the camera looks straight down with north up-screen and east right.
    pub fn orientation(&self) -> (Vec3, Vec3) {
        let up = self.focus.normalize();
        let (east, north) = east_north(up);
        let heading = north * self.yaw_rad.cos() + east * self.yaw_rad.sin();
        let (sp, cp) = self.pitch_rad.sin_cos();
        let forward = (-up * cp + heading * sp).normalize();
        let cam_up = (up * sp + heading * cp).normalize();
        (forward, cam_up)
    }

    /// Right-handed view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye();
        let (forward, up) = self.orientation();
        Mat4::look_at_rh(eye, eye + forward, up)
    }

    /// Near plane distance, km. Scaled with altitude; reverse-Z keeps the
    /// precision even with an infinite far plane.
    pub fn near_km(&self) -> f32 {
        (self.altitude_km * 0.02).clamp(0.05, 200.0)
    }

    /// Reverse-Z infinite perspective projection with the Vulkan Y flip.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let mut p =
            Mat4::perspective_infinite_reverse_rh(FOV_Y_RAD, aspect.max(1e-3), self.near_km());
        p.y_axis.y *= -1.0;
        p
    }

    /// Orthographic Mercator projection with the Vulkan Y flip. Depth is
    /// written directly by the vertex shader, so z maps identity here.
    pub fn mercator_projection(&self, aspect: f32) -> Mat4 {
        let hw = self.mercator_half_width.max(1e-4);
        let hh = hw / aspect.max(1e-3);
        let c = self.mercator_center;
        let mut p = Mat4::orthographic_rh(c.x - hw, c.x + hw, c.y - hh, c.y + hh, -10.0, 10.0);
        p.y_axis.y *= -1.0;
        p
    }

    /// Combined view-projection for the current mode.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        match self.mode {
            ViewMode::Globe => self.projection(aspect) * self.view_matrix(),
            ViewMode::Mercator => self.mercator_projection(aspect),
        }
    }

    /// Multiplicative zoom along the view ray. `steps` is scroll notches:
    /// positive zooms in.
    pub fn zoom(&mut self, steps: f32) {
        match self.mode {
            ViewMode::Globe => {
                self.altitude_km = (self.altitude_km * (0.86f32).powf(steps))
                    .clamp(MIN_ALTITUDE_KM, MAX_ALTITUDE_KM);
            }
            ViewMode::Mercator => {
                self.mercator_half_width = (self.mercator_half_width * (0.86f32).powf(steps))
                    .clamp(0.002, std::f32::consts::PI);
            }
        }
    }

    /// Accumulate a free-look drag, in radians of yaw and pitch.
    pub fn free_look(&mut self, d_yaw: f32, d_pitch: f32) {
        self.recenter_left_s = 0.0;
        self.yaw_rad = (self.yaw_rad + d_yaw).rem_euclid(std::f32::consts::TAU);
        self.pitch_rad = (self.pitch_rad + d_pitch).clamp(0.0, MAX_PITCH_RAD);
    }

    /// Start easing the free-look offsets back to straight-down framing.
    pub fn begin_recenter(&mut self) {
        // Take the shortest way round for yaw.
        let mut yaw = self.yaw_rad.rem_euclid(std::f32::consts::TAU);
        if yaw > std::f32::consts::PI {
            yaw -= std::f32::consts::TAU;
        }
        self.yaw_rad = yaw;
        self.recenter_from = Vec2::new(yaw, self.pitch_rad);
        self.recenter_left_s = RECENTER_SECS;
    }

    /// True while a recenter ease is running.
    pub fn is_recentering(&self) -> bool {
        self.recenter_left_s > 0.0
    }

    /// Pan the Mercator view by a projection-space delta.
    pub fn mercator_pan(&mut self, delta: Vec2) {
        self.mercator_center += delta;
        let limit = 3.2; // slightly past the +/-85 deg mercator y extent
        self.mercator_center.y = self.mercator_center.y.clamp(-limit, limit);
        self.mercator_center.x = wrap_pi(self.mercator_center.x);
    }

    /// Advance one frame of camera simulation.
    pub fn update(&mut self, input: MoveInput, dt: f32) {
        let dt = dt.clamp(0.0, 0.25);
        if dt <= 0.0 {
            return;
        }

        if self.mode == ViewMode::Mercator {
            // WASD pans the plane; speed scales with the zoom level.
            let v = input.clamped();
            let rate = self.mercator_half_width * 0.9;
            self.mercator_pan(Vec2::new(v.x, v.y) * rate * dt);
        } else {
            let v = input.clamped();
            let blend = momentum_blend(self.altitude_km);
            let damped = self.angular_velocity * damping(dt);
            let momentum = damped + v * MOMENTUM_ACCEL_RAD_S2 * dt;
            let direct = v * direct_rate(self.altitude_km);
            self.angular_velocity = momentum * blend + direct * (1.0 - blend);

            let step = self.angular_velocity * dt;
            if step.length_squared() > 0.0 {
                let (east, north) = east_north(self.focus);
                let tangent = east * step.x + north * step.y;
                self.focus = great_circle_step(self.focus, tangent, step.length());
            }
        }

        if self.recenter_left_s > 0.0 {
            self.recenter_left_s = (self.recenter_left_s - dt).max(0.0);
            let s = smoothstep01(1.0 - self.recenter_left_s / RECENTER_SECS);
            self.yaw_rad = self.recenter_from.x * (1.0 - s);
            self.pitch_rad = self.recenter_from.y * (1.0 - s);
            if self.recenter_left_s == 0.0 {
                self.yaw_rad = 0.0;
                self.pitch_rad = 0.0;
            }
        }
    }
}

/// Wrap an angle into [-pi, pi).
#[inline]
pub fn wrap_pi(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    a - TAU * ((a + PI) / TAU).floor()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EQUATOR: Vec3 = Vec3::new(1.0, 0.0, 0.0);

    fn held(north: f32, east: f32) -> MoveInput {
        MoveInput { north, east }
    }

    #[test]
    fn basis_is_orthonormal_and_oriented() {
        let (e, n) = east_north(EQUATOR);
        assert!((e.length() - 1.0).abs() < 1e-5);
        assert!((n.length() - 1.0).abs() < 1e-5);
        assert!(e.dot(n).abs() < 1e-5);
        // At (1,0,0) east is +y and north is +z.
        assert!((e - Vec3::Y).length() < 1e-5, "east was {e}");
        assert!((n - Vec3::Z).length() < 1e-5, "north was {n}");
    }

    #[test]
    fn basis_at_pole_is_finite() {
        let (e, n) = east_north(Vec3::Z);
        assert!(e.is_finite() && n.is_finite());
        assert!((e.length() - 1.0).abs() < 1e-5);
        assert!((n.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn great_circle_step_preserves_unit_length() {
        let (e, n) = east_north(EQUATOR);
        for a in [0.01f32, 0.5, 1.5, 3.0, -2.0] {
            let p = great_circle_step(EQUATOR, e * 0.3 + n * 0.7, a);
            assert!((p.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn quarter_turn_north_reaches_the_pole() {
        let (_, n) = east_north(EQUATOR);
        let p = great_circle_step(EQUATOR, n, std::f32::consts::FRAC_PI_2);
        assert!((p - Vec3::Z).length() < 1e-5, "got {p}");
    }

    #[test]
    fn wasd_moves_the_right_way_from_the_equator() {
        // Low altitude => direct velocity, no momentum ramp.
        let mut cam = Camera {
            focus: EQUATOR,
            altitude_km: 200.0,
            ..Default::default()
        };
        cam.update(held(1.0, 0.0), 0.1);
        assert!(cam.focus.z > 0.0, "W should move north: {}", cam.focus);
        assert!((cam.focus.length() - 1.0).abs() < 1e-5);

        let mut cam = Camera {
            focus: EQUATOR,
            altitude_km: 200.0,
            ..Default::default()
        };
        cam.update(held(-1.0, 0.0), 0.1);
        assert!(cam.focus.z < 0.0, "S should move south");

        let mut cam = Camera {
            focus: EQUATOR,
            altitude_km: 200.0,
            ..Default::default()
        };
        cam.update(held(0.0, 1.0), 0.1);
        assert!(cam.focus.y > 0.0, "D should move east (+y at lon 0)");

        let mut cam = Camera {
            focus: EQUATOR,
            altitude_km: 200.0,
            ..Default::default()
        };
        cam.update(held(0.0, -1.0), 0.1);
        assert!(cam.focus.y < 0.0, "A should move west");
    }

    #[test]
    fn wasd_from_the_pole_stays_on_the_sphere() {
        let mut cam = Camera {
            focus: Vec3::Z,
            altitude_km: 300.0,
            ..Default::default()
        };
        let mut min_z = 1.0f32;
        for _ in 0..20 {
            cam.update(held(1.0, 0.3), 0.05);
            assert!((cam.focus.length() - 1.0).abs() < 1e-4);
            assert!(cam.focus.is_finite());
            min_z = min_z.min(cam.focus.z);
        }
        // Travelling "north" must take us off the pole (and, correctly, back
        // over it, since north reverses once past 90 degrees).
        assert!(min_z < 1.0 - 1e-6, "never left the pole");
    }

    #[test]
    fn direct_mode_stops_immediately_on_release() {
        let mut cam = Camera {
            focus: EQUATOR,
            altitude_km: 100.0,
            ..Default::default()
        };
        cam.update(held(1.0, 0.0), 0.1);
        assert!(cam.angular_velocity.length() > 0.0);
        cam.update(MoveInput::default(), 0.1);
        assert!(
            cam.angular_velocity.length() < 1e-6,
            "low altitude must not glide: {}",
            cam.angular_velocity
        );
    }

    #[test]
    fn momentum_damps_by_half_over_the_half_life() {
        let mut cam = Camera {
            focus: EQUATOR,
            altitude_km: 20_000.0,
            angular_velocity: Vec2::new(0.4, 0.0),
            ..Default::default()
        };
        let v0 = cam.angular_velocity.length();
        let dt = 1.0 / 120.0;
        let mut t = 0.0;
        while t < MOMENTUM_HALF_LIFE_S {
            cam.update(MoveInput::default(), dt);
            t += dt;
        }
        let v1 = cam.angular_velocity.length();
        assert!(
            (v1 / v0 - 0.5).abs() < 0.02,
            "expected ~half after {MOMENTUM_HALF_LIFE_S}s, got {}",
            v1 / v0
        );
    }

    #[test]
    fn momentum_glides_after_release_and_eventually_stops() {
        let mut cam = Camera {
            focus: EQUATOR,
            altitude_km: 30_000.0,
            ..Default::default()
        };
        for _ in 0..60 {
            cam.update(held(1.0, 0.0), 1.0 / 60.0);
        }
        let after_hold = cam.focus;
        assert!(cam.angular_velocity.length() > 0.05);
        for _ in 0..60 {
            cam.update(MoveInput::default(), 1.0 / 60.0);
        }
        assert!(cam.focus.distance(after_hold) > 1e-3, "should coast");
        for _ in 0..(60 * 30) {
            cam.update(MoveInput::default(), 1.0 / 60.0);
        }
        assert!(cam.angular_velocity.length() < 1e-4, "should settle");
    }

    #[test]
    fn momentum_blend_is_monotonic_and_saturating() {
        assert_eq!(momentum_blend(0.0), 0.0);
        assert_eq!(momentum_blend(MOMENTUM_LO_KM), 0.0);
        assert_eq!(momentum_blend(MOMENTUM_HI_KM), 1.0);
        assert_eq!(momentum_blend(MAX_ALTITUDE_KM), 1.0);
        let mut prev = -1.0;
        for i in 0..=100 {
            let v = momentum_blend(i as f32 * 400.0);
            assert!(v >= prev - 1e-6);
            prev = v;
        }
    }

    #[test]
    fn free_look_persists_until_recentered_then_converges() {
        let mut cam = Camera::default();
        cam.free_look(0.7, 0.9);
        for _ in 0..30 {
            cam.update(MoveInput::default(), 1.0 / 60.0);
        }
        assert!((cam.yaw_rad - 0.7).abs() < 1e-5, "offsets must persist");
        assert!((cam.pitch_rad - 0.9).abs() < 1e-5);

        cam.begin_recenter();
        let mut t = 0.0;
        while t < RECENTER_SECS {
            cam.update(MoveInput::default(), 1.0 / 240.0);
            t += 1.0 / 240.0;
        }
        assert!(cam.yaw_rad.abs() < 1e-5, "yaw {}", cam.yaw_rad);
        assert!(cam.pitch_rad.abs() < 1e-5, "pitch {}", cam.pitch_rad);
        assert!(!cam.is_recentering());
    }

    #[test]
    fn recenter_is_monotone_toward_zero() {
        let mut cam = Camera::default();
        cam.free_look(0.0, 1.2);
        cam.begin_recenter();
        let mut prev = cam.pitch_rad;
        for _ in 0..60 {
            cam.update(MoveInput::default(), 1.0 / 120.0);
            assert!(cam.pitch_rad <= prev + 1e-6);
            prev = cam.pitch_rad;
        }
        assert!(prev.abs() < 1e-5);
    }

    #[test]
    fn free_look_pitch_is_clamped() {
        let mut cam = Camera::default();
        cam.free_look(0.0, 100.0);
        assert!(cam.pitch_rad <= MAX_PITCH_RAD);
        cam.free_look(0.0, -100.0);
        assert!(cam.pitch_rad >= 0.0);
    }

    #[test]
    fn zoom_is_multiplicative_and_clamped() {
        let mut cam = Camera::default();
        let a0 = cam.altitude_km;
        cam.zoom(1.0);
        assert!(cam.altitude_km < a0);
        cam.zoom(-1.0);
        assert!((cam.altitude_km - a0).abs() < a0 * 1e-4);
        for _ in 0..500 {
            cam.zoom(1.0);
        }
        assert_eq!(cam.altitude_km, MIN_ALTITUDE_KM);
        for _ in 0..500 {
            cam.zoom(-1.0);
        }
        assert_eq!(cam.altitude_km, MAX_ALTITUDE_KM);
    }

    #[test]
    fn orientation_at_zero_offsets_looks_down_with_north_up() {
        let cam = Camera {
            focus: EQUATOR,
            ..Default::default()
        };
        let (fwd, up) = cam.orientation();
        assert!((fwd + EQUATOR).length() < 1e-5, "forward {fwd}");
        assert!((up - Vec3::Z).length() < 1e-5, "up {up}");
        // Screen-right (RH look-at convention) must be east.
        let right = fwd.cross(up).normalize();
        assert!((right - Vec3::Y).length() < 1e-5, "right {right}");
    }

    #[test]
    fn orientation_at_full_pitch_looks_along_the_horizon() {
        let cam = Camera {
            focus: EQUATOR,
            pitch_rad: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let (fwd, up) = cam.orientation();
        assert!(fwd.dot(EQUATOR).abs() < 1e-5, "forward must be tangent");
        assert!((up - EQUATOR).length() < 1e-5, "up must be radial");
    }

    #[test]
    fn reverse_z_projection_maps_near_to_one() {
        let cam = Camera::default();
        let p = cam.projection(16.0 / 9.0);
        let near_pt = glam::Vec4::new(0.0, 0.0, -cam.near_km(), 1.0);
        let clip = p * near_pt;
        assert!(
            (clip.z / clip.w - 1.0).abs() < 1e-3,
            "near depth {}",
            clip.z / clip.w
        );
        let far_pt = glam::Vec4::new(0.0, 0.0, -1.0e7, 1.0);
        let clip = p * far_pt;
        let d = clip.z / clip.w;
        assert!((0.0..0.01).contains(&d), "far depth {d}");
    }

    #[test]
    fn exaggeration_eases_to_one_near_the_surface() {
        assert!((effective_exaggeration(50.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((effective_exaggeration(50.0, EXAG_EASE_ALT_KM) - 50.0).abs() < 1e-4);
        assert!((effective_exaggeration(50.0, 20_000.0) - 50.0).abs() < 1e-4);
        let mid = effective_exaggeration(50.0, EXAG_EASE_ALT_KM * 0.5);
        assert!(mid > 1.0 && mid < 50.0);
        // Monotone in altitude for exaggeration > 1.
        let mut prev = 0.0;
        for i in 0..=100 {
            let v = effective_exaggeration(20.0, i as f32 * 10.0);
            assert!(v >= prev - 1e-6);
            prev = v;
        }
    }

    #[test]
    fn direct_rate_scales_with_altitude() {
        assert!(direct_rate(100.0) < direct_rate(400.0));
        assert!((direct_rate(0.0)).abs() < 1e-9);
    }

    #[test]
    fn wrap_pi_wraps() {
        use std::f32::consts::PI;
        assert!((wrap_pi(0.0)).abs() < 1e-6);
        assert!((wrap_pi(PI + 0.5) - (-PI + 0.5)).abs() < 1e-5);
        assert!((wrap_pi(-PI - 0.5) - (PI - 0.5)).abs() < 1e-5);
        for i in -50..50 {
            let a = i as f32 * 0.37;
            let w = wrap_pi(a);
            assert!((-PI..PI).contains(&w), "{a} -> {w}");
        }
    }
}
