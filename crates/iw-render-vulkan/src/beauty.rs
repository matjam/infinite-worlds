//! Beauty-view lighting parameters (WP11, DESIGN.md §9).
//!
//! This module owns the CPU half of the beauty knobs: where the sun is, how
//! fast the cloud deck drifts, and how the atmosphere fades out on descent.
//! The GPU half — every colour, exponent and strength used inside a shader —
//! lives in `shaders/beauty.glsl`; the albedo palette and the per-cell shading
//! inputs live in `crates/iw-app/src/beauty.rs`.
//!
//! Everything here is pure math: no GPU, unit tested.

use glam::Vec3;

/// Angle between the camera axis and the sun in camera-relative mode. Small
/// enough that the disc is almost fully lit (the Blue Marble was shot from
/// nearly straight down-sun), large enough for relief to cast visible shading.
pub const SUN_TILT_DEG: f32 = 30.0;
/// Declination of the sun in fixed mode: a northern-summer-ish angle that puts
/// the terminator across a pleasing diagonal.
pub const SUN_DECLINATION_DEG: f32 = 20.0;
/// Default azimuth of the sun offset, degrees. 0 puts the sun to the east of
/// the view axis, i.e. screen right with north up.
pub const DEFAULT_SUN_AZIMUTH_DEG: f32 = 0.0;

/// Altitude (km) below which the atmospheric halo and limb are gone entirely.
pub const HAZE_FADE_LO_KM: f32 = 200.0;
/// Altitude (km) above which they are at full strength. The half-way point sits
/// at ~450 km, so a near-surface fly-through is never washed out.
pub const HAZE_FADE_HI_KM: f32 = 700.0;

/// Drift of the cloud deck relative to the surface, degrees of longitude per
/// minute of wall time. Slow enough to read as weather, not as a spinning
/// texture.
pub const CLOUD_ROTATION_DEG_PER_MIN: f32 = 0.5;

/// Where the sun is anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SunMode {
    /// Sun sits [`SUN_TILT_DEG`] off the camera axis, so the globe is lit from
    /// behind the viewer however the camera moves.
    #[default]
    CameraRelative,
    /// Sun is fixed in planet coordinates; orbiting the globe walks the
    /// terminator across the view.
    Fixed,
}

impl SunMode {
    /// Label for the UI toggle.
    pub fn name(self) -> &'static str {
        match self {
            SunMode::CameraRelative => "camera-relative sun",
            SunMode::Fixed => "fixed sun",
        }
    }
}

/// User-facing sun controls (one slider and one toggle).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunSettings {
    pub mode: SunMode,
    /// Camera-relative: direction of the offset around the view axis.
    /// Fixed: longitude of the sub-solar point. Degrees.
    pub azimuth_deg: f32,
}

impl Default for SunSettings {
    fn default() -> Self {
        SunSettings {
            mode: SunMode::default(),
            azimuth_deg: DEFAULT_SUN_AZIMUTH_DEG,
        }
    }
}

/// Unit vector from the surface *towards* the sun.
///
/// `eye` is the camera position in planet coordinates (km); only its direction
/// matters, and a degenerate (zero) eye falls back to the fixed sun so the
/// caller never has to special-case start-up.
pub fn sun_direction(settings: SunSettings, eye: Vec3) -> Vec3 {
    let az = settings.azimuth_deg.to_radians();
    if settings.mode == SunMode::Fixed || eye.length_squared() < 1e-12 {
        let dec = SUN_DECLINATION_DEG.to_radians();
        return Vec3::new(dec.cos() * az.cos(), dec.cos() * az.sin(), dec.sin()).normalize();
    }
    let up = eye.normalize();
    let east = {
        let e = Vec3::Z.cross(up);
        if e.length_squared() < 1e-12 {
            Vec3::X
        } else {
            e.normalize()
        }
    };
    let north = up.cross(east).normalize();
    let tilt = SUN_TILT_DEG.to_radians();
    (up * tilt.cos() + (east * az.cos() + north * az.sin()) * tilt.sin()).normalize()
}

/// Strength of the atmospheric halo and limb at this altitude, 0..1. Fades to
/// nothing on descent so near-surface views stay crisp.
pub fn haze_fade(altitude_km: f32) -> f32 {
    let t = ((altitude_km - HAZE_FADE_LO_KM) / (HAZE_FADE_HI_KM - HAZE_FADE_LO_KM)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Cloud-deck rotation phase (radians) after `elapsed_secs` of wall time.
pub fn cloud_phase_rad(elapsed_secs: f32) -> f32 {
    (CLOUD_ROTATION_DEG_PER_MIN * elapsed_secs / 60.0).to_radians()
}

/// Blinn-Phong specular lobe, the Rust twin of the ocean glint in
/// `globe.frag`: `n` surface normal, `l` towards the light, `v` towards the
/// viewer, all unit. Peaks when the half vector aligns with the normal, i.e.
/// when `v` is the mirror reflection of `l`.
pub fn specular_blinn(n: Vec3, l: Vec3, v: Vec3, shininess: f32) -> f32 {
    let h = (l + v).normalize_or_zero();
    n.dot(h).max(0.0).powf(shininess)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_relative_sun_tracks_the_camera() {
        let s = SunSettings::default();
        for eye in [
            Vec3::new(0.0, 0.0, 8000.0),
            Vec3::new(7000.0, 0.0, 0.0),
            Vec3::new(-3000.0, 4000.0, 2000.0),
        ] {
            let l = sun_direction(s, eye);
            assert!((l.length() - 1.0).abs() < 1e-5);
            // Sun is SUN_TILT_DEG off the camera axis, so the sub-camera point
            // is always in daylight.
            let cos = l.dot(eye.normalize());
            assert!(
                (cos - SUN_TILT_DEG.to_radians().cos()).abs() < 1e-4,
                "tilt wrong for {eye}: {cos}"
            );
        }
    }

    #[test]
    fn azimuth_rotates_the_offset_around_the_view_axis() {
        let eye = Vec3::new(0.0, 0.0, 6371.0 + 8000.0);
        let up = eye.normalize();
        // At azimuth 0 the offset is due east (+x at the north pole basis).
        let east_lit = sun_direction(SunSettings::default(), eye);
        let offset = east_lit - up * east_lit.dot(up);
        assert!(offset.x > 0.9 * offset.length(), "{offset}");
        // 90 degrees later it is due north.
        let north_lit = sun_direction(
            SunSettings {
                azimuth_deg: 90.0,
                ..SunSettings::default()
            },
            eye,
        );
        let offset = north_lit - up * north_lit.dot(up);
        assert!(offset.y > 0.9 * offset.length(), "{offset}");
        // Half a turn is the mirror image.
        let back = sun_direction(
            SunSettings {
                azimuth_deg: 180.0,
                ..SunSettings::default()
            },
            eye,
        );
        assert!((back.x + east_lit.x).abs() < 1e-5);
    }

    #[test]
    fn fixed_sun_ignores_the_camera() {
        let s = SunSettings {
            mode: SunMode::Fixed,
            azimuth_deg: 0.0,
        };
        let a = sun_direction(s, Vec3::new(0.0, 0.0, 9000.0));
        let b = sun_direction(s, Vec3::new(9000.0, 0.0, 0.0));
        assert!((a - b).length() < 1e-6);
        assert!(a.x > 0.0 && a.z > 0.0, "{a}");
        // A degenerate camera position must not produce NaN.
        let c = sun_direction(SunSettings::default(), Vec3::ZERO);
        assert!(c.is_finite() && (c.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn haze_fades_out_on_descent() {
        assert_eq!(haze_fade(0.0), 0.0);
        assert_eq!(haze_fade(HAZE_FADE_LO_KM), 0.0);
        assert_eq!(haze_fade(HAZE_FADE_HI_KM), 1.0);
        assert_eq!(haze_fade(20_000.0), 1.0);
        let mid = haze_fade(450.0);
        assert!((0.4..=0.6).contains(&mid), "{mid}");
        // Monotone in altitude.
        let mut prev = -1.0;
        for km in 0..1000 {
            let f = haze_fade(km as f32);
            assert!(f >= prev - 1e-6);
            prev = f;
        }
    }

    #[test]
    fn glint_peaks_at_the_reflection_vector() {
        let n = Vec3::Z;
        let l = Vec3::new(0.6, 0.0, 0.8).normalize();
        // Mirror reflection of l about n.
        let r = (2.0 * n.dot(l) * n - l).normalize();
        let peak = specular_blinn(n, l, r, 320.0);
        assert!((peak - 1.0).abs() < 1e-4, "{peak}");
        // Anything off the reflection direction is dimmer, and the lobe is
        // tight: one degree away is already far down.
        for deg in [1.0f32, 5.0, 20.0, 60.0] {
            let a = deg.to_radians();
            let off = (r * a.cos() + Vec3::X.cross(r).normalize() * a.sin()).normalize();
            let v = specular_blinn(n, l, off, 320.0);
            assert!(v < peak, "{deg} deg: {v} >= {peak}");
        }
        assert!(specular_blinn(n, l, Vec3::new(0.0, 0.0, -1.0), 320.0) < 1e-6);
    }

    #[test]
    fn cloud_phase_is_slow_and_linear() {
        assert_eq!(cloud_phase_rad(0.0), 0.0);
        let one_minute = cloud_phase_rad(60.0);
        assert!((one_minute.to_degrees() - CLOUD_ROTATION_DEG_PER_MIN).abs() < 1e-5);
        assert!((cloud_phase_rad(120.0) - 2.0 * one_minute).abs() < 1e-6);
    }
}
