//! Spherical-geometry helpers shared by the tectonics passes.
//!
//! Everything here is a pure function of its arguments: no hidden state, no
//! global tables, so the whole crate stays reconstructible from `Planet`.

use glam::{DVec3, Vec3};
use iw_mesh::{Mesh, EARTH_RADIUS_M};
use rand::Rng;
use rand_pcg::Pcg64Mcg;

/// Rodrigues rotation of `v` about the unit axis `axis` by `angle` radians.
pub(crate) fn rotate_about(v: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

/// Uniformly distributed unit vector (area-preserving z/phi sampling).
pub(crate) fn random_unit(rng: &mut Pcg64Mcg) -> Vec3 {
    let z: f32 = rng.random_range(-1.0f32..1.0);
    let phi: f32 = rng.random_range(0.0f32..std::f32::consts::TAU);
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

/// Some unit tangent at `r`; deterministic, direction otherwise unspecified.
pub(crate) fn any_tangent(r: Vec3) -> Vec3 {
    let a = if r.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    a.cross(r).normalize()
}

/// Unit tangent at `r` pointing along the great circle toward `target`.
/// Falls back to [`any_tangent`] when the two are (anti)parallel.
pub(crate) fn tangent_toward(r: Vec3, target: Vec3) -> Vec3 {
    let t = target - r * r.dot(target);
    if t.length_squared() > 1e-14 {
        t.normalize()
    } else {
        any_tangent(r)
    }
}

/// Re-project `dir` onto the tangent plane at `r` and renormalize. Used to
/// carry a heading from cell to cell when walking a great circle.
pub(crate) fn reproject(dir: Vec3, r: Vec3) -> Vec3 {
    let t = dir - r * r.dot(dir);
    if t.length_squared() > 1e-14 {
        t.normalize()
    } else {
        any_tangent(r)
    }
}

/// Angular velocity (rad/Myr) whose surface velocity at unit position `r` is
/// `v_m_yr` (must be tangent at `r`).
///
/// Inverse of [`iw_core::Plate::velocity_m_yr`]: `(w x r) * R / 1e6 == v`.
pub(crate) fn omega_for_velocity(r: Vec3, v_m_yr: Vec3) -> DVec3 {
    r.as_dvec3().cross(v_m_yr.as_dvec3()) * (1.0e6 / EARTH_RADIUS_M)
}

/// Mean centre-to-centre cell spacing in metres, from the hexagonal relation
/// `area = (sqrt(3)/2) * pitch^2`.
pub(crate) fn cell_pitch_m(mesh: &Mesh) -> f64 {
    let total_km2: f64 = mesh.areas_km2.iter().map(|a| *a as f64).sum();
    let mean_m2 = total_km2 * 1.0e6 / mesh.n_cells() as f64;
    (2.0 * mean_m2 / 3.0f64.sqrt()).sqrt()
}

/// Great-circle distance between unit vectors, in metres.
pub(crate) fn arc_m(a: Vec3, b: Vec3) -> f64 {
    (a.dot(b).clamp(-1.0, 1.0) as f64).acos() * EARTH_RADIUS_M
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_round_trip() {
        // Round-trips against the real contract in iw-core, not a local copy.
        let r = Vec3::new(0.3, -0.5, 0.8).normalize();
        let v = tangent_toward(r, Vec3::Z) * 0.05;
        let w = omega_for_velocity(r, v);
        let plate = iw_core::Plate {
            euler_pole: w.normalize(),
            omega_rad_myr: w.length(),
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
        };
        let back = plate.velocity_m_yr(r);
        assert!((back - v).length() < 1e-6, "{back:?} vs {v:?}");
    }

    #[test]
    fn rotation_preserves_length() {
        let v = DVec3::new(1.0, 2.0, 3.0).normalize();
        let axis = DVec3::new(-1.0, 0.4, 0.2).normalize();
        let out = rotate_about(v, axis, 0.7);
        assert!((out.length() - 1.0).abs() < 1e-12);
    }
}
