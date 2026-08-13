//! CPU-side chunk culling: frustum planes plus a horizon (sphere-occlusion)
//! test, both driven by a chunk's bounding cone.
//!
//! A chunk is a spherical cap of directions (`axis`, `cos_radius`) together
//! with a radial extent `[r_min, r_max]` in km once elevation displacement is
//! applied. Both tests are conservative: they may keep a chunk that is not
//! actually visible, never drop one that is.

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

/// A bounding sphere in world space, km.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

/// The six planes of a view frustum, each `(nx, ny, nz, d)` with the interior
/// on the positive side of `dot(n, p) + d`.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract the planes from a view-projection matrix (Gribb/Hartmann).
    /// Works for both the reverse-Z perspective and the orthographic matrices
    /// used here, because the near/far rows are derived from the matrix itself.
    pub fn from_view_proj(m: Mat4) -> Frustum {
        // Rows of the matrix (glam stores columns).
        let r0 = Vec4::new(m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x);
        let r1 = Vec4::new(m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y);
        let r2 = Vec4::new(m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z);
        let r3 = Vec4::new(m.x_axis.w, m.y_axis.w, m.z_axis.w, m.w_axis.w);
        let mut planes = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r2,      // near/far pair for a [0,1] depth range
            r3 - r2,
        ];
        for p in &mut planes {
            let len = p.xyz().length();
            if len > 0.0 {
                *p /= len;
            }
        }
        Frustum { planes }
    }

    /// True when the sphere is not entirely outside any single plane.
    pub fn intersects_sphere(&self, s: Sphere) -> bool {
        self.planes
            .iter()
            .all(|p| p.xyz().dot(s.center) + p.w >= -s.radius)
    }
}

/// Bounding sphere of the shell cap `{ r*d : r in [r_min, r_max],
/// angle(d, axis) <= acos(cos_radius) }`.
pub fn cap_bounding_sphere(axis: Vec3, cos_radius: f32, r_min: f32, r_max: f32) -> Sphere {
    let axis = axis.normalize();
    let c = cos_radius.clamp(-1.0, 1.0);
    let s = (1.0 - c * c).max(0.0).sqrt();
    // Anchor the sphere on the axis at the outer rim's projection, then take
    // the furthest of the four extreme points of the shell cap.
    let center = axis * (r_max * c);
    let d_outer_pole = (r_max - r_max * c).abs();
    let d_inner_pole = (r_max * c - r_min).abs();
    let d_outer_rim = r_max * s;
    let d_inner_rim = ((r_max * c - r_min * c).powi(2) + (r_min * s).powi(2)).sqrt();
    let radius = d_outer_pole
        .max(d_inner_pole)
        .max(d_outer_rim)
        .max(d_inner_rim);
    Sphere { center, radius }
}

/// True when every point of the shell cap is hidden behind the horizon of the
/// occluding sphere of radius `r_occ` seen from `camera` (both in km).
///
/// Uses the exact point-vs-sphere shadow-cone test on the cap's most visible
/// point (the one closest in angle to the camera, at the cap's outer radius),
/// which is conservative for the whole cap.
pub fn cap_below_horizon(
    camera: Vec3,
    axis: Vec3,
    cos_radius: f32,
    r_max: f32,
    r_occ: f32,
) -> bool {
    let d2 = camera.length_squared();
    if d2 <= r_occ * r_occ {
        // Inside the occluder: nothing is hidden by it.
        return false;
    }
    let n = camera / d2.sqrt();
    let axis = axis.normalize();
    let cos_ang = axis.dot(n).clamp(-1.0, 1.0);
    let cos_r = cos_radius.clamp(-1.0, 1.0);
    // Most visible direction in the cap: rotate `axis` toward `n` by the cap
    // half-angle, or `n` itself when the camera axis is inside the cap.
    let best_dir = if cos_ang >= cos_r {
        n
    } else {
        let sin_r = (1.0 - cos_r * cos_r).max(0.0).sqrt();
        let perp = (n - axis * cos_ang).normalize_or_zero();
        (axis * cos_r + perp * sin_r).normalize()
    };
    let p = best_dir * r_max;

    // Exact occlusion of point `p` by a sphere of radius r_occ at the origin,
    // seen from `camera`.
    let vc = -camera;
    let vt = p - camera;
    let vc_mag_sq = d2;
    let dot = vc.dot(vt);
    let limit = vc_mag_sq - r_occ * r_occ;
    dot > limit && dot * dot > limit * vt.length_squared()
}

/// Decide visibility for one chunk.
#[allow(clippy::too_many_arguments)]
pub fn chunk_visible(
    frustum: &Frustum,
    camera: Vec3,
    axis: Vec3,
    cos_radius: f32,
    r_min: f32,
    r_max: f32,
    r_occ: f32,
    horizon_cull: bool,
) -> bool {
    if horizon_cull && cap_below_horizon(camera, axis, cos_radius, r_max, r_occ) {
        return false;
    }
    frustum.intersects_sphere(cap_bounding_sphere(axis, cos_radius, r_min, r_max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    const R: f32 = 6371.0;

    fn look_down_frustum(eye: Vec3, aspect: f32) -> (Frustum, Mat4) {
        let mut proj = Mat4::perspective_infinite_reverse_rh(0.9, aspect, 1.0);
        proj.y_axis.y *= -1.0;
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Z);
        let vp = proj * view;
        (Frustum::from_view_proj(vp), vp)
    }

    fn cos_deg(d: f32) -> f32 {
        d.to_radians().cos()
    }

    #[test]
    fn bounding_sphere_contains_the_cap() {
        let axis = Vec3::new(0.3, -0.5, 0.81).normalize();
        for half_deg in [1.0f32, 10.0, 30.0, 60.0, 89.0] {
            let cr = cos_deg(half_deg);
            let (r_min, r_max) = (R - 12.0, R + 40.0);
            let s = cap_bounding_sphere(axis, cr, r_min, r_max);
            // Sample the cap and check containment.
            let (e, n) = {
                let mut e = Vec3::Z.cross(axis);
                if e.length_squared() < 1e-9 {
                    e = Vec3::X;
                }
                let e = e.normalize();
                (e, axis.cross(e).normalize())
            };
            let half = half_deg.to_radians();
            for i in 0..24 {
                let phi = i as f32 / 24.0 * std::f32::consts::TAU;
                for t in [0.0, 0.5, 1.0] {
                    let a = half * t;
                    let d = axis * a.cos() + (e * phi.cos() + n * phi.sin()) * a.sin();
                    for r in [r_min, (r_min + r_max) * 0.5, r_max] {
                        let p = d.normalize() * r;
                        assert!(
                            p.distance(s.center) <= s.radius + 1e-2,
                            "half={half_deg} r={r} out by {}",
                            p.distance(s.center) - s.radius
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bounding_sphere_of_a_point_cap_is_tiny() {
        let s = cap_bounding_sphere(Vec3::X, 1.0, R, R);
        assert!((s.center - Vec3::X * R).length() < 1e-3);
        assert!(s.radius < 1e-3);
    }

    #[test]
    fn frustum_keeps_the_chunk_under_the_camera() {
        let eye = Vec3::X * (R + 3000.0);
        let (f, _) = look_down_frustum(eye, 1.6);
        assert!(f.intersects_sphere(cap_bounding_sphere(Vec3::X, cos_deg(8.0), R, R + 10.0)));
    }

    #[test]
    fn frustum_rejects_a_chunk_far_off_axis() {
        // Close in, so the frustum is a narrow cone over the surface.
        let eye = Vec3::X * (R + 60.0);
        let (f, _) = look_down_frustum(eye, 1.6);
        let far_axis = Vec3::new(0.0, 1.0, 0.0);
        assert!(!f.intersects_sphere(cap_bounding_sphere(far_axis, cos_deg(3.0), R, R + 10.0)));
    }

    #[test]
    fn frustum_rejects_something_behind_the_camera() {
        let eye = Vec3::X * (R + 1000.0);
        let (f, _) = look_down_frustum(eye, 1.6);
        let behind = Sphere {
            center: eye + Vec3::X * 500.0,
            radius: 10.0,
        };
        assert!(!f.intersects_sphere(behind));
    }

    #[test]
    fn frustum_planes_agree_with_the_projection() {
        // A point projected inside NDC must be inside the frustum.
        let eye = Vec3::X * (R + 2000.0);
        let (f, vp) = look_down_frustum(eye, 1.0);
        for axis in [
            Vec3::X,
            Vec3::new(1.0, 0.1, 0.0).normalize(),
            Vec3::new(1.0, 0.0, 0.2).normalize(),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::NEG_X,
        ] {
            let p = axis * R;
            let clip = vp * p.extend(1.0);
            let inside_ndc = clip.w > 0.0
                && (clip.x / clip.w).abs() <= 1.0
                && (clip.y / clip.w).abs() <= 1.0
                && (0.0..=1.0).contains(&(clip.z / clip.w));
            if inside_ndc {
                assert!(
                    f.intersects_sphere(Sphere {
                        center: p,
                        radius: 0.0
                    }),
                    "visible point {p} rejected"
                );
            }
        }
    }

    #[test]
    fn horizon_hides_the_far_side() {
        let cam = Vec3::X * (R + 500.0);
        assert!(cap_below_horizon(cam, Vec3::NEG_X, cos_deg(5.0), R, R));
        assert!(cap_below_horizon(cam, Vec3::Y, cos_deg(5.0), R, R));
    }

    #[test]
    fn horizon_keeps_the_near_side() {
        let cam = Vec3::X * (R + 500.0);
        assert!(!cap_below_horizon(cam, Vec3::X, cos_deg(5.0), R, R));
        let just_inside = Vec3::new(cos_deg(10.0), sin_deg(10.0), 0.0);
        assert!(!cap_below_horizon(cam, just_inside, cos_deg(2.0), R, R));
    }

    fn sin_deg(d: f32) -> f32 {
        d.to_radians().sin()
    }

    #[test]
    fn horizon_keeps_a_cap_that_only_partly_pokes_over() {
        // Cap centre is beyond the horizon, but its near rim is not.
        let cam = Vec3::X * (R + 500.0);
        let horizon_deg = (R / (R + 500.0)).acos().to_degrees();
        let center_deg = horizon_deg + 6.0;
        let axis = Vec3::new(cos_deg(center_deg), sin_deg(center_deg), 0.0);
        assert!(cap_below_horizon(cam, axis, cos_deg(1.0), R, R));
        assert!(!cap_below_horizon(cam, axis, cos_deg(12.0), R, R));
    }

    #[test]
    fn tall_terrain_beyond_the_horizon_plane_is_still_visible() {
        // A peak just past the geometric horizon but high enough to be seen
        // must not be culled: the plane-only test would wrongly drop it.
        let cam = Vec3::X * (R + 500.0);
        let horizon_deg = (R / (R + 500.0)).acos().to_degrees();
        let axis = Vec3::new(cos_deg(horizon_deg + 0.5), sin_deg(horizon_deg + 0.5), 0.0);
        assert!(cap_below_horizon(cam, axis, 1.0, R, R), "at ground level");
        assert!(
            !cap_below_horizon(cam, axis, 1.0, R + 60.0, R),
            "60 km of relief must poke over"
        );
    }

    #[test]
    fn horizon_never_culls_when_the_camera_is_inside_the_planet() {
        let cam = Vec3::X * (R * 0.5);
        assert!(!cap_below_horizon(cam, Vec3::NEG_X, cos_deg(5.0), R, R));
    }

    #[test]
    fn chunk_visible_combines_both_tests() {
        let eye = Vec3::X * (R + 400.0);
        let (f, _) = look_down_frustum(eye, 1.6);
        assert!(chunk_visible(
            &f,
            eye,
            Vec3::X,
            cos_deg(4.0),
            R,
            R + 10.0,
            R,
            true
        ));
        assert!(!chunk_visible(
            &f,
            eye,
            Vec3::NEG_X,
            cos_deg(4.0),
            R,
            R + 10.0,
            R,
            true
        ));
    }
}
