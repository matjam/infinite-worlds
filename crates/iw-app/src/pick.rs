//! Cell picking: screen pixel -> ray -> sphere -> cell id.
//!
//! Pure math, no GPU and no window: the ray is rebuilt from the camera basis
//! and the vertical field of view rather than by inverting the projection
//! matrix, so it is independent of the reverse-Z / Y-flip conventions the
//! renderer uses and can be unit tested on its own.

use glam::{Vec2, Vec3};
use iw_render_vulkan::camera::{Camera, FOV_Y_RAD};
use iw_render_vulkan::mercator;
use iw_render_vulkan::ViewMode;

/// A world-space ray, origin in kilometres.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Start of the ray, km from the planet centre.
    pub origin: Vec3,
    /// Unit direction.
    pub dir: Vec3,
}

/// Normalised device coordinates of a pixel: x right, y **down**, both in
/// -1..1. `size` is in the same pixel units as `pos`.
pub fn ndc_from_pixel(pos: Vec2, size: (u32, u32)) -> Vec2 {
    let w = size.0.max(1) as f32;
    let h = size.1.max(1) as f32;
    Vec2::new(2.0 * pos.x / w - 1.0, 2.0 * pos.y / h - 1.0)
}

/// The camera ray through a point in NDC (perspective/globe mode).
pub fn ray_through(camera: &Camera, aspect: f32, ndc: Vec2) -> Ray {
    let (forward, up) = camera.orientation();
    let right = forward.cross(up).normalize();
    let tan_half = (FOV_Y_RAD * 0.5).tan();
    // Screen y points down, camera up points up, hence the negation.
    let dir = forward + right * (ndc.x * tan_half * aspect.max(1e-3)) + up * (-ndc.y * tan_half);
    Ray {
        origin: camera.eye(),
        dir: dir.normalize(),
    }
}

/// Distance along `ray` to the first intersection with a sphere of radius
/// `radius_km` centred on the origin, or `None` when the ray misses or the
/// sphere is entirely behind the ray's origin.
pub fn ray_sphere(ray: Ray, radius_km: f32) -> Option<f32> {
    let b = ray.origin.dot(ray.dir);
    let c = ray.origin.length_squared() - radius_km * radius_km;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let t_near = -b - root;
    if t_near > 0.0 {
        return Some(t_near);
    }
    let t_far = -b + root;
    if t_far > 0.0 {
        // Origin is inside the sphere: exit point.
        return Some(t_far);
    }
    None
}

/// Unit direction of the surface point under a pixel in Mercator mode, or
/// `None` when the pixel is off the projected plane (beyond ±85° latitude).
pub fn mercator_direction(camera: &Camera, aspect: f32, ndc: Vec2) -> Option<Vec3> {
    let hw = camera.mercator_half_width.max(1e-4);
    let hh = hw / aspect.max(1e-3);
    let x = camera.mercator_center.x + ndc.x * hw;
    // The orthographic projection is Y-flipped, so screen-down is -y.
    let y = camera.mercator_center.y - ndc.y * hh;
    let limit = mercator::mercator_y(mercator::LAT_LIMIT_RAD);
    if y.abs() > limit {
        return None;
    }
    let lat = mercator::inverse_mercator_y(y);
    Some(mercator::dir_from_latlon(lat, mercator::wrap_pi(x)))
}

/// Unit direction of the planet surface under a pixel, in either view mode.
/// `None` when the pixel shows space (or, in Mercator, off-map).
pub fn surface_direction(camera: &Camera, aspect: f32, ndc: Vec2, radius_km: f32) -> Option<Vec3> {
    match camera.mode {
        ViewMode::Globe => {
            let ray = ray_through(camera, aspect, ndc);
            let t = ray_sphere(ray, radius_km)?;
            Some((ray.origin + ray.dir * t).normalize())
        }
        ViewMode::Mercator => mercator_direction(camera, aspect, ndc),
    }
}

/// Project a world-space point (km) to pixel coordinates, or `None` when it is
/// behind the camera or outside the viewport. Used for the plate velocity
/// arrows, which are drawn by the egui painter in screen space.
pub fn project_to_pixels(view_proj: glam::Mat4, point_km: Vec3, size: (u32, u32)) -> Option<Vec2> {
    let clip = view_proj * point_km.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
    if !ndc.is_finite() || ndc.x.abs() > 1.2 || ndc.y.abs() > 1.2 {
        return None;
    }
    let w = size.0.max(1) as f32;
    let h = size.1.max(1) as f32;
    Some(Vec2::new((ndc.x + 1.0) * 0.5 * w, (ndc.y + 1.0) * 0.5 * h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iw_mesh::EARTH_RADIUS_KM;

    fn camera_over(dir: Vec3, altitude_km: f32) -> Camera {
        // `Camera` keeps private recenter state, so build it by mutation
        // rather than struct-update syntax.
        let mut cam = Camera::default();
        cam.focus = dir.normalize();
        cam.altitude_km = altitude_km;
        cam
    }

    #[test]
    fn centre_pixel_hits_the_point_under_the_camera() {
        let cam = camera_over(Vec3::new(0.3, -0.7, 0.5), 12_000.0);
        let hit = surface_direction(&cam, 16.0 / 9.0, Vec2::ZERO, EARTH_RADIUS_KM)
            .expect("the centre of the screen is the planet");
        assert!(
            hit.distance(cam.focus.normalize()) < 1e-4,
            "expected {} got {hit}",
            cam.focus.normalize()
        );
    }

    #[test]
    fn centre_pixel_hits_the_near_side_not_the_far_side() {
        let cam = camera_over(Vec3::X, 5_000.0);
        let ray = ray_through(&cam, 1.0, Vec2::ZERO);
        let t = ray_sphere(ray, EARTH_RADIUS_KM).unwrap();
        assert!(
            (t - 5_000.0).abs() < 1.0,
            "should hit at the altitude, got {t}"
        );
    }

    #[test]
    fn pixels_off_the_limb_miss() {
        // At 8000 km the planet subtends ~26 degrees from nadir while a corner
        // pixel of a 16:9 frame looks ~45 degrees off it: past the limb.
        let cam = camera_over(Vec3::X, 8_000.0);
        let aspect = 16.0 / 9.0;
        assert!(surface_direction(&cam, aspect, Vec2::ZERO, EARTH_RADIUS_KM).is_some());
        let corner = Vec2::new(1.0, 1.0);
        assert!(
            surface_direction(&cam, aspect, corner, EARTH_RADIUS_KM).is_none(),
            "a corner pixel at 8000 km altitude looks past the limb"
        );
        // ...and the miss is a clean geometric one, not a sign error.
        let ray = ray_through(&cam, aspect, corner);
        assert!(ray_sphere(ray, EARTH_RADIUS_KM).is_none());
    }

    #[test]
    fn a_sphere_behind_the_ray_is_not_hit() {
        // Pointing straight away from the planet.
        let ray = Ray {
            origin: Vec3::X * (EARTH_RADIUS_KM + 1_000.0),
            dir: Vec3::X,
        };
        assert_eq!(ray_sphere(ray, EARTH_RADIUS_KM), None);
        // Grazing tangent line offset outside the sphere: still a miss.
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, -50_000.0) + Vec3::X * (EARTH_RADIUS_KM * 1.5),
            dir: Vec3::Z,
        };
        assert_eq!(ray_sphere(ray, EARTH_RADIUS_KM), None);
    }

    #[test]
    fn rays_are_unit_length_and_move_the_right_way_on_screen() {
        let cam = camera_over(Vec3::X, 2_000.0);
        let aspect = 2.0;
        for ndc in [
            Vec2::ZERO,
            Vec2::new(0.5, 0.0),
            Vec2::new(0.0, 0.5),
            Vec2::new(-1.0, -1.0),
        ] {
            let r = ray_through(&cam, aspect, ndc);
            assert!((r.dir.length() - 1.0).abs() < 1e-5);
        }
        // At the equator with default framing, north is +z on screen-up and
        // east is +y on screen-right.
        let up_pixel =
            surface_direction(&cam, aspect, Vec2::new(0.0, -0.5), EARTH_RADIUS_KM).expect("hit");
        assert!(up_pixel.z > 0.0, "screen-up should be north: {up_pixel}");
        let right_pixel =
            surface_direction(&cam, aspect, Vec2::new(0.5, 0.0), EARTH_RADIUS_KM).expect("hit");
        assert!(right_pixel.y > 0.0, "screen-right should be east");
    }

    #[test]
    fn ndc_maps_the_pixel_grid() {
        let size = (1600, 900);
        assert_eq!(ndc_from_pixel(Vec2::new(800.0, 450.0), size), Vec2::ZERO);
        let tl = ndc_from_pixel(Vec2::ZERO, size);
        assert_eq!(tl, Vec2::new(-1.0, -1.0));
        let br = ndc_from_pixel(Vec2::new(1600.0, 900.0), size);
        assert_eq!(br, Vec2::new(1.0, 1.0));
    }

    #[test]
    fn free_look_moves_the_picked_point_off_the_focus() {
        let mut cam = camera_over(Vec3::X, 300.0);
        let straight_down = surface_direction(&cam, 1.0, Vec2::ZERO, EARTH_RADIUS_KM).expect("hit");
        cam.free_look(0.0, 0.6);
        let tilted = surface_direction(&cam, 1.0, Vec2::ZERO, EARTH_RADIUS_KM).expect("hit");
        assert!(
            tilted.distance(straight_down) > 1e-3,
            "pitching the camera must move the centre pixel's target"
        );
    }

    #[test]
    fn mercator_picking_inverts_the_projection() {
        let mut cam = Camera::default();
        cam.mode = ViewMode::Mercator;
        cam.mercator_center = Vec2::new(0.4, 0.2);
        cam.mercator_half_width = 1.0;
        let dir = surface_direction(&cam, 1.0, Vec2::ZERO, EARTH_RADIUS_KM).expect("on map");
        let ll = iw_mesh::latlon_of(dir);
        assert!((ll[1] - 0.4).abs() < 1e-4, "longitude {}", ll[1]);
        let expect_lat = mercator::inverse_mercator_y(0.2);
        assert!((ll[0] - expect_lat).abs() < 1e-4, "latitude {}", ll[0]);

        // Screen-up must be north here too.
        let up = surface_direction(&cam, 1.0, Vec2::new(0.0, -0.5), EARTH_RADIUS_KM).unwrap();
        assert!(iw_mesh::latlon_of(up)[0] > ll[0]);

        // Far off the top of a zoomed-out map is off the projection.
        let mut wide = cam.clone();
        wide.mercator_half_width = std::f32::consts::PI;
        wide.mercator_center = Vec2::new(0.0, 3.0);
        assert!(surface_direction(&wide, 1.0, Vec2::new(0.0, -1.0), EARTH_RADIUS_KM).is_none());
    }

    #[test]
    fn projection_to_pixels_rejects_points_behind_the_camera() {
        let cam = camera_over(Vec3::X, 10_000.0);
        let aspect = 16.0 / 9.0;
        let vp = cam.view_proj(aspect);
        let size = (1600u32, 900u32);
        let front = project_to_pixels(vp, Vec3::X * EARTH_RADIUS_KM, size).expect("visible");
        assert!((front.x - 800.0).abs() < 1.0 && (front.y - 450.0).abs() < 1.0);
        // Straight up from the camera is behind it.
        let behind = Vec3::X * (EARTH_RADIUS_KM + 2.0 * cam.altitude_km);
        assert!(project_to_pixels(vp, behind, size).is_none());
        // The antipode is in front of the camera (through the planet), so it
        // projects: occlusion is the caller's job, not this function's.
        assert!(project_to_pixels(vp, Vec3::NEG_X * EARTH_RADIUS_KM, size).is_some());
    }

    #[test]
    fn picking_agrees_with_the_mesh_lookup() {
        // A coarse mesh keeps the test cheap; the point is that the direction
        // we compute lands on the cell whose centre is nearest to it.
        let mesh = iw_mesh::Mesh::build(3);
        let cam = camera_over(mesh.centers[57], 8_000.0);
        let dir = surface_direction(&cam, 1.0, Vec2::ZERO, EARTH_RADIUS_KM).expect("hit");
        assert_eq!(mesh.cell_at(dir), 57);
    }
}
