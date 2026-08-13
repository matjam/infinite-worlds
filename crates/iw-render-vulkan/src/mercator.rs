//! Mercator projection math, mirrored exactly by `shaders/globe.vert`.
//!
//! Seam handling (the shader does the same thing): each vertex wraps its
//! longitude relative to its own cell's centre longitude, and the cell centre
//! wraps relative to the view centre longitude. A cell therefore never spans
//! the seam, and the seam falls between cells at the antimeridian of the view
//! centre. No triangles are discarded and no geometry is duplicated.

use glam::{Vec2, Vec3};

/// Latitude beyond which Mercator is clamped, radians (85 degrees).
pub const LAT_LIMIT_RAD: f32 = 1.483_529_9;

/// Mercator y for a latitude in radians, clamped to +/-85 degrees.
#[inline]
pub fn mercator_y(lat_rad: f32) -> f32 {
    let lat = lat_rad.clamp(-LAT_LIMIT_RAD, LAT_LIMIT_RAD);
    (std::f32::consts::FRAC_PI_4 + 0.5 * lat).tan().ln()
}

/// Inverse of [`mercator_y`]: latitude in radians for a Mercator y.
#[inline]
pub fn inverse_mercator_y(y: f32) -> f32 {
    2.0 * y.exp().atan() - std::f32::consts::FRAC_PI_2
}

/// Wrap an angle into [-pi, pi).
#[inline]
pub fn wrap_pi(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    a - TAU * ((a + PI) / TAU).floor()
}

/// Project a unit direction into Mercator space relative to a view centre
/// longitude, resolving the seam through the owning cell's centre longitude.
pub fn project(dir: Vec3, cell_center_lon_rad: f32, center_lon_rad: f32) -> Vec2 {
    let n = dir.normalize();
    let lon = n.y.atan2(n.x);
    let lat = n.z.clamp(-1.0, 1.0).asin();
    let local = wrap_pi(lon - cell_center_lon_rad);
    let x = wrap_pi(cell_center_lon_rad - center_lon_rad) + local;
    Vec2::new(x, mercator_y(lat))
}

/// Unit direction from a latitude/longitude pair in radians.
pub fn dir_from_latlon(lat_rad: f32, lon_rad: f32) -> Vec3 {
    let (sla, cla) = lat_rad.sin_cos();
    let (slo, clo) = lon_rad.sin_cos();
    Vec3::new(cla * clo, cla * slo, sla)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_within_the_clamp_band() {
        for i in -84..=84 {
            let lat = (i as f32).to_radians();
            let back = inverse_mercator_y(mercator_y(lat));
            assert!((back - lat).abs() < 1e-4, "lat {lat} -> {back}");
        }
    }

    #[test]
    fn equator_is_zero_and_sign_follows_latitude() {
        assert!(mercator_y(0.0).abs() < 1e-6);
        assert!(mercator_y(0.5) > 0.0);
        assert!(mercator_y(-0.5) < 0.0);
    }

    #[test]
    fn latitude_is_clamped_at_85_degrees() {
        let at_limit = mercator_y(LAT_LIMIT_RAD);
        assert_eq!(mercator_y(1.5), at_limit);
        assert_eq!(mercator_y(std::f32::consts::FRAC_PI_2), at_limit);
        assert!((mercator_y(-std::f32::consts::FRAC_PI_2) + at_limit).abs() < 1e-5);
        assert!(at_limit.is_finite());
        // The classic Mercator y at 85 degrees is ~3.13.
        assert!((at_limit - 3.131).abs() < 0.01, "{at_limit}");
    }

    #[test]
    fn mercator_y_is_strictly_increasing() {
        let mut prev = f32::NEG_INFINITY;
        for i in -85..=85 {
            let y = mercator_y((i as f32).to_radians());
            assert!(y > prev, "not increasing at {i}");
            prev = y;
        }
    }

    #[test]
    fn projection_matches_direction_round_trip() {
        for lat_deg in [-80.0f32, -30.0, 0.0, 12.5, 60.0] {
            for lon_deg in [-179.0f32, -90.0, 0.0, 45.0, 178.0] {
                let lat = lat_deg.to_radians();
                let lon = lon_deg.to_radians();
                let d = dir_from_latlon(lat, lon);
                let p = project(d, lon, 0.0);
                assert!((p.x - lon).abs() < 1e-4, "lon {lon} -> {}", p.x);
                assert!((inverse_mercator_y(p.y) - lat).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn a_cell_never_spans_the_seam() {
        // A cell centred just west of the antimeridian, with a corner just east
        // of it: both must land adjacent in projection space, not 2*pi apart.
        let center_lon = 179.5f32.to_radians();
        let corner_lon = (-179.5f32).to_radians();
        let c = project(dir_from_latlon(0.0, center_lon), center_lon, 0.0);
        let v = project(dir_from_latlon(0.0, corner_lon), center_lon, 0.0);
        assert!((v.x - c.x).abs() < 0.02, "cell split: {} vs {}", c.x, v.x);
    }

    #[test]
    fn seam_sits_opposite_the_view_centre() {
        // With the view centred at lon 90 deg, a cell at lon -91 deg (i.e. just
        // past the antimeridian of the centre) wraps to the far positive side.
        let center = 90f32.to_radians();
        let cell = (-91f32).to_radians();
        let p = project(dir_from_latlon(0.0, cell), cell, center);
        assert!(p.x > 3.0, "expected wrap to +pi side, got {}", p.x);
        // And a cell just short of it stays on the near side.
        let cell = (-89f32).to_radians();
        let p = project(dir_from_latlon(0.0, cell), cell, center);
        assert!(p.x < -3.0, "expected -pi side, got {}", p.x);
    }

    #[test]
    fn projected_x_stays_in_a_bounded_band() {
        for i in 0..360 {
            let lon = ((i - 180) as f32).to_radians();
            let p = project(dir_from_latlon(0.3, lon), lon, 0.4);
            assert!(p.x.abs() <= std::f32::consts::PI + 1e-4, "{}", p.x);
        }
    }
}
