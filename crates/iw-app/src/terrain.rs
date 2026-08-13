//! Procedural test terrain for renderer bring-up: a few spherical-harmonic
//! terms plus continent blobs, coloured by a hypsometric ramp.
//!
//! This is placeholder data only — WP10 replaces it with real `PlanetView`
//! snapshots from the simulation.

use glam::Vec3;

/// Deepest and highest elevation the generator produces, metres.
pub const MIN_ELEVATION_M: f32 = -6000.0;
/// See [`MIN_ELEVATION_M`].
pub const MAX_ELEVATION_M: f32 = 6000.0;

/// Fraction of cells placed above sea level.
const LAND_FRACTION: f32 = 0.32;

/// A cheap deterministic 64-bit mix (splitmix64 finaliser).
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn unit_f32(seed: u64, i: u32) -> f32 {
    (mix64(seed ^ ((i as u64) << 32)) >> 40) as f32 / (1u32 << 24) as f32
}

/// A deterministic unit vector from a seed and index.
pub fn hash_dir(seed: u64, i: u32) -> Vec3 {
    let z = unit_f32(seed, i * 3) * 2.0 - 1.0;
    let phi = unit_f32(seed, i * 3 + 1) * std::f32::consts::TAU;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Unnormalised terrain field: low-order harmonics for the global shape,
/// continent blobs for landmasses, and a little high-frequency relief.
pub fn terrain_field(dir: Vec3, seed: u64) -> f32 {
    let d = dir.normalize();
    let (x, y, z) = (d.x, d.y, d.z);

    // Low-order real spherical harmonics.
    let mut h = 0.0;
    h += 0.30 * (3.0 * z * z - 1.0) * 0.5; // Y20
    h += 0.45 * (x * x - y * y); // Y22
    h += 0.35 * (5.0 * x * y * z); // Y3-2
    h += 0.25 * (5.0 * z * z * z - 3.0 * z) * 0.5; // Y30

    // Continent blobs: broad positive lobes at deterministic directions.
    for i in 0..9u32 {
        let c = hash_dir(seed, i + 1);
        let radius = 0.30 + 0.35 * unit_f32(seed, 1000 + i);
        let weight = 0.55 + 0.75 * unit_f32(seed, 2000 + i);
        h += weight * smoothstep(1.0 - radius, 1.0 - radius * 0.15, d.dot(c));
    }

    // Mountain-belt scale detail, still smooth enough for a level-6 mesh.
    for i in 0..5u32 {
        let a = hash_dir(seed, 5000 + i);
        h += 0.14 * (d.dot(a) * 9.0 + unit_f32(seed, 6000 + i) * std::f32::consts::TAU).sin();
    }
    h
}

/// Generate elevations for every cell centre, in metres, calibrated so that
/// roughly [`LAND_FRACTION`] of cells sit above zero and the range fills
/// [`MIN_ELEVATION_M`]..[`MAX_ELEVATION_M`].
pub fn generate_elevation(centers: &[Vec3], seed: u64) -> Vec<f32> {
    let mut raw: Vec<f32> = centers.iter().map(|c| terrain_field(*c, seed)).collect();
    if raw.is_empty() {
        return raw;
    }
    let mut sorted = raw.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((1.0 - LAND_FRACTION) * sorted.len() as f32) as usize).min(sorted.len() - 1);
    let sea = sorted[idx];
    let lo = sorted[0] - sea;
    let hi = sorted[sorted.len() - 1] - sea;
    for v in &mut raw {
        let t = *v - sea;
        // `t/hi` and `t/lo` are both fractions in [0,1]; the sign comes from
        // which side of sea level the sample is on.
        *v = if t >= 0.0 {
            if hi > 0.0 {
                t / hi * MAX_ELEVATION_M
            } else {
                0.0
            }
        } else if lo < 0.0 {
            (t / lo) * MIN_ELEVATION_M
        } else {
            0.0
        };
        *v = v.clamp(MIN_ELEVATION_M, MAX_ELEVATION_M);
    }
    raw
}

/// Ocean colour stops: (elevation_m, rgb). Deep blue up to shelf turquoise.
const OCEAN_STOPS: [(f32, [u8; 3]); 4] = [
    (-6000.0, [4, 14, 58]),
    (-3000.0, [10, 42, 110]),
    (-400.0, [26, 96, 168]),
    (0.0, [64, 158, 198]),
];

/// Land colour stops: (elevation_m, rgb). Green through brown to snow.
const LAND_STOPS: [(f32, [u8; 3]); 6] = [
    (0.0, [62, 116, 62]),
    (500.0, [96, 142, 72]),
    (1500.0, [134, 132, 82]),
    (3000.0, [156, 122, 92]),
    (4500.0, [198, 186, 178]),
    (6000.0, [252, 252, 255]),
];

fn ramp(stops: &[(f32, [u8; 3])], e: f32) -> [u8; 3] {
    if e <= stops[0].0 {
        return stops[0].1;
    }
    let last = stops[stops.len() - 1];
    if e >= last.0 {
        return last.1;
    }
    for w in stops.windows(2) {
        let (e0, c0) = w[0];
        let (e1, c1) = w[1];
        if e <= e1 {
            let t = ((e - e0) / (e1 - e0)).clamp(0.0, 1.0);
            return [
                (c0[0] as f32 + (c1[0] as f32 - c0[0] as f32) * t).round() as u8,
                (c0[1] as f32 + (c1[1] as f32 - c0[1] as f32) * t).round() as u8,
                (c0[2] as f32 + (c1[2] as f32 - c0[2] as f32) * t).round() as u8,
            ];
        }
    }
    last.1
}

/// Hypsometric tint for an elevation in metres: deep blue -> light blue ->
/// green -> brown -> white. Alpha is always opaque.
pub fn hypsometric(elevation_m: f32) -> [u8; 4] {
    let rgb = if elevation_m < 0.0 {
        ramp(&OCEAN_STOPS, elevation_m)
    } else {
        ramp(&LAND_STOPS, elevation_m)
    };
    [rgb[0], rgb[1], rgb[2], 255]
}

/// Colour every cell from its elevation.
pub fn generate_colors(elevation_m: &[f32]) -> Vec<[u8; 4]> {
    elevation_m.iter().copied().map(hypsometric).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn land_ramp_red_channel_is_monotonic() {
        let mut prev = 0u8;
        let mut e = 0.0f32;
        while e <= 6000.0 {
            let c = hypsometric(e);
            assert!(c[0] >= prev, "red dipped at {e}: {} < {prev}", c[0]);
            prev = c[0];
            e += 25.0;
        }
    }

    #[test]
    fn land_ramp_ends_white_and_starts_green() {
        let low = hypsometric(0.0);
        assert!(
            low[1] > low[0] && low[1] > low[2],
            "sea-level land is green"
        );
        let high = hypsometric(6000.0);
        assert!(
            high[0] > 240 && high[1] > 240 && high[2] > 240,
            "peaks are white"
        );
    }

    #[test]
    fn ocean_ramp_blue_channel_is_monotonic() {
        let mut prev = 0u8;
        let mut e = -6000.0f32;
        while e < 0.0 {
            let c = hypsometric(e);
            assert!(c[2] >= prev, "blue dipped at {e}");
            assert!(c[2] > c[0], "ocean must be blue-dominant at {e}");
            prev = c[2];
            e += 25.0;
        }
    }

    #[test]
    fn ocean_gets_brighter_toward_the_shelf() {
        let deep = hypsometric(-6000.0);
        let shelf = hypsometric(-100.0);
        let sum = |c: [u8; 4]| c[0] as u32 + c[1] as u32 + c[2] as u32;
        assert!(sum(shelf) > sum(deep));
    }

    #[test]
    fn ramp_clamps_outside_the_range() {
        assert_eq!(hypsometric(-999_999.0), hypsometric(-6000.0));
        assert_eq!(hypsometric(999_999.0), hypsometric(6000.0));
        assert_eq!(hypsometric(0.0)[3], 255);
    }

    /// Each side of the ramp is continuous. The only deliberate discontinuity
    /// is at sea level, which is what draws the coastline.
    #[test]
    fn each_ramp_side_is_continuous() {
        for (lo, hi) in [(-6000.0f32, -1.0f32), (0.0, 6000.0)] {
            let mut prev = hypsometric(lo);
            let mut e = lo;
            while e <= hi {
                let c = hypsometric(e);
                for k in 0..3 {
                    let d = (c[k] as i32 - prev[k] as i32).abs();
                    assert!(d <= 12, "jump of {d} in channel {k} at {e}");
                }
                prev = c;
                e += 50.0;
            }
        }
        let sea = hypsometric(-1.0);
        let shore = hypsometric(0.0);
        assert!(
            sea[2] as i32 - shore[2] as i32 > 60,
            "coastline must be sharp"
        );
    }

    #[test]
    fn hash_dirs_are_unit_and_deterministic() {
        for i in 0..32 {
            let d = hash_dir(7, i);
            assert!((d.length() - 1.0).abs() < 1e-4);
            assert_eq!(d, hash_dir(7, i));
        }
        assert_ne!(hash_dir(7, 0), hash_dir(8, 0));
    }

    #[test]
    fn generated_elevation_is_in_range_with_land_and_sea() {
        // A cheap sample of the sphere; no mesh needed.
        let centers: Vec<Vec3> = (0..4000).map(|i| hash_dir(99, i)).collect();
        let e = generate_elevation(&centers, 42);
        assert_eq!(e.len(), centers.len());
        assert!(e
            .iter()
            .all(|v| (MIN_ELEVATION_M..=MAX_ELEVATION_M).contains(v)));
        let land = e.iter().filter(|v| **v > 0.0).count() as f32 / e.len() as f32;
        assert!((0.15..0.55).contains(&land), "land fraction {land}");
        assert!(e.iter().cloned().fold(f32::MIN, f32::max) > 3000.0);
        assert!(e.iter().cloned().fold(f32::MAX, f32::min) < -3000.0);
    }

    #[test]
    fn generation_is_deterministic() {
        let centers: Vec<Vec3> = (0..500).map(|i| hash_dir(1, i)).collect();
        assert_eq!(
            generate_elevation(&centers, 42),
            generate_elevation(&centers, 42)
        );
        assert_ne!(
            generate_elevation(&centers, 42),
            generate_elevation(&centers, 43)
        );
    }
}
