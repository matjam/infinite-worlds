//! Density-driven generator sampling for the Voronoi tessellation.
//!
//! Cell size is dictated by the terrain (docs/voronoi-v2.md): the density
//! field says where cells crowd (mountains, coasts, plate boundaries) and
//! where they sprawl (plains, abyss). Sampling is rejection from the density,
//! then a couple of DENSITY-WEIGHTED relaxation rounds — each generator moves
//! toward the density-weighted centroid of its Voronoi cell, which spaces
//! neighbors into froth without equalizing sizes (vanilla Lloyd would, and is
//! deliberately not used).

use glam::DVec3;

use crate::hull::convex_hull;

/// Deterministic splitmix64 stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn unit_dir(&mut self) -> DVec3 {
        let z = self.unit_f64() * 2.0 - 1.0;
        let phi = self.unit_f64() * std::f64::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        DVec3::new(r * phi.cos(), r * phi.sin(), z)
    }
}

/// Draw `n` generators from `density` (values in (0, 1]; 1 = densest) and
/// relax them `rounds` times against the density-weighted cell centroids.
/// Deterministic in `(n, seed, density)`.
pub fn sample_generators(
    n: usize,
    seed: u64,
    density: &(dyn Fn(glam::Vec3) -> f32 + Sync),
    rounds: u32,
) -> Vec<DVec3> {
    assert!(n >= 4, "tessellation needs at least 4 generators");
    let mut rng = Rng(seed ^ 0x564f_524f_4e4f_4931); // "VORONOI1"
    let mut pts: Vec<DVec3> = Vec::with_capacity(n);
    let mut guard = 0u64;
    while pts.len() < n {
        let d = rng.unit_dir();
        let p = density(d.as_vec3()).clamp(0.0, 1.0) as f64;
        if rng.unit_f64() < p {
            pts.push(d);
        }
        guard += 1;
        assert!(
            guard < (n as u64) * 10_000 + 1_000_000,
            "density field appears to be ~0 everywhere"
        );
    }

    for _ in 0..rounds {
        pts = weighted_relax(&pts, density);
    }
    pts
}

/// One density-weighted relaxation round: every generator moves to the
/// density-weighted mean of its cell's geometry (its own position plus its
/// cell corners = incident triangle circumcenters). High-density generators
/// barely move (their cells are tight); low-density cells even out into big
/// calm polygons. Size contrast survives because the weights do.
fn weighted_relax(pts: &[DVec3], density: &(dyn Fn(glam::Vec3) -> f32 + Sync)) -> Vec<DVec3> {
    use rayon::prelude::*;
    let faces = convex_hull(pts);
    // Accumulate circumcenters per generator.
    let mut sums: Vec<DVec3> = vec![DVec3::ZERO; pts.len()];
    let mut weights: Vec<f64> = vec![0.0; pts.len()];
    for f in &faces {
        let (a, b, c) = (pts[f.a as usize], pts[f.b as usize], pts[f.c as usize]);
        let cc = (b - a).cross(c - a);
        let cc = if cc.length_squared() > 1e-30 {
            cc.normalize()
        } else {
            (a + b + c).normalize()
        };
        let w = density(cc.as_vec3()).clamp(1e-4, 1.0) as f64;
        for v in [f.a, f.b, f.c] {
            sums[v as usize] += cc * w;
            weights[v as usize] += w;
        }
    }
    pts.par_iter()
        .enumerate()
        .map(|(i, p)| {
            let wp = density(p.as_vec3()).clamp(1e-4, 1.0) as f64 * 2.0;
            let m = sums[i] + *p * wp;
            let t = weights[i] + wp;
            if t > 0.0 && m.length_squared() > 1e-20 {
                m.normalize()
            } else {
                *p
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn sampling_is_deterministic_and_density_biased() {
        let density = |v: Vec3| if v.z > 0.0 { 1.0 } else { 0.1 };
        let a = sample_generators(4_000, 42, &density, 2);
        let b = sample_generators(4_000, 42, &density, 2);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x, y);
        }
        let north = a.iter().filter(|p| p.z > 0.0).count();
        let south = a.len() - north;
        // 10:1 density ratio: the north must dominate strongly (relaxation
        // bleeds a little across the boundary).
        assert!(
            north as f64 > south as f64 * 4.0,
            "density ignored: {north} north vs {south} south"
        );
    }

    #[test]
    fn relaxation_preserves_size_contrast() {
        // With uniform density, relaxation must NOT be the thing that fails —
        // and with contrast, the contrast must survive relaxation.
        let density = |v: Vec3| if v.z > 0.5 { 1.0 } else { 0.05 };
        let pts = sample_generators(3_000, 7, &density, 2);
        let cap = pts.iter().filter(|p| p.z > 0.5).count();
        // Cap above z=0.5 is 25% of the sphere; at 20x density it should hold
        // far more than 25% of the generators after relaxation.
        assert!(
            cap as f64 / pts.len() as f64 > 0.55,
            "size contrast lost: {cap} of {} in the dense cap",
            pts.len()
        );
    }
}
