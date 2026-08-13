//! Seeded, deterministic 3D gradient noise + fBm for spherical fields.
//!
//! Used to give the primordial crust fractal structure: craton outlines,
//! initial thickness texture, and plate-boundary warping all sample this
//! instead of analytic shapes. Classic Perlin-style gradient noise with a
//! seeded permutation (not table-free simplex, but visually equivalent for
//! fBm on a sphere); each fBm octave is rotated by a fixed orthogonal basis
//! change so lattice axes never align across octaves.
//!
//! Determinism: pure integer hashing from the seed — identical output on
//! every platform, thread count, and call order.

use glam::Vec3;

/// Seeded 3D gradient noise. Cheap to construct; copyable.
#[derive(Debug, Clone, Copy)]
pub struct Noise3 {
    seed: u64,
}

impl Noise3 {
    pub fn new(seed: u64) -> Self {
        Noise3 { seed }
    }

    /// Single-octave gradient noise at `p`, roughly in [-1, 1].
    pub fn sample(&self, p: Vec3) -> f32 {
        let xf = p.x.floor();
        let yf = p.y.floor();
        let zf = p.z.floor();
        let (x0, y0, z0) = (xf as i64, yf as i64, zf as i64);
        let f = Vec3::new(p.x - xf, p.y - yf, p.z - zf);
        let u = fade(f.x);
        let v = fade(f.y);
        let w = fade(f.z);
        let mut corner = [0.0f32; 8];
        for (i, c) in corner.iter_mut().enumerate() {
            let (dx, dy, dz) = ((i & 1) as i64, ((i >> 1) & 1) as i64, ((i >> 2) & 1) as i64);
            let g = self.gradient(x0 + dx, y0 + dy, z0 + dz);
            let d = f - Vec3::new(dx as f32, dy as f32, dz as f32);
            *c = g.dot(d);
        }
        let x00 = lerp(corner[0], corner[1], u);
        let x10 = lerp(corner[2], corner[3], u);
        let x01 = lerp(corner[4], corner[5], u);
        let x11 = lerp(corner[6], corner[7], u);
        let y0v = lerp(x00, x10, v);
        let y1v = lerp(x01, x11, v);
        // Perlin dot-gradient range is ~±0.87 for unit gradients; renormalize.
        lerp(y0v, y1v, w) * 1.15
    }

    /// Fractal Brownian motion: `octaves` layers, each `lacunarity`× the
    /// frequency and `gain`× the amplitude of the last, axes de-correlated
    /// per octave. Output roughly in [-1, 1].
    pub fn fbm(&self, p: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut sum = 0.0;
        let mut amp = 1.0;
        let mut norm = 0.0;
        let mut q = p;
        for oct in 0..octaves {
            let n = Noise3 {
                seed: self.seed.wrapping_add((oct as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            };
            sum += amp * n.sample(q);
            norm += amp;
            amp *= gain;
            // Fixed rotation + scale between octaves hides lattice alignment.
            q = Vec3::new(
                q.y * 0.80 + q.z * 0.60,
                q.z * 0.80 - q.x * 0.60,
                q.x * 0.80 + q.y * 0.60,
            ) * lacunarity;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }

    /// Ridged fBm (1 − |fbm|-style per octave), in [0, 1]: sharp crease lines,
    /// useful for proto-rift/weakness fields.
    pub fn ridged(&self, p: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut sum = 0.0;
        let mut amp = 1.0;
        let mut norm = 0.0;
        let mut q = p;
        for oct in 0..octaves {
            let n = Noise3 {
                seed: self
                    .seed
                    .wrapping_add(0x517c_c1b7_2722_0a95)
                    .wrapping_add((oct as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            };
            let r = 1.0 - n.sample(q).abs().min(1.0);
            sum += amp * r * r;
            norm += amp;
            amp *= gain;
            q = Vec3::new(
                q.y * 0.80 + q.z * 0.60,
                q.z * 0.80 - q.x * 0.60,
                q.x * 0.80 + q.y * 0.60,
            ) * lacunarity;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }

    fn gradient(&self, x: i64, y: i64, z: i64) -> Vec3 {
        let h = hash3(self.seed, x, y, z);
        // 12 edge-of-cube gradient directions, Perlin's classic set.
        const G: [[f32; 3]; 12] = [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
        ];
        let g = G[(h % 12) as usize];
        Vec3::from_array(g) * std::f32::consts::FRAC_1_SQRT_2
    }
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// splitmix64-style avalanche over (seed, lattice point).
#[inline]
fn hash3(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    let mut h = seed
        ^ (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ (z as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_sensitive() {
        let a = Noise3::new(42);
        let b = Noise3::new(42);
        let c = Noise3::new(43);
        let p = Vec3::new(1.7, -2.3, 0.9);
        assert_eq!(a.sample(p), b.sample(p));
        assert_ne!(a.sample(p), c.sample(p));
        assert_eq!(a.fbm(p, 6, 2.0, 0.5), b.fbm(p, 6, 2.0, 0.5));
    }

    #[test]
    fn bounded_and_varied() {
        let n = Noise3::new(7);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for i in 0..4000 {
            let t = i as f32 * 0.137;
            let p = Vec3::new(t.sin() * 3.0, (t * 0.7).cos() * 3.0, t * 0.11);
            let v = n.fbm(p, 6, 2.0, 0.5);
            min = min.min(v);
            max = max.max(v);
            assert!(v.is_finite() && (-1.2..=1.2).contains(&v));
            let r = n.ridged(p, 5, 2.0, 0.5);
            assert!((0.0..=1.0001).contains(&r));
        }
        assert!(max - min > 0.5, "fbm range too flat: {min}..{max}");
    }

    #[test]
    fn continuous() {
        let n = Noise3::new(9);
        let p = Vec3::new(0.5, 0.5, 0.5);
        let eps = 1e-3;
        let d = (n.sample(p) - n.sample(p + Vec3::splat(eps))).abs();
        assert!(d < 0.05, "discontinuity: {d}");
    }
}
