//! Craton geometry: the noise-derived shape of a continental nucleus.
//!
//! A craton is a spherical cap whose outline is deformed by 3D fBm sampled on
//! the unit sphere, so its coastline has structure at every scale from
//! continent-wide embayments down to a couple of cells. Two noise stages act on
//! the outline: a low-frequency *domain warp* that bends the whole cap (this is
//! what produces peninsulas and overhanging bays, which a radius-only
//! modulation cannot), then a radius modulation with `detail_octaves` of fBm.
//!
//! # The craton-local frame (drift stability)
//!
//! The shape is defined once and for all in a *local frame* whose pole is
//! `+Z`; nothing about it depends on where the craton currently is. To place it
//! on the planet, the craton's current world centroid (recomputed each step
//! from the cells it owns — see [`crate::phase1`]) is rotated onto
//! [`CratonShape::centroid_local`], the local-frame direction of the same
//! centroid, by the minimal rotation between the two (no roll). Sampling the
//! noise through that rotation makes the outline *rigid*: as the craton drifts,
//! the shape translates with it instead of being re-drawn slightly differently
//! every step. That is what removes the concentric ripple striping the old
//! analytic lobes left behind — their azimuth reference was a fixed world axis,
//! so the lobe pattern swung around the craton as it moved and the boundary
//! swept back and forth over the same cells.
//!
//! Aligning the *centroid* rather than the cap's pole matters: the mask's area
//! centroid sits some way off its pole (the noise is not symmetric), and
//! `phase1` can only observe the centroid. Aligning pole-to-centroid would make
//! the placement inconsistent with the observation and the craton would crawl by
//! that offset every single step. Aligning centroid-to-centroid is a fixed
//! point.
//!
//! Roll is left undetermined by the minimal rotation, so a craton slowly rolls
//! relative to its own noise (holonomy of the transport, ~0.1 rad over a whole
//! phase). That is far below one cell per step and invisible in fBm.
//!
//! Everything here is a pure function of `(seed, craton index, detail octaves)`,
//! so a fresh process instance rebuilds bit-identical shapes from a checkpoint.

use glam::{Quat, Vec3};
use iw_core::noise::Noise3;
use iw_core::rng_for;
use iw_mesh::EARTH_RADIUS_M;
use rand::Rng;

use crate::{CRATON_CORE_THICKNESS_M, CRATON_EDGE_THICKNESS_M};

/// Radius modulation, as a multiple of the fBm value. Measured over the sphere
/// this fBm has RMS 0.124 and peaks at +-0.37, so 1.8 means the coastline
/// typically sits +-22% off the mean radius and reaches +-65% — the difference
/// between a disc and a landmass with capes and gulfs.
const OUTLINE_AMP: f32 = 1.8;
/// First-octave outline frequency on the unit sphere: ~2.5 features per
/// hemisphere, i.e. continent scale.
const OUTLINE_FREQ: f32 = 2.6;
/// Domain-warp amplitude, in radians per unit of fBm (RMS 0.124, see
/// [`OUTLINE_AMP`]): displaces the query direction by ~0.08 rad typically and
/// up to 0.2, i.e. a fifth to half a craton radius. This is what lets the
/// outline fold back on itself into peninsulas, drowned bays and offshore
/// fragments — a radius-only modulation stays star-shaped and cannot.
const WARP_AMP: f32 = 0.6;
/// Domain-warp frequency; deliberately below the outline frequency so the warp
/// bends whole coast sections rather than roughening them.
const WARP_FREQ: f32 = 2.4;
/// Octaves in the domain warp. The warp only needs its low frequencies; the
/// fine detail comes from the radius modulation.
const WARP_OCTAVES: u32 = 4;
/// Interior thickness texture, as a fraction of the profile value.
const THICKNESS_TEXTURE: f32 = 0.10;
/// Thickness-texture frequency: ~1,500 km features.
const THICKNESS_FREQ: f32 = 7.0;
const LACUNARITY: f32 = 2.0;
/// fBm gain. 0.62 rather than the usual 0.5 puts more power in the 300-800 km
/// octaves, which is the band that reads as coastline character (fjords,
/// isthmuses, capes) rather than as either continent outline or pixel noise.
const GAIN: f32 = 0.62;
/// Floor on the modulated radius, as a fraction of the mean, so a deep noise
/// trough cannot invert the cap.
const MIN_RADIUS_FRAC: f32 = 0.35;

/// Fraction of the planet the craton nuclei are sized to cover in total.
///
/// Earth's continental crust, shelves included, is a little under 40%, of which
/// ~29% of the globe stands above water. Calibration: 0.26 was too little. It
/// gave a 26% continental planet whose *entire* continental area was dry land
/// (nothing to flood into shelves) and, worse, left 74% of the planet as deep
/// basin, so the fixed water budget had to pool 3.5 km deep and dragged sea
/// level a kilometre below the geoid. 0.37 leaves room for arc growth to reach
/// Earth's ~40% and produces ~30% land with a real drowned margin around it —
/// at 0.34 the drowned margin ate the gain and land came out under 25%.
///
/// Re-checked after the move to noise outlines (which add ~5% area for the same
/// nominal radius, the modulated radius being squared in the cap area) and to
/// the every-step thickness profile: 0.37 still lands sea level within +-320 m
/// and land at 24-25% on seeds 42/7/1337.
pub(crate) const CONTINENTAL_TARGET_FRACTION: f64 = 0.37;
/// Per-craton area spread about the mean, as a multiplier.
const CRATON_AREA_SPREAD: (f64, f64) = (0.65, 1.35);

/// Quadrature samples used to locate a shape's area centroid in its own frame.
const CENTROID_SAMPLES: u32 = 24_000;
/// Golden angle, for the spiral quadrature.
const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// One craton's shape, in its own frame. Pure function of `(seed, index)`.
pub(crate) struct CratonShape {
    /// Mean cap radius, metres.
    pub(crate) radius_m: f64,
    /// Mean cap radius, radians.
    radius_rad: f32,
    /// Per-craton noise field: outline warp, outline radius, and interior
    /// thickness texture are all drawn from it at different offsets.
    noise: Noise3,
    /// Octaves in the outline radius modulation; chosen from the mesh pitch so
    /// the coastline stays fractal down to cell scale at any subdivision level.
    detail_octaves: u32,
    /// Local-frame direction of the shape's area centroid. See the module docs.
    centroid_local: Vec3,
}

impl CratonShape {
    /// Rotation carrying world directions into this craton's local frame, given
    /// the craton's current world centroid (unit).
    pub(crate) fn frame(&self, centroid: Vec3) -> Quat {
        Quat::from_rotation_arc(centroid, self.centroid_local)
    }

    /// World direction of the cap's pole under `frame`.
    pub(crate) fn pole_world(&self, frame: Quat) -> Vec3 {
        frame.inverse() * Vec3::Z
    }

    /// Angular radius the shape is truncated at. The absolute worst case
    /// (`OUTLINE_AMP` and `WARP_AMP` both at their peak fBm value, in the same
    /// place) is 2.8 radii plus 1.2 rad, which is most of the sphere and makes
    /// the cull useless; measured extents run to 1.75 radii, so the bound is set
    /// just past that. Both the rasterizer and the centroid quadrature use it,
    /// so the truncation is part of the shape's definition and the two agree.
    pub(crate) fn max_radius_rad(&self) -> f32 {
        (self.radius_rad * 1.85 + 0.10).min(std::f32::consts::PI)
    }

    /// Cosine of [`Self::max_radius_rad`]: cells outside this cone are skipped
    /// before any noise is evaluated.
    pub(crate) fn cos_bound(&self) -> f32 {
        self.max_radius_rad().cos()
    }

    /// `Some(f)` when the local-frame direction `q` is inside the craton, where
    /// `f` is the normalized radial coordinate (0 at the pole, 1 at the rim).
    pub(crate) fn contains(&self, q: Vec3) -> Option<f32> {
        if q.z < self.cos_bound() {
            return None;
        }
        let w = self.warp(q);
        let r = self.radius_at(w);
        let d = w.z.clamp(-1.0, 1.0).acos();
        if d <= r {
            Some((d / r).min(1.0))
        } else {
            None
        }
    }

    /// Crustal thickness at local direction `q` and radial coordinate `f`:
    /// the core-to-edge profile, textured so interiors are not billiard-smooth.
    ///
    /// The taper is `sqrt(f)`: thickness falls away quickly from the shield
    /// core and then flattens, so most of a craton is a low platform with a
    /// broad drowned margin rather than a dome. Calibration: the profile is now
    /// re-applied every step (see `phase1::PROFILE_RATE_PER_MYR`) instead of
    /// being frozen at each cell's claim thickness, which raised mean
    /// continental thickness by ~3 km; with a linear taper that put permanent
    /// ice at 9.7% of the planet on seed 1337, over the 9% acceptance ceiling.
    /// `sqrt` drops the area-weighted mean by ~1.2 km and ice back to 7-8%.
    pub(crate) fn thickness_m(&self, q: Vec3, f: f32) -> f32 {
        let base = CRATON_CORE_THICKNESS_M
            - (CRATON_CORE_THICKNESS_M - CRATON_EDGE_THICKNESS_M) * f.sqrt();
        let t = self.noise.fbm(
            q * THICKNESS_FREQ + Vec3::new(31.2, -17.6, 9.4),
            self.detail_octaves.min(6),
            LACUNARITY,
            GAIN,
        );
        base * (1.0 + THICKNESS_TEXTURE * t)
    }

    /// Domain warp: bend the query direction before the radius test, so the
    /// outline can fold back on itself (bays, peninsulas, offshore fragments).
    fn warp(&self, q: Vec3) -> Vec3 {
        let p = q * WARP_FREQ;
        let v = Vec3::new(
            self.noise.fbm(
                p + Vec3::new(11.3, 4.7, -2.1),
                WARP_OCTAVES,
                LACUNARITY,
                GAIN,
            ),
            self.noise.fbm(
                p + Vec3::new(-5.9, 13.1, 7.3),
                WARP_OCTAVES,
                LACUNARITY,
                GAIN,
            ),
            self.noise.fbm(
                p + Vec3::new(3.7, -8.2, 19.4),
                WARP_OCTAVES,
                LACUNARITY,
                GAIN,
            ),
        );
        (q + v * WARP_AMP).normalize_or(q)
    }

    /// Modulated cap radius, radians, at (already warped) direction `w`.
    fn radius_at(&self, w: Vec3) -> f32 {
        let n = self
            .noise
            .fbm(w * OUTLINE_FREQ, self.detail_octaves, LACUNARITY, GAIN);
        self.radius_rad * (1.0 + OUTLINE_AMP * n).max(MIN_RADIUS_FRAC)
    }

    /// Area centroid of the mask, in the local frame, by spiral quadrature over
    /// the bounding cap. Only the low-frequency content matters here, so a few
    /// thousand samples locate it to well under a cell.
    fn centroid(&self) -> Vec3 {
        let cos_cap = self.cos_bound();
        let mut sum = Vec3::ZERO;
        for i in 0..CENTROID_SAMPLES {
            let u = (i as f32 + 0.5) / CENTROID_SAMPLES as f32;
            let cz = 1.0 - u * (1.0 - cos_cap);
            let sr = (1.0 - cz * cz).max(0.0).sqrt();
            let phi = i as f32 * GOLDEN_ANGLE;
            let q = Vec3::new(sr * phi.cos(), sr * phi.sin(), cz);
            if self.contains(q).is_some() {
                sum += q;
            }
        }
        sum.normalize_or(Vec3::Z)
    }
}

/// The craton shapes of one planet. Rebuilt (identically) by any process
/// instance from `(seed, count, detail octaves)`; cached because the centroid
/// quadrature is not free.
pub(crate) struct CratonSet {
    seed: u64,
    detail_octaves: u32,
    shapes: Vec<CratonShape>,
}

impl CratonSet {
    /// Build the shapes for `count` cratons. `pitch_m` is the mesh's mean cell
    /// spacing; it fixes how many octaves the coastline gets, so a level-9
    /// planet has cell-scale crenellation and a level-5 planet does not waste
    /// time on octaves it cannot resolve.
    pub(crate) fn new(seed: u64, count: usize, pitch_m: f64) -> CratonSet {
        let detail_octaves = detail_octaves(pitch_m);
        let mut rng = rng_for(seed, "tectonics/craton-shape", 0);
        let shapes = (0..count)
            .map(|_| {
                let radius_m = next_radius_m(&mut rng, count);
                let mut shape = CratonShape {
                    radius_m,
                    radius_rad: (radius_m / EARTH_RADIUS_M) as f32,
                    noise: Noise3::new(rng.random::<u64>()),
                    detail_octaves,
                    centroid_local: Vec3::Z,
                };
                shape.centroid_local = shape.centroid();
                shape
            })
            .collect();
        CratonSet {
            seed,
            detail_octaves,
            shapes,
        }
    }

    /// Just the mean radii, without the centroid quadrature — for callers that
    /// only need the seeder's spacing constraint.
    pub(crate) fn radii_m(seed: u64, count: usize) -> Vec<f64> {
        let mut rng = rng_for(seed, "tectonics/craton-shape", 0);
        (0..count)
            .map(|_| {
                let r = next_radius_m(&mut rng, count);
                // Keep the draw sequence identical to `new`.
                let _: u64 = rng.random();
                r
            })
            .collect()
    }

    /// True when this set is the one `(seed, count, pitch)` asks for.
    pub(crate) fn matches(&self, seed: u64, count: usize, pitch_m: f64) -> bool {
        self.seed == seed
            && self.shapes.len() == count
            && self.detail_octaves == detail_octaves(pitch_m)
    }

    pub(crate) fn shapes(&self) -> &[CratonShape] {
        &self.shapes
    }
}

/// Next craton's mean radius from the shape stream. Both constructors draw it
/// the same way so `radii_m` and `new` agree.
fn next_radius_m(rng: &mut rand_pcg::Pcg64Mcg, count: usize) -> f64 {
    cap_radius_m(
        CONTINENTAL_TARGET_FRACTION / count.max(1) as f64
            * rng.random_range(CRATON_AREA_SPREAD.0..CRATON_AREA_SPREAD.1),
    )
}

/// Octaves needed for outline detail down to ~2 cells, clamped to a sane band.
fn detail_octaves(pitch_m: f64) -> u32 {
    let finest = EARTH_RADIUS_M / (2.0 * pitch_m.max(1.0)); // cycles per radian
    let octaves = (finest / OUTLINE_FREQ as f64).max(1.0).log2().ceil() + 1.0;
    (octaves as u32).clamp(4, 9)
}

/// Great-circle radius of a spherical cap covering `fraction` of the sphere.
fn cap_radius_m(fraction: f64) -> f64 {
    let cos_theta = (1.0 - 2.0 * fraction).clamp(-0.99, 1.0);
    cos_theta.acos() * EARTH_RADIUS_M
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Level-6-ish pitch.
    const PITCH: f64 = 70_000.0;

    #[test]
    fn radii_match_the_full_build() {
        let set = CratonSet::new(42, 8, PITCH);
        let radii = CratonSet::radii_m(42, 8);
        let full: Vec<f64> = set.shapes().iter().map(|s| s.radius_m).collect();
        assert_eq!(radii, full);
    }

    #[test]
    fn shapes_are_deterministic() {
        let a = CratonSet::new(42, 8, PITCH);
        let b = CratonSet::new(42, 8, PITCH);
        let q = Vec3::new(0.2, 0.1, 0.97).normalize();
        for (x, y) in a.shapes().iter().zip(b.shapes()) {
            assert_eq!(x.radius_m, y.radius_m);
            assert_eq!(x.centroid_local, y.centroid_local);
            assert_eq!(x.contains(q), y.contains(q));
        }
    }

    #[test]
    fn outline_is_irregular_but_bounded() {
        let set = CratonSet::new(7, 6, PITCH);
        for s in set.shapes() {
            let mut min = f32::INFINITY;
            let mut max: f32 = 0.0;
            for i in 0..256 {
                let phi = i as f32 / 256.0 * std::f32::consts::TAU;
                // Walk out along one azimuth until we leave the shape.
                let mut last = 0.0f32;
                for j in 1..400 {
                    let d = j as f32 * 0.002;
                    let q = Vec3::new(d.sin() * phi.cos(), d.sin() * phi.sin(), d.cos());
                    if s.contains(q).is_some() {
                        last = d;
                    }
                }
                min = min.min(last);
                max = max.max(last);
            }
            let mean = s.radius_m as f32 / EARTH_RADIUS_M as f32;
            println!("outline {min:.3}..{max:.3} about {mean:.3}");
            assert!(
                max > mean * 1.25 && min < mean * 0.85,
                "outline too round: {min}..{max} about {mean}"
            );
            assert!(max < mean * 2.5, "outline runaway: {max} vs {mean}");
        }
    }

    /// The stored centroid is what `phase1` aligns on every step, so the
    /// quadrature has to have converged: an error there is a per-step placement
    /// error, i.e. a craton that crawls sideways forever.
    #[test]
    fn centroid_quadrature_has_converged() {
        let set = CratonSet::new(1337, 8, PITCH);
        for s in set.shapes() {
            // Independent, 8x denser quadrature with a different spiral phase.
            let cos_cap = s.cos_bound();
            let m = CENTROID_SAMPLES * 8;
            let mut sum = Vec3::ZERO;
            for i in 0..m {
                let u = (i as f32 + 0.31) / m as f32;
                let cz = 1.0 - u * (1.0 - cos_cap);
                let sr = (1.0 - cz * cz).max(0.0).sqrt();
                let phi = i as f32 * GOLDEN_ANGLE + 0.7;
                let q = Vec3::new(sr * phi.cos(), sr * phi.sin(), cz);
                if s.contains(q).is_some() {
                    sum += q;
                }
            }
            let err = s.centroid_local.angle_between(sum.normalize());
            // An error here is a constant per-step placement offset, i.e. an
            // extra drift velocity of `err * EARTH_RADIUS / dt`. 2e-3 rad is
            // 13 km per 1 Myr step against a craton drift of ~90 km: a few per
            // cent of speed error, and — being constant in the craton frame —
            // smooth motion, not the boundary flicker that leaves banding.
            println!("centroid err {err:.5}");
            assert!(err < 2e-3, "centroid off by {err} rad");
        }
    }

    #[test]
    fn detail_scales_with_resolution() {
        assert!(detail_octaves(70_000.0) < detail_octaves(9_000.0));
        assert_eq!(detail_octaves(1.0), 9);
    }
}
