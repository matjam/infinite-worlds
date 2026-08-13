//! Acceptance tests for the Goldberg planet mesh (IMPLEMENTATION_PLAN.md §3 WP1).

use std::f64::consts::PI;

use glam::Vec3;
use iw_mesh::{great_circle_km, latlon_of, Mesh, EARTH_RADIUS_KM};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

/// Level used by everything that doesn't need a specific one; ~41k cells.
const L: u8 = 6;

fn mesh(level: u8) -> Mesh {
    Mesh::build(level)
}

fn unit_sample(rng: &mut Pcg64Mcg) -> Vec3 {
    loop {
        let v = Vec3::new(
            rng.random_range(-1.0f32..1.0),
            rng.random_range(-1.0f32..1.0),
            rng.random_range(-1.0f32..1.0),
        );
        let l2 = v.length_squared();
        if (1e-4..=1.0).contains(&l2) {
            return v / l2.sqrt();
        }
    }
}

/// Reference implementation: the cell whose center maximizes `dot`. Uses `f64`
/// for the same reason `cell_at` does — an `f32` dot near 1.0 has ~1 ULP of
/// resolution, which is coarser than the gap between adjacent cell centers at
/// high levels, so an `f32` scan would report arbitrary winners near cell edges.
fn brute_force_cell(m: &Mesh, d: Vec3) -> u32 {
    let d = d.as_dvec3().normalize();
    let mut best = 0u32;
    let mut best_dot = f64::NEG_INFINITY;
    for (i, c) in m.centers.iter().enumerate() {
        let s = d.dot(c.as_dvec3());
        if s > best_dot {
            best_dot = s;
            best = i as u32;
        }
    }
    best
}

#[test]
fn cell_count_formula() {
    for level in 3..=6u8 {
        let m = mesh(level);
        assert_eq!(m.n_cells(), 10 * 4usize.pow(level as u32) + 2);
        assert_eq!(m.n_cells(), Mesh::expected_cells(level));
        assert_eq!(m.level, level);
        assert_eq!(m.areas_km2.len(), m.n_cells());
        assert_eq!(m.latlon.len(), m.n_cells());
        assert_eq!(m.cell_chunk.len(), m.n_cells());
        assert_eq!(m.neighbor_offsets.len(), m.n_cells() + 1);
        assert_eq!(m.corner_offsets.len(), m.n_cells() + 1);
        // One dual corner per primal triangle, each shared by exactly 3 cells.
        assert_eq!(m.vertices.len(), 20 * 4usize.pow(level as u32));
        assert_eq!(m.corners.len(), 3 * m.vertices.len());
    }
}

#[test]
fn degenerate_levels_build() {
    for level in 0..=2u8 {
        let m = mesh(level);
        assert_eq!(m.n_cells(), Mesh::expected_cells(level));
        assert!(!m.chunks.is_empty());
        assert!(m.chunks.iter().all(|c| !c.cells.is_empty()));
    }
}

#[test]
fn exactly_twelve_pentagons() {
    for level in 3..=6u8 {
        let m = mesh(level);
        let pent: Vec<u32> = (0..m.n_cells() as u32)
            .filter(|c| m.is_pentagon(*c))
            .collect();
        assert_eq!(pent.len(), 12, "level {level}");
        // Pentagons are the original icosahedron vertices == cells 0..12.
        assert_eq!(pent, (0..12u32).collect::<Vec<_>>());
        for c in 0..m.n_cells() as u32 {
            let want = if c < 12 { 5 } else { 6 };
            assert_eq!(m.corners_of(c).len(), want, "cell {c} corner count");
            assert_eq!(m.neighbors_of(c).len(), want, "cell {c} neighbor count");
        }
    }
}

#[test]
fn adjacency_is_symmetric_and_well_formed() {
    let m = mesh(L);
    for c in 0..m.n_cells() as u32 {
        let ns = m.neighbors_of(c);
        assert_eq!(ns.len(), m.corners_of(c).len());
        let mut seen = ns.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), ns.len(), "cell {c} has a duplicate neighbor");
        for &n in ns {
            assert_ne!(n, c);
            assert!(m.neighbors_of(n).contains(&c), "{c}->{n} not symmetric");
            // Neighbors share exactly two corners (one edge).
            let shared = m
                .corners_of(c)
                .iter()
                .filter(|v| m.corners_of(n).contains(v))
                .count();
            assert_eq!(shared, 2, "cells {c},{n} share {shared} corners");
        }
    }
}

#[test]
fn corner_sharing_is_exactly_three_cells() {
    let m = mesh(4);
    let mut use_count = vec![0u32; m.vertices.len()];
    for v in &m.corners {
        use_count[*v as usize] += 1;
    }
    assert!(use_count.iter().all(|c| *c == 3));
}

#[test]
fn areas_sum_to_sphere_and_are_near_uniform() {
    for level in 3..=6u8 {
        let m = mesh(level);
        let total: f64 = m.areas_km2.iter().map(|a| *a as f64).sum();
        let sphere = 4.0 * PI * (EARTH_RADIUS_KM as f64) * (EARTH_RADIUS_KM as f64);
        let rel = (total - sphere).abs() / sphere;
        assert!(
            rel < 0.005,
            "level {level}: area off by {:.4}%",
            rel * 100.0
        );

        let max = m.areas_km2.iter().cloned().fold(f32::MIN, f32::max);
        let min = m.areas_km2.iter().cloned().fold(f32::MAX, f32::min);
        assert!(min > 0.0);
        assert!(max / min < 2.0, "level {level}: area ratio {}", max / min);
    }
}

#[test]
fn corners_are_ccw_and_inside_circumradius() {
    let m = mesh(5);
    for c in 0..m.n_cells() as u32 {
        let center = m.centers[c as usize];
        assert!((center.length() - 1.0).abs() < 1e-5);
        let cs = m.corners_of(c);
        let mut circum = 0.0f32;
        for (k, v) in cs.iter().enumerate() {
            let a = m.vertices[*v as usize];
            let b = m.vertices[cs[(k + 1) % cs.len()] as usize];
            assert!((a.length() - 1.0).abs() < 1e-5);
            // CCW seen from outside: the fan triangle normal points outward.
            assert!(
                (a - center).cross(b - center).dot(center) > 0.0,
                "cell {c} corner {k} not CCW"
            );
            circum = circum.max(great_circle_km(center, a));
        }
        // Every corner is nearer to its own center than to any neighbor's, so
        // the cell is star-shaped and the circumradius bounds all corners.
        let nearest_neighbor = m
            .neighbors_of(c)
            .iter()
            .map(|n| great_circle_km(center, m.centers[*n as usize]))
            .fold(f32::MAX, f32::min);
        assert!(
            circum < nearest_neighbor,
            "cell {c}: circumradius {circum} exceeds half-pitch"
        );
        for v in cs {
            assert!(great_circle_km(center, m.vertices[*v as usize]) <= circum + 1e-3);
        }
    }
}

#[test]
fn latlon_matches_centers() {
    let m = mesh(4);
    for (c, ll) in m.latlon.iter().enumerate() {
        let want = latlon_of(m.centers[c]);
        assert!((ll[0] - want[0]).abs() < 1e-6);
        assert!((ll[1] - want[1]).abs() < 1e-6);
    }
}

#[test]
fn cell_at_recovers_own_center() {
    let m = mesh(5);
    let n = m.n_cells();
    let step = (n / 500).max(1);
    let mut checked = 0;
    for i in (0..n).step_by(step) {
        assert_eq!(m.cell_at(m.centers[i]), i as u32, "cell {i}");
        checked += 1;
    }
    assert!(checked >= 500, "sampled only {checked} cells");
    // Pentagons explicitly.
    for i in 0..12usize {
        assert_eq!(m.cell_at(m.centers[i]), i as u32);
    }
}

#[test]
fn cell_at_matches_brute_force_on_random_dirs() {
    let m = mesh(5);
    let mut rng = Pcg64Mcg::seed_from_u64(0xC0FFEE);
    for k in 0..1000 {
        let d = unit_sample(&mut rng);
        let got = m.cell_at(d);
        let want = brute_force_cell(&m, d);
        assert_eq!(got, want, "sample {k} dir {d:?}");
        // Non-unit input must behave identically. Scaling by a power of two is
        // exact in f32, so the answer must be bit-for-bit the same.
        assert_eq!(m.cell_at(d * 64.0), want);
        assert_eq!(m.cell_at(d * 0.03125), want);
    }
    // Degenerate input is handled, not panicking.
    let _ = m.cell_at(Vec3::ZERO);
}

#[test]
fn cell_at_matches_brute_force_at_level6() {
    let m = mesh(L);
    let mut rng = Pcg64Mcg::seed_from_u64(7);
    for k in 0..300 {
        let d = unit_sample(&mut rng);
        assert_eq!(m.cell_at(d), brute_force_cell(&m, d), "sample {k}");
    }
}

#[test]
fn chunks_partition_cells_and_cones_cover_corners() {
    for level in [3u8, 6, 8] {
        let m = mesh(level);
        assert_eq!(m.chunks.len(), if level >= 8 { 80 } else { 20 });
        let mut seen = vec![false; m.n_cells()];
        for (ci, ch) in m.chunks.iter().enumerate() {
            assert!(!ch.cells.is_empty());
            assert!((ch.center.length() - 1.0).abs() < 1e-5);
            for &c in &ch.cells {
                assert!(!seen[c as usize], "cell {c} in two chunks");
                seen[c as usize] = true;
                assert_eq!(m.cell_chunk[c as usize] as usize, ci);
                assert!(
                    ch.center.dot(m.centers[c as usize]) >= ch.cos_radius,
                    "chunk {ci} cone misses center of cell {c}"
                );
                for v in m.corners_of(c) {
                    assert!(
                        ch.center.dot(m.vertices[*v as usize]) >= ch.cos_radius,
                        "chunk {ci} cone misses a corner of cell {c}"
                    );
                }
            }
        }
        assert!(
            seen.iter().all(|s| *s),
            "level {level}: cells left unchunked"
        );
    }
}

#[test]
fn build_is_deterministic() {
    let a = mesh(4);
    let b = mesh(4);
    assert_eq!(a.centers, b.centers);
    assert_eq!(a.areas_km2, b.areas_km2);
    assert_eq!(a.latlon, b.latlon);
    assert_eq!(a.neighbors, b.neighbors);
    assert_eq!(a.corners, b.corners);
    assert_eq!(a.vertices, b.vertices);
    assert_eq!(a.cell_chunk, b.cell_chunk);
    for (x, y) in a.chunks.iter().zip(b.chunks.iter()) {
        assert_eq!(x.cells, y.cells);
        assert_eq!(x.center, y.center);
        assert_eq!(x.cos_radius, y.cos_radius);
    }
}

#[test]
fn east_north_is_an_orthonormal_tangent_frame() {
    let m = mesh(3);
    for c in 0..m.n_cells() as u32 {
        let r = m.centers[c as usize];
        let (e, n) = m.east_north(c);
        assert!((e.length() - 1.0).abs() < 1e-4);
        assert!((n.length() - 1.0).abs() < 1e-4);
        assert!(e.dot(r).abs() < 1e-4);
        assert!(n.dot(r).abs() < 1e-4);
        assert!(e.dot(n).abs() < 1e-4);
        // East points along increasing longitude.
        assert!(e.cross(n).dot(r) > 0.0);
    }
}

#[test]
#[ignore = "performance budget; run with --ignored"]
fn build_level8_performance() {
    let t0 = std::time::Instant::now();
    let m = Mesh::build(8);
    let dt = t0.elapsed();
    assert_eq!(m.n_cells(), Mesh::expected_cells(8));
    println!("Mesh::build(8) -> {} cells in {:.3?}", m.n_cells(), dt);
    assert!(dt.as_secs_f64() < 10.0, "level 8 build took {dt:?}");
}
