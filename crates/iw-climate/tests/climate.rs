//! Acceptance tests for the climate model (IMPLEMENTATION_PLAN.md §3 WP6).

use glam::Vec3;
use iw_climate::*;
use iw_core::{MassLedger, NullProgress, Phase, Planet, PlanetConfig, Process, StepCtx};
use iw_mesh::Mesh;

const OCEAN_FLOOR_M: f32 = -4000.0;

fn config(level: u8) -> PlanetConfig {
    PlanetConfig {
        subdivision_level: level,
        ..PlanetConfig::default()
    }
}

/// All-ocean planet: uniform abyssal floor, sea level 0.
fn aquaplanet(level: u8) -> (Planet, Mesh) {
    let mesh = Mesh::build(level);
    let mut planet = Planet::new(config(level), mesh.n_cells());
    planet.elevation_m.fill(OCEAN_FLOOR_M);
    planet.sea_level_m = 0.0;
    (planet, mesh)
}

fn run(planet: &mut Planet, mesh: &Mesh, process: &mut ClimateProcess, steps: usize) {
    let progress = NullProgress;
    for _ in 0..steps {
        let mut ledger = MassLedger::default();
        let mut ctx = StepCtx {
            rng: iw_core::rng_for(planet.config.seed, "climate", planet.step_index),
            progress: &progress,
            ledger: &mut ledger,
        };
        process.step(planet, mesh, 1.0, &mut ctx);
        planet.step_index += 1;
    }
}

fn dir_at(lat_deg: f32, lon_deg: f32) -> Vec3 {
    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
    Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
}

fn lat_deg(mesh: &Mesh, cell: usize) -> f32 {
    mesh.latlon[cell][0].to_degrees()
}

fn lon_deg(mesh: &Mesh, cell: usize) -> f32 {
    mesh.latlon[cell][1].to_degrees()
}

/// Area-weighted mean of `field` over the cells passing `keep`.
fn mean_where(mesh: &Mesh, field: &[f32], keep: impl Fn(usize) -> bool) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, v) in field.iter().enumerate() {
        if keep(i) {
            num += *v as f64 * mesh.areas_km2[i] as f64;
            den += mesh.areas_km2[i] as f64;
        }
    }
    assert!(den > 0.0, "empty selection");
    (num / den) as f32
}

fn area_total(mesh: &Mesh, field: &[f32]) -> f64 {
    (0..field.len())
        .map(|i| field[i] as f64 * mesh.areas_km2[i] as f64)
        .sum()
}

// --- temperature ---------------------------------------------------------

#[test]
fn equator_hotter_than_poles() {
    let (mut planet, mesh) = aquaplanet(5);
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 1);

    let tropics = mean_where(&mesh, &planet.temperature_c, |i| {
        lat_deg(&mesh, i).abs() < 10.0
    });
    let mid = mean_where(&mesh, &planet.temperature_c, |i| {
        (40.0..50.0).contains(&lat_deg(&mesh, i).abs())
    });
    let polar = mean_where(&mesh, &planet.temperature_c, |i| {
        lat_deg(&mesh, i).abs() > 80.0
    });

    assert!(
        tropics > mid && mid > polar,
        "monotone equator->pole gradient expected, got {tropics} / {mid} / {polar}"
    );
    assert!(
        (tropics - 26.0).abs() < 3.0,
        "tropical mean {tropics} out of range"
    );
    assert!(polar < -15.0, "polar mean {polar} not cold enough");
}

#[test]
fn lapse_rate_cools_mountains() {
    let (mut planet, mesh) = aquaplanet(5);
    // Two single-cell islands near the equator, where the latitudinal curve is
    // flat, so their difference isolates the lapse rate. Both are coastal, so
    // the maritime term cancels too.
    let low = mesh.cell_at(dir_at(0.0, 0.0)) as usize;
    let high = mesh.cell_at(dir_at(0.0, 90.0)) as usize;
    planet.elevation_m[low] = 5.0;
    planet.elevation_m[high] = 3005.0;
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 1);

    let drop = planet.temperature_c[low] - planet.temperature_c[high];
    assert!(
        (drop - 19.5).abs() < 0.5,
        "3 km of relief should cool by ~19.5 C (6.5 C/km), got {drop}"
    );
    // The pure helper must match the field up to the (uniform) global
    // thermostat offset: the same offset the low island received.
    let helper = |i: usize, surface: f32| {
        annual_mean_temperature_c(
            mesh.latlon[i][0],
            surface,
            1,
            &planet.config,
            planet.time_myr,
        )
    };
    let offset_low = planet.temperature_c[low] - helper(low, 5.0);
    let offset_high = planet.temperature_c[high] - helper(high, 3005.0);
    assert!(
        (offset_high - offset_low).abs() < 1e-4,
        "thermostat offset must be uniform: {offset_low} vs {offset_high}"
    );
}

/// Zonal continent spanning 35..65 deg N and 180 deg of longitude: wide enough
/// that its middle saturates the distance-to-ocean cap.
fn continent_planet(level: u8) -> (Planet, Mesh) {
    let (mut planet, mesh) = aquaplanet(level);
    for i in 0..mesh.n_cells() {
        let (la, lo) = (lat_deg(&mesh, i), lon_deg(&mesh, i));
        if (35.0..65.0).contains(&la) && lo.abs() < 90.0 {
            planet.elevation_m[i] = 200.0;
        }
    }
    (planet, mesh)
}

#[test]
fn maritime_moderates_interiors() {
    let (mut planet, mesh) = continent_planet(5);
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 1);

    let dist = distance_to_ocean_cells(&planet, &mesh);
    let amp = seasonal_amplitude_c(&planet, &mesh);
    let (_summer, winter) = seasonal_extremes_c(&planet, &mesh);

    let band = |i: usize| (46.0..54.0).contains(&lat_deg(&mesh, i));
    let coastal = |i: usize| band(i) && dist[i] == 1;
    let interior = |i: usize| band(i) && dist[i] >= 6;

    let amp_coast = mean_where(&mesh, &amp, coastal);
    let amp_inland = mean_where(&mesh, &amp, interior);
    assert!(
        amp_inland > amp_coast * 1.5,
        "continental interiors must be far more seasonal: {amp_inland} vs {amp_coast}"
    );

    let win_coast = mean_where(&mesh, &winter, coastal);
    let win_inland = mean_where(&mesh, &winter, interior);
    assert!(
        win_coast > win_inland + 5.0,
        "coastal winters must be milder: {win_coast} vs {win_inland}"
    );

    // Annual mean is also nudged toward the maritime reference at high latitude.
    let t_coast = mean_where(&mesh, &planet.temperature_c, coastal);
    let t_inland = mean_where(&mesh, &planet.temperature_c, interior);
    assert!(t_coast > t_inland, "{t_coast} vs {t_inland}");
}

#[test]
fn temperature_offset_shifts_global_mean() {
    let (mut base, mesh) = aquaplanet(5);
    run(&mut base, &mesh, &mut ClimateProcess::new(), 1);

    let mut warmed = base.clone();
    warmed.config.temperature_offset_c = 4.5;
    run(&mut warmed, &mesh, &mut ClimateProcess::new(), 1);

    let m0 = mean_where(&mesh, &base.temperature_c, |_| true);
    let m1 = mean_where(&mesh, &warmed.temperature_c, |_| true);
    assert!(
        (m1 - m0 - 4.5).abs() < 1e-3,
        "offset must shift the global mean exactly: {m0} -> {m1}"
    );
}

// --- glacial forcing -----------------------------------------------------

#[test]
fn glacial_forcing_shape() {
    let cfg = config(5);
    let start: f64 = cfg.phase_durations_myr[..3].iter().sum();
    assert_eq!(cfg.phase_durations_myr[Phase::RecentPast.index()], 2.0);

    // Zero everywhere outside RecentPast.
    for t in [0.0, 100.0, start - 1e-6, start + 2.0 + 1e-6, start + 50.0] {
        assert_eq!(glacial_forcing_c(&cfg, t), 0.0, "t = {t}");
    }
    // Interglacial at the phase boundary and at every cycle boundary.
    for k in 0..10 {
        let t = start + k as f64 * GLACIAL_PERIOD_MYR;
        assert!(glacial_forcing_c(&cfg, t).abs() < 1e-5, "t = {t}");
    }
    // Glacial maximum half a cycle later.
    for k in 0..10 {
        let t = start + (k as f64 + 0.5) * GLACIAL_PERIOD_MYR;
        assert!(
            (glacial_forcing_c(&cfg, t) + GLACIAL_DEPTH_C).abs() < 1e-4,
            "t = {t}"
        );
    }
    // Never positive, never below the full depth.
    let mut min = 0.0f32;
    for k in 0..2001 {
        let t = start + k as f64 * 0.001;
        let f = glacial_forcing_c(&cfg, t);
        assert!((-GLACIAL_DEPTH_C..=0.0).contains(&f));
        min = min.min(f);
    }
    assert!((min + GLACIAL_DEPTH_C).abs() < 1e-3);
}

#[test]
fn glacial_forcing_scales_with_intensity() {
    let mut cfg = config(5);
    let start: f64 = cfg.phase_durations_myr[..3].iter().sum();
    let peak = start + 0.5 * GLACIAL_PERIOD_MYR;

    cfg.glacial_intensity = 0.0;
    assert_eq!(glacial_forcing_c(&cfg, peak), 0.0);
    cfg.glacial_intensity = 0.5;
    assert!((glacial_forcing_c(&cfg, peak) + 3.0).abs() < 1e-4);
    cfg.glacial_intensity = 2.0;
    assert!((glacial_forcing_c(&cfg, peak) + 12.0).abs() < 1e-4);
}

#[test]
fn glacial_forcing_reaches_the_temperature_field() {
    let (mut planet, mesh) = aquaplanet(5);
    let start: f64 = planet.config.phase_durations_myr[..3].iter().sum();
    planet.phase = Phase::RecentPast;
    planet.time_myr = start + 0.5 * GLACIAL_PERIOD_MYR;
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 1);
    let cold = mean_where(&mesh, &planet.temperature_c, |_| true);

    let mut warm = aquaplanet(5).0;
    warm.time_myr = start;
    warm.phase = Phase::RecentPast;
    run(&mut warm, &mesh, &mut ClimateProcess::new(), 1);
    let interglacial = mean_where(&mesh, &warm.temperature_c, |_| true);

    assert!(
        (interglacial - cold - GLACIAL_DEPTH_C).abs() < 1e-3,
        "{interglacial} vs {cold}"
    );
}

// --- winds ---------------------------------------------------------------

#[test]
fn wind_belts() {
    let mesh = Mesh::build(5);
    let wind = wind_field(&mesh);

    for (cell, w) in wind.iter().enumerate() {
        let (east, north) = mesh.east_north(cell as u32);
        let (u, v) = (w.dot(east), w.dot(north));
        let la = lat_deg(&mesh, cell);
        let a = la.abs();
        // Poleward is +north in the north, -north in the south.
        let poleward = if la >= 0.0 { v } else { -v };

        if (10.0..20.0).contains(&a) {
            assert!(u < -3.0, "trades must blow west at {la}: u = {u}");
            assert!(poleward < -2.0, "trades must blow equatorward at {la}: {v}");
            assert!(
                (5.0..=10.0).contains(&w.length()),
                "trade speed {} at {la}",
                w.length()
            );
        }
        if (40.0..50.0).contains(&a) {
            assert!(u > 3.0, "westerlies must blow east at {la}: u = {u}");
            assert!(poleward > 1.0, "westerlies must blow poleward at {la}: {v}");
            assert!(
                (5.0..=10.0).contains(&w.length()),
                "westerly speed {} at {la}",
                w.length()
            );
        }
        if (70.0..80.0).contains(&a) {
            assert!(u < -1.0, "polar easterlies must blow west at {la}: u = {u}");
            assert!(poleward < -1.0, "polar easterlies blow equatorward: {v}");
        }
        assert!(w.length() <= 10.0, "wind {} too strong at {la}", w.length());
        assert!(
            w.dot(mesh.centers[cell]).abs() < 1e-4,
            "wind not tangent at cell {cell}"
        );
    }
}

// --- process contract ----------------------------------------------------

#[test]
fn process_contract() {
    // iw-headless constructs the process with Default.
    let process = ClimateProcess::default();
    assert_eq!(process.name(), "climate");
    let boxed: Box<dyn Process> = Box::new(ClimateProcess::new());
    assert_eq!(boxed.name(), "climate");
}

#[test]
fn writes_only_its_own_fields() {
    let (mut planet, mesh) = ridge_planet(5);
    let before = planet.clone();
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 3);

    assert_eq!(planet.elevation_m, before.elevation_m);
    assert_eq!(planet.sea_level_m, before.sea_level_m);
    assert_eq!(planet.sediment_m, before.sediment_m);
    assert_eq!(planet.ice_thickness_m, before.ice_thickness_m);
    assert_eq!(planet.lake_depth_m, before.lake_depth_m);
    assert_eq!(planet.water_flux_m3_yr, before.water_flux_m3_yr);
    assert_eq!(planet.crust_thickness_m, before.crust_thickness_m);
    assert_eq!(planet.plate_id, before.plate_id);
    assert_ne!(planet.temperature_c, before.temperature_c);
    assert_ne!(planet.precip_mm_yr, before.precip_mm_yr);
}

#[test]
fn wind_field_is_written_to_the_planet() {
    let (mut planet, mesh) = aquaplanet(5);
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 1);
    let expect = wind_field(&mesh);
    assert_eq!(planet.wind_m_s, expect);
}

// --- precipitation -------------------------------------------------------

#[test]
fn aquaplanet_precipitation_belts() {
    let (mut planet, mesh) = aquaplanet(5);
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 30);
    let p = &planet.precip_mm_yr;

    let band =
        |lo: f32, hi: f32| mean_where(&mesh, p, |i| (lo..hi).contains(&lat_deg(&mesh, i).abs()));
    let equator = band(0.0, 10.0);
    let tropic = band(10.0, 20.0);
    let horse = band(25.0, 35.0);
    let mid = band(40.0, 50.0);
    let sub_polar = band(55.0, 65.0);
    let polar = band(75.0, 90.0);

    assert!(
        equator > tropic && equator > mid,
        "equator must be wettest: {equator} vs {tropic} / {mid}"
    );
    assert!(
        horse < tropic && horse < mid,
        "~30 deg must be a local minimum: {horse} between {tropic} and {mid}"
    );
    assert!(
        polar < mid && polar < sub_polar,
        "poles must be dry: {polar} vs {mid} / {sub_polar}"
    );
    assert!(equator > 1000.0, "equatorial precip {equator} too low");
}

/// Aquaplanet plus a north-south land band in the westerly belt. With `ridge`,
/// the band carries a meridional range: a windward ramp from the west coast up
/// to a 3 km crest at lon 0, then a low lee plain. Without it the same land is
/// a uniform 150 m plain — the control that separates the orographic effect
/// from plain distance-from-the-sea depletion.
fn land_band_planet(level: u8, ridge: bool) -> (Planet, Mesh) {
    let (mut planet, mesh) = aquaplanet(level);
    for i in 0..mesh.n_cells() {
        let (la, lo) = (lat_deg(&mesh, i), lon_deg(&mesh, i));
        if !(30.0..60.0).contains(&la) || lo.abs() > 14.0 {
            continue;
        }
        planet.elevation_m[i] = if !ridge {
            150.0
        } else if lo <= 0.0 {
            // West flank: 100 m at lon -14 rising to 3000 m at the crest.
            100.0 + 2900.0 * (1.0 + lo / 14.0)
        } else {
            // Lee: drops straight off the crest to a low plain.
            (3000.0 - 1400.0 * lo).max(150.0)
        };
    }
    (planet, mesh)
}

fn ridge_planet(level: u8) -> (Planet, Mesh) {
    land_band_planet(level, true)
}

/// Mean precipitation on the windward slope (the lower half of the west flank,
/// where the air does its climbing) and on the lee plain.
fn flank_precip(planet: &Planet, mesh: &Mesh) -> (f32, f32) {
    let band = |i: usize| (38.0..52.0).contains(&lat_deg(mesh, i));
    let windward = mean_where(mesh, &planet.precip_mm_yr, |i| {
        band(i) && (-14.0..-6.0).contains(&lon_deg(mesh, i))
    });
    let leeward = mean_where(mesh, &planet.precip_mm_yr, |i| {
        band(i) && (2.0..12.0).contains(&lon_deg(mesh, i))
    });
    (windward, leeward)
}

#[test]
fn ridge_casts_a_rain_shadow() {
    let (mut with_ridge, mesh) = land_band_planet(5, true);
    let (mut flat, _) = land_band_planet(5, false);
    run(&mut with_ridge, &mesh, &mut ClimateProcess::new(), 30);
    run(&mut flat, &mesh, &mut ClimateProcess::new(), 30);

    let (windward, leeward) = flank_precip(&with_ridge, &mesh);
    let (flat_west, flat_east) = flank_precip(&flat, &mesh);
    assert!(
        windward >= 2.0 * leeward,
        "rain shadow too weak: windward {windward} vs leeward {leeward}"
    );

    // The same land without the range: the contrast must come from the relief,
    // not merely from distance to the coast.
    let shadow = leeward / windward;
    let control = flat_east / flat_west;
    assert!(
        shadow < 0.4 * control,
        "the range must deepen the shadow: lee/windward {shadow} with relief \
         vs {control} on flat land ({windward}/{leeward} vs {flat_west}/{flat_east})"
    );
    // Band MEAN, not peak: per-edge orographic rainout concentrates the
    // squeeze at the leading slope cells and depletes the air across the
    // rest of the band, so the mean gain over flat land is modest even when
    // the shadow (asserted at 2x above) is strong. The claim here is only
    // that relief wets its windward side at all.
    assert!(
        windward > 1.02 * flat_west,
        "orographic lift must wet the windward slope: {windward} vs {flat_west}"
    );
}

#[test]
fn precip_multiplier_scales_totals() {
    let (mut base, mesh) = ridge_planet(5);
    let mut doubled = base.clone();
    doubled.config.precip_multiplier = 2.0;
    run(&mut base, &mesh, &mut ClimateProcess::new(), 40);
    run(&mut doubled, &mesh, &mut ClimateProcess::new(), 40);

    let a = area_total(&mesh, &base.precip_mm_yr);
    let b = area_total(&mesh, &doubled.precip_mm_yr);
    let ratio = b / a;
    assert!(
        (ratio - 2.0).abs() < 0.01,
        "precip_multiplier must scale totals: ratio {ratio}"
    );
}

#[test]
fn dry_interiors_and_global_water_balance() {
    let (mut planet, mesh) = continent_planet(5);
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 40);

    let dist = distance_to_ocean_cells(&planet, &mesh);
    // `dist` buckets are METRIC now (~112 km each, saturating at 8): "deep
    // interior" is the saturated bucket, >=900 km from any coast.
    let interior = mean_where(&mesh, &planet.precip_mm_yr, |i| dist[i] >= 8);
    let coastal = mean_where(&mesh, &planet.precip_mm_yr, |i| {
        dist[i] == 1 && (35.0..65.0).contains(&lat_deg(&mesh, i))
    });
    assert!(
        interior < 130.0,
        "deep interiors must be near-desert, got {interior} mm/yr"
    );
    // Metric distance buckets pull the "interior" band nearer the coast than
    // the old hop count did, softening the contrast this selection sees.
    assert!(
        coastal > 2.5 * interior.max(1.0),
        "coasts must be far wetter than interiors: {coastal} vs {interior}"
    );

    // Global precipitation matches global evaporation to within the temporal
    // blend's residual.
    let total_mm =
        area_total(&mesh, &planet.precip_mm_yr) / area_total(&mesh, &vec![1.0; mesh.n_cells()]);
    assert!(
        (200.0..2000.0).contains(&total_mm),
        "global mean precip {total_mm} mm/yr is not planet-like"
    );
}

// --- determinism and statelessness ---------------------------------------

fn fingerprint(planet: &Planet) -> Vec<u8> {
    let mut out = Vec::new();
    for t in &planet.temperature_c {
        out.extend_from_slice(&t.to_bits().to_le_bytes());
    }
    for p in &planet.precip_mm_yr {
        out.extend_from_slice(&p.to_bits().to_le_bytes());
    }
    for w in &planet.wind_m_s {
        for c in w.to_array() {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    out
}

#[test]
fn deterministic_across_runs() {
    let (mut a, mesh) = ridge_planet(5);
    let mut b = a.clone();
    run(&mut a, &mesh, &mut ClimateProcess::new(), 12);
    run(&mut b, &mesh, &mut ClimateProcess::new(), 12);
    assert_eq!(fingerprint(&a), fingerprint(&b));
}

#[test]
fn stateless_across_process_instances() {
    let (mut planet, mesh) = ridge_planet(5);
    let mut warm = ClimateProcess::new();
    run(&mut planet, &mesh, &mut warm, 5);

    // Change the coastline mid-run so a stale cache would show up.
    for i in 0..mesh.n_cells() {
        if lon_deg(&mesh, i) > 100.0 && lat_deg(&mesh, i).abs() < 25.0 {
            planet.elevation_m[i] = 500.0;
        }
    }
    let mut cold_start = planet.clone();
    run(&mut planet, &mesh, &mut warm, 1);
    run(&mut cold_start, &mesh, &mut ClimateProcess::new(), 1);
    assert_eq!(fingerprint(&planet), fingerprint(&cold_start));
}

#[test]
fn tolerates_a_planet_with_no_ocean() {
    let mesh = Mesh::build(5);
    let mut planet = Planet::new(config(5), mesh.n_cells());
    planet.elevation_m.fill(1000.0);
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 20);
    assert!(planet
        .precip_mm_yr
        .iter()
        .all(|p| p.is_finite() && *p < 1.0));
    assert!(planet.temperature_c.iter().all(|t| t.is_finite()));
}

// --- budget --------------------------------------------------------------

#[test]
#[ignore = "timing; run with --ignored --release"]
fn step_budget_level_6() {
    use std::time::Instant;
    let (mut planet, mesh) = ridge_planet(6);
    let mut process = ClimateProcess::new();
    run(&mut planet, &mesh, &mut process, 3); // warm caches

    let start = Instant::now();
    let steps = 20;
    run(&mut planet, &mesh, &mut process, steps);
    let per_step = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!("climate step at level 6: {per_step:.2} ms");
    assert!(per_step < 10.0, "climate step took {per_step:.2} ms");
}

#[test]
#[ignore = "diagnostic output only"]
fn zonal_profile_report() {
    let (mut planet, mesh) = aquaplanet(5);
    run(&mut planet, &mesh, &mut ClimateProcess::new(), 40);
    let sphere = area_total(&mesh, &vec![1.0; mesh.n_cells()]);
    println!(
        "aquaplanet global mean precip: {:.0} mm/yr",
        area_total(&mesh, &planet.precip_mm_yr) / sphere
    );
    println!("lat band | T (C) | precip (mm/yr)");
    for k in 0..9 {
        let (lo, hi) = (k as f32 * 10.0, k as f32 * 10.0 + 10.0);
        let sel = |i: usize| (lo..hi).contains(&lat_deg(&mesh, i).abs());
        let t = mean_where(&mesh, &planet.temperature_c, sel);
        let p = mean_where(&mesh, &planet.precip_mm_yr, sel);
        println!("{lo:5.0}-{hi:<3.0} | {t:6.1} | {p:8.1}");
    }

    let (mut ridge, mesh) = land_band_planet(5, true);
    let (mut flat, _) = land_band_planet(5, false);
    run(&mut ridge, &mesh, &mut ClimateProcess::new(), 40);
    run(&mut flat, &mesh, &mut ClimateProcess::new(), 40);
    println!("lon | precip across the band at 38-52 deg N | with ridge | flat");
    for k in -6..8 {
        let (lo, hi) = (k as f32 * 4.0, k as f32 * 4.0 + 4.0);
        let sel = |i: usize| {
            (38.0..52.0).contains(&lat_deg(&mesh, i)) && (lo..hi).contains(&lon_deg(&mesh, i))
        };
        let a = mean_where(&mesh, &ridge.precip_mm_yr, sel);
        let b = mean_where(&mesh, &flat.precip_mm_yr, sel);
        println!("{lo:5.0}..{hi:<4.0} | {a:8.1} | {b:8.1}");
    }
    let (windward, leeward) = flank_precip(&ridge, &mesh);
    let (flat_west, flat_east) = flank_precip(&flat, &mesh);
    println!(
        "windward {windward:.1} / leeward {leeward:.1} = {:.1}x (flat control: \
         {flat_west:.1} / {flat_east:.1} = {:.1}x)",
        windward / leeward,
        flat_west / flat_east
    );
}

/// A fixed analytic world painted onto an arbitrary mesh: one big continent
/// (30 W..60 E, 50 S..50 N) with a north-south ridge near its west coast,
/// everything else abyssal ocean. Same planet whatever the tessellation.
fn analytic_world(mesh: &Mesh) -> Planet {
    let mut planet = Planet::new(config(6), mesh.n_cells());
    planet.sea_level_m = 0.0;
    for i in 0..mesh.n_cells() {
        let (la, lo) = (lat_deg(mesh, i), lon_deg(mesh, i));
        planet.elevation_m[i] = if (-50.0..50.0).contains(&la) && (-30.0..60.0).contains(&lo) {
            // Coastal plain rising inland, plus a meridional ridge at 10 W.
            let ridge = 2500.0 * (1.0 - ((lo + 10.0) / 8.0).abs()).max(0.0);
            200.0 + ridge
        } else {
            OCEAN_FLOOR_M
        };
    }
    planet
}

/// The climate loop must be (approximately) TESSELLATION-INVARIANT: the same
/// analytic world at 40k and at 150k Voronoi cells has to converge to the
/// same land precipitation, or every calibration constant silently depends
/// on the fidelity slider. (Guards the metric sweeps/widths machinery.)
#[test]
fn precipitation_is_budget_invariant() {
    let stats = |budget: u32| -> (f32, f32) {
        let density = |_: glam::Vec3| 1.0f32;
        let mesh = Mesh::build_voronoi(budget, 7, &density);
        let mut planet = analytic_world(&mesh);
        run(&mut planet, &mesh, &mut ClimateProcess::new(), 30);
        let land_mean = mean_where(&mesh, &planet.precip_mm_yr, |i| {
            planet.elevation_m[i] >= 0.0
        });
        // Interior sample: the continent's east-central bulk, far from coasts.
        let interior = mean_where(&mesh, &planet.precip_mm_yr, |i| {
            planet.elevation_m[i] >= 0.0
                && (-25.0..25.0).contains(&lat_deg(&mesh, i))
                && (10.0..40.0).contains(&lon_deg(&mesh, i))
        });
        (land_mean, interior)
    };
    let (mean_a, int_a) = stats(40_000);
    let (mean_b, int_b) = stats(150_000);
    let rel = |a: f32, b: f32| (a - b).abs() / a.max(b).max(1.0);
    assert!(
        rel(mean_a, mean_b) < 0.20,
        "land mean precip must not depend on cell budget: {mean_a:.0} @40k vs {mean_b:.0} @150k"
    );
    assert!(
        rel(int_a, int_b) < 0.35,
        "interior precip must not depend on cell budget: {int_a:.0} @40k vs {int_b:.0} @150k"
    );
}
