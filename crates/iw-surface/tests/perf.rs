//! Performance probe and calibration report. Both are `#[ignore]`d; run with
//!
//! ```text
//! cargo test -p iw-surface --release -- --ignored --nocapture
//! ```

use iw_core::{
    rng_for, CrustType, MassLedger, NullProgress, Phase, Planet, PlanetConfig, Process, RockType,
    StepCtx,
};
use iw_mesh::Mesh;
use iw_surface::{fluvial, glacial, hillslope, hydro, SurfaceProcess};

const LEVEL: u8 = 6;

/// A planet with everything the surface engine can trip over: continents with
/// ranges and interior basins, shelves and abyss, polar ice, deserts, wind.
fn busy_planet(mesh: &Mesh) -> Planet {
    let n = mesh.n_cells();
    let mut p = Planet::new(
        PlanetConfig {
            subdivision_level: LEVEL,
            ..Default::default()
        },
        n,
    );
    p.phase = Phase::RecentPast;
    p.sea_level_m = 0.0;

    for i in 0..n {
        let [lat, lon] = mesh.latlon[i];
        // Two continents with ranges along their western margins, plus a broad
        // interior depression that has to be depression-filled every step.
        let land = (lon.sin() * 1.7 + (lat * 2.0).cos() * 0.9) > 0.35;
        let range = (lon * 3.0).sin().max(0.0) * (lat.cos()).max(0.0);
        let basin = ((lon - 1.0).powi(2) + (lat - 0.5).powi(2)) < 0.05;
        let e = if land {
            let mut e = 200.0 + 3500.0 * range;
            if basin {
                e -= 900.0;
            }
            e
        } else {
            -200.0 - 4500.0 * (1.0 - range)
        };
        p.elevation_m[i] = e;
        if e >= 0.0 {
            p.crust_type[i] = CrustType::Continental;
            p.crust_thickness_m[i] = 35_000.0 + 4.0 * e;
            p.crust_density_kg_m3[i] = 2700.0;
            p.columns
                .deposit(i as u32, RockType::Granite, 15_000.0, 0.0);
            p.columns.deposit(i as u32, RockType::Sandstone, 800.0, 0.0);
        } else {
            p.crust_type[i] = CrustType::Oceanic;
            p.crust_thickness_m[i] = 7_000.0;
            p.crust_density_kg_m3[i] = 3000.0;
            p.columns.deposit(i as u32, RockType::Basalt, 5_000.0, 0.0);
        }
        // Latitudinal climate with a lapse rate; ice caps at both poles.
        let t = 28.0 - 55.0 * lat.sin().powi(2) - 0.0065 * e.max(0.0);
        p.temperature_c[i] = t;
        let dry_belt = (lat.abs() - 0.5).abs() < 0.15;
        p.precip_mm_yr[i] = if dry_belt {
            120.0
        } else {
            900.0 * lat.cos() + 150.0
        };
        let (east, north) = mesh.east_north(i as u32);
        p.wind_m_s[i] = (east * lat.cos() + north * 0.2) * 9.0;
        p.sediment_m[i] = 2.0;
    }
    p
}

#[test]
#[ignore = "performance probe"]
fn recent_past_at_level_6_is_under_budget() {
    let mesh = Mesh::build(LEVEL);
    let mut p = busy_planet(&mesh);
    let mut proc = SurfaceProcess::new();
    let mut ledger = MassLedger::default();

    let land = p.elevation_m.iter().filter(|e| **e >= 0.0).count();
    let mut times = Vec::new();
    for _ in 0..100 {
        let t0 = std::time::Instant::now();
        {
            let mut ctx = StepCtx {
                rng: rng_for(p.config.seed, "surface", p.step_index),
                progress: &NullProgress,
                ledger: &mut ledger,
            };
            proc.step(&mut p, &mesh, 0.005, &mut ctx);
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        p.step_index += 1;
        p.time_myr += 0.005;
    }
    let warm = &times[1..];
    let mean = warm.iter().sum::<f64>() / warm.len() as f64;
    let mut sorted = warm.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let max = sorted[sorted.len() - 1];
    let lakes = p.lake_depth_m.iter().filter(|d| **d > 0.0).count();
    let iced = p.ice_thickness_m.iter().filter(|t| **t > 0.0).count();

    // What the strata recorder actually produced, for WP10's inspector.
    let mut new_rock: Vec<(RockType, f64)> = Vec::new();
    for c in 0..p.n_cells() as u32 {
        for s in p.columns.col(c) {
            if s.deposited_myr == 0.0 {
                continue;
            }
            match new_rock.iter_mut().find(|(r, _)| *r == s.rock) {
                Some((_, t)) => *t += s.thickness_m as f64,
                None => new_rock.push((s.rock, s.thickness_m as f64)),
            }
        }
    }
    new_rock.sort_by(|a, b| b.1.total_cmp(&a.1));
    let facies: Vec<String> = new_rock
        .iter()
        .map(|(r, t)| format!("{r:?} {t:.0} m"))
        .collect();
    println!("new strata after 100 steps: {}", facies.join(", "));
    println!(
        "max loose sediment: land {:.1} m, sea {:.1} m",
        (0..p.n_cells())
            .filter(|i| p.elevation_m[*i] >= 0.0)
            .map(|i| p.sediment_m[i])
            .fold(0.0f32, f32::max),
        (0..p.n_cells())
            .filter(|i| p.elevation_m[*i] < 0.0)
            .map(|i| p.sediment_m[i])
            .fold(0.0f32, f32::max)
    );
    println!(
        "level {LEVEL}: {} cells ({land} land, {lakes} lake, {iced} glaciated)\n\
         mean {mean:.1} ms/step, median {median:.1} ms, max {max:.1} ms, \
         400 steps would take {:.1} s",
        mesh.n_cells(),
        mean * 400.0 / 1000.0
    );
    assert!(
        mean < 225.0,
        "mean step {mean:.1} ms exceeds the 225 ms budget"
    );
}

#[test]
#[ignore = "calibration report"]
fn print_calibration() {
    println!("fluvial   K_F                 = {:e}", fluvial::K_F);
    println!(
        "fluvial   channel fraction    = {}",
        fluvial::CHANNEL_FRACTION
    );
    println!("fluvial   K_T (capacity)      = {}", fluvial::K_T);
    println!(
        "hydro     land ET             = {} m/yr",
        hydro::LAND_ET_M_YR
    );
    println!(
        "hydro     lake evaporation    = {} m/yr",
        hydro::LAKE_EVAP_M_YR
    );
    println!(
        "hillslope weathering          = {:e} m/yr",
        hillslope::WEATHERING_M_PER_YR
    );
    println!(
        "hillslope D at level 6 pitch  = {} m^2/yr",
        hillslope::HILLSLOPE_D_M2_YR
    );
    println!(
        "glacial   flow C              = {} /yr",
        glacial::FLOW_C_PER_YR
    );
    println!("glacial   K_glacial           = {:e}", glacial::K_GLACIAL);
    println!();
    for (q, s) in [
        (1.0e12f64, 0.002f64),
        (1.0e11, 0.005),
        (1.0e10, 0.02),
        (1.0e9, 0.05),
    ] {
        let channel = fluvial::K_F as f64 * q.sqrt() * s * 1.0e6;
        let cell = channel * fluvial::CHANNEL_FRACTION as f64;
        println!(
            "Q={q:9.1e} m^3/yr S={s:<6} -> channel {channel:7.0} m/Myr, cell mean {cell:6.0} m/Myr"
        );
    }
    println!();
    for (h, s) in [(3000.0f64, 0.02), (1000.0, 0.01), (300.0, 0.05)] {
        let u = glacial::FLOW_C_PER_YR as f64 * h * s;
        println!(
            "ice H={h:6.0} m S={s:<5} -> sliding {u:6.1} m/yr, abrasion {:5.1} m/Myr",
            glacial::K_GLACIAL as f64 * u * 1.0e6
        );
    }
}
