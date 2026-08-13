//! End-to-end smoke test: `iw-headless gen` against whatever process crates
//! are wired in `src/processes.rs`, checked via the `summary.json` it writes.
//! Kept small (level 4, tiny durations) to stay well under 60s.

use std::process::Command;

#[test]
fn gen_end_to_end() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("planet");

    let status = Command::new(env!("CARGO_BIN_EXE_iw-headless"))
        .args(["gen", "--seed", "42", "--level", "4", "--out"])
        .arg(&out)
        .args(["--durations", "10,10,5,0.05"])
        .status()
        .expect("running iw-headless gen");
    assert!(status.success(), "iw-headless gen exited with {status:?}");

    let summary_path = out.join("summary.json");
    let text = std::fs::read_to_string(&summary_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", summary_path.display()));
    let summary: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");

    let land = summary["land_fraction"]
        .as_f64()
        .expect("land_fraction present");
    assert!(
        (0.0..=1.0).contains(&land),
        "land_fraction {land} out of range"
    );

    let plate_count = summary["plate_count"]
        .as_u64()
        .expect("plate_count present");
    assert!(plate_count >= 1, "plate_count {plate_count} too low");

    let sea_level = summary["sea_level_m"].as_f64();
    assert!(sea_level.is_some(), "sea_level_m present");
}
