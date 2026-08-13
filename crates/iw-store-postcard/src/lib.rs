//! Postcard + zstd persistence for Infinite Worlds (DESIGN.md §12).
//!
//! Two independent stores live here:
//!
//! - [`FileStore`] implements [`iw_core::CheckpointStore`]: full-fidelity,
//!   resumable [`Planet`] checkpoints, one file per tag.
//! - [`HistoryStore`] is a separate, much lighter ring of per-cell snapshots
//!   for the UI's time scrubber, capped to a disk budget
//!   (`PlanetConfig::history_cap_bytes`).
//!
//! Both use the same on-disk shape: a small uncompressed header (magic +
//! version, so corruption and format drift are cheap to detect) followed by
//! zstd-compressed `postcard` bytes.

#![warn(missing_docs)]

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use iw_core::{Biome, CheckpointStore, Phase, Planet, PlanetView};
use serde::{Deserialize, Serialize};

/// 8-byte magic at the start of every `.iwp` checkpoint file.
const PLANET_MAGIC: &[u8; 8] = b"IWPLANET";
/// Checkpoint format version. Bump when the on-disk layout changes
/// incompatibly; [`FileStore::load`] rejects mismatches with a clear error.
const PLANET_FORMAT_VERSION: u32 = 5;
/// `magic (8) + format version (4) + flags (4)`.
const PLANET_HEADER_LEN: usize = 16;
/// zstd compression level for checkpoints and history snapshots. Level 3 is
/// the library default: fast, and plenty of ratio for structured planet data.
const ZSTD_LEVEL: i32 = 3;

/// Full-fidelity checkpoint store: one `<tag>.iwp` file per saved tag in a
/// directory, atomically written.
pub struct FileStore {
    dir: PathBuf,
}

impl FileStore {
    /// Open (creating if needed) a checkpoint directory.
    pub fn new(dir: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating checkpoint dir {}", dir.display()))?;
        Ok(FileStore { dir })
    }

    fn path_for(&self, tag: &str) -> PathBuf {
        self.dir.join(format!("{tag}.iwp"))
    }
}

impl CheckpointStore for FileStore {
    fn save(&self, tag: &str, planet: &Planet) -> anyhow::Result<()> {
        let body = postcard::to_allocvec(planet).context("serializing planet")?;
        let compressed =
            zstd::stream::encode_all(&body[..], ZSTD_LEVEL).context("compressing planet")?;

        let mut buf = Vec::with_capacity(PLANET_HEADER_LEN + compressed.len());
        buf.extend_from_slice(PLANET_MAGIC);
        buf.extend_from_slice(&PLANET_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags, reserved
        buf.extend_from_slice(&compressed);

        let final_path = self.path_for(tag);
        let tmp_path = self.dir.join(format!("{tag}.iwp.tmp"));
        std::fs::write(&tmp_path, &buf)
            .with_context(|| format!("writing {}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "renaming {} -> {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        Ok(())
    }

    fn load(&self, tag: &str) -> anyhow::Result<Planet> {
        let path = self.path_for(tag);
        let buf = std::fs::read(&path)
            .with_context(|| format!("reading checkpoint {}", path.display()))?;
        if buf.len() < PLANET_HEADER_LEN {
            bail!(
                "checkpoint {} is truncated: {} bytes, need at least {}",
                path.display(),
                buf.len(),
                PLANET_HEADER_LEN
            );
        }
        if &buf[0..8] != PLANET_MAGIC {
            bail!(
                "checkpoint {} has bad magic {:?}, expected {:?}",
                path.display(),
                &buf[0..8],
                PLANET_MAGIC
            );
        }
        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        if version != PLANET_FORMAT_VERSION {
            bail!(
                "checkpoint {} has format version {}, this build supports {}",
                path.display(),
                version,
                PLANET_FORMAT_VERSION
            );
        }
        let compressed = &buf[PLANET_HEADER_LEN..];
        let decompressed = zstd::stream::decode_all(compressed)
            .with_context(|| format!("decompressing checkpoint {}", path.display()))?;
        let planet: Planet = postcard::from_bytes(&decompressed)
            .with_context(|| format!("deserializing checkpoint {}", path.display()))?;
        Ok(planet)
    }

    fn list(&self) -> anyhow::Result<Vec<String>> {
        let mut entries: Vec<(std::time::SystemTime, String)> = Vec::new();
        for entry in std::fs::read_dir(&self.dir)
            .with_context(|| format!("listing {}", self.dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("iwp") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let modified = entry.metadata()?.modified()?;
            entries.push((modified, stem.to_string()));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        Ok(entries.into_iter().map(|(_, tag)| tag).collect())
    }
}

/// A thin per-cell snapshot for the time scrubber: enough to redraw the
/// planet's silhouette without the cost (or size) of a full checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistorySnapshot {
    /// Snapshot version, matching [`iw_core::PlanetView::version`].
    pub version: u64,
    /// Elapsed simulation time, Myr.
    pub time_myr: f64,
    /// Sea level at capture time, meters.
    pub sea_level_m: f32,
    /// Phase at capture time.
    pub phase: Phase,
    /// Per-cell surface elevation, meters.
    pub elevation_m: Vec<f32>,
    /// Per-cell biome classification.
    pub biome: Vec<Biome>,
    /// Per-cell owning plate id.
    pub plate_id: Vec<u16>,
    /// Per-cell ice thickness, meters.
    pub ice_thickness_m: Vec<f32>,
}

/// 8-byte magic at the start of every `.iwh` history file.
const HIST_MAGIC: &[u8; 8] = b"IWHIST\0\0";
/// `magic (8) + version (8) + time_myr (8) + phase (1)`, uncompressed so
/// [`HistoryStore::list`] can read it without touching the zstd body.
const HIST_HEADER_LEN: usize = 25;

/// Disk-capped ring of [`HistorySnapshot`]s, one `<version>.iwh` file per
/// push, under `<dir>/history/`.
///
/// Not a [`CheckpointStore`]: history snapshots are not resumable state, just
/// a coarse record for scrubbing back through a run.
pub struct HistoryStore {
    dir: PathBuf,
    cap_bytes: u64,
}

impl HistoryStore {
    /// Open (creating if needed) `<dir>/history`, capped to `cap_bytes` total
    /// on disk (typically [`iw_core::PlanetConfig::history_cap_bytes`]).
    pub fn new(dir: PathBuf, cap_bytes: u64) -> anyhow::Result<Self> {
        let dir = dir.join("history");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating history dir {}", dir.display()))?;
        Ok(HistoryStore { dir, cap_bytes })
    }

    fn path_for(&self, version: u64) -> PathBuf {
        self.dir.join(format!("{version}.iwh"))
    }

    /// Append a snapshot of `view`, then evict the oldest entries until total
    /// history size is back under the configured cap.
    pub fn push(&self, view: &PlanetView) -> anyhow::Result<()> {
        let snap = HistorySnapshot {
            version: view.version,
            time_myr: view.time_myr,
            sea_level_m: view.sea_level_m,
            phase: view.phase,
            elevation_m: view.cells.elevation_m.clone(),
            biome: view.cells.biome.clone(),
            plate_id: view.cells.plate_id.clone(),
            ice_thickness_m: view.cells.ice_thickness_m.clone(),
        };
        self.write_snapshot(&snap)?;
        self.enforce_cap()?;
        Ok(())
    }

    fn write_snapshot(&self, snap: &HistorySnapshot) -> anyhow::Result<()> {
        let body = postcard::to_allocvec(snap).context("serializing history snapshot")?;
        let compressed = zstd::stream::encode_all(&body[..], ZSTD_LEVEL)
            .context("compressing history snapshot")?;

        let mut buf = Vec::with_capacity(HIST_HEADER_LEN + compressed.len());
        buf.extend_from_slice(HIST_MAGIC);
        buf.extend_from_slice(&snap.version.to_le_bytes());
        buf.extend_from_slice(&snap.time_myr.to_le_bytes());
        buf.push(snap.phase.index() as u8);
        buf.extend_from_slice(&compressed);

        let final_path = self.path_for(snap.version);
        let tmp_path = self.dir.join(format!("{}.iwh.tmp", snap.version));
        std::fs::write(&tmp_path, &buf)
            .with_context(|| format!("writing {}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "renaming {} -> {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        Ok(())
    }

    /// Delete the oldest (lowest-version) snapshots until the directory's
    /// total size is at or under `cap_bytes`. Always leaves at least one
    /// snapshot, even if that single file exceeds the cap on its own.
    fn enforce_cap(&self) -> anyhow::Result<()> {
        let mut files = self.history_files()?;
        // Oldest (lowest version) first, so we evict from the front.
        files.sort_by_key(|f| f.version);
        let mut total: u64 = files.iter().map(|f| f.size_bytes).sum();
        let mut i = 0;
        while total > self.cap_bytes && files.len() - i > 1 {
            let f = &files[i];
            std::fs::remove_file(&f.path)
                .with_context(|| format!("evicting {}", f.path.display()))?;
            total = total.saturating_sub(f.size_bytes);
            i += 1;
        }
        Ok(())
    }

    /// All `.iwh` files currently on disk, unordered.
    fn history_files(&self) -> anyhow::Result<Vec<HistFile>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)
            .with_context(|| format!("listing {}", self.dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("iwh") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Ok(version) = stem.parse::<u64>() else {
                continue;
            };
            let size_bytes = entry.metadata()?.len();
            out.push(HistFile {
                path,
                version,
                size_bytes,
            });
        }
        Ok(out)
    }

    /// Snapshot `(version, time_myr)` pairs available, sorted oldest first.
    /// Reads only the small uncompressed header of each file.
    pub fn list(&self) -> anyhow::Result<Vec<(u64, f64)>> {
        let mut files = self.history_files()?;
        files.sort_by_key(|f| f.version);
        let mut out = Vec::with_capacity(files.len());
        for f in files {
            let header = read_header(&f.path)?;
            out.push((header.0, header.1));
        }
        Ok(out)
    }

    /// Load a specific snapshot by version.
    pub fn load(&self, version: u64) -> anyhow::Result<HistorySnapshot> {
        let path = self.path_for(version);
        let buf = std::fs::read(&path)
            .with_context(|| format!("reading history snapshot {}", path.display()))?;
        validate_header(&path, &buf)?;
        let compressed = &buf[HIST_HEADER_LEN..];
        let decompressed = zstd::stream::decode_all(compressed)
            .with_context(|| format!("decompressing history snapshot {}", path.display()))?;
        let snap: HistorySnapshot = postcard::from_bytes(&decompressed)
            .with_context(|| format!("deserializing history snapshot {}", path.display()))?;
        Ok(snap)
    }
}

/// Bookkeeping for one `.iwh` file on disk.
struct HistFile {
    path: PathBuf,
    version: u64,
    size_bytes: u64,
}

fn validate_header(path: &Path, buf: &[u8]) -> anyhow::Result<()> {
    if buf.len() < HIST_HEADER_LEN {
        bail!(
            "history snapshot {} is truncated: {} bytes, need at least {}",
            path.display(),
            buf.len(),
            HIST_HEADER_LEN
        );
    }
    if &buf[0..8] != HIST_MAGIC {
        bail!(
            "history snapshot {} has bad magic {:?}, expected {:?}",
            path.display(),
            &buf[0..8],
            HIST_MAGIC
        );
    }
    Ok(())
}

/// Read `(version, time_myr)` from a history file's uncompressed header only.
fn read_header(path: &Path) -> anyhow::Result<(u64, f64)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut header = [0u8; HIST_HEADER_LEN];
    f.read_exact(&mut header)
        .with_context(|| format!("reading header of {}", path.display()))?;
    validate_header(path, &header)?;
    let version = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let time_myr = f64::from_le_bytes(header[16..24].try_into().unwrap());
    Ok((version, time_myr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use iw_core::{CrustType, PlanetConfig, RockType, ViewCells};
    use std::sync::Arc;

    fn small_planet() -> Planet {
        let config = PlanetConfig {
            subdivision_level: 4,
            ..PlanetConfig::default()
        };
        let mut planet = Planet::new(config, 6);
        planet.elevation_m[0] = 123.5;
        planet.elevation_m[3] = -456.25;
        planet.plate_id[2] = 7;
        planet.sea_level_m = 12.0;
        planet.time_myr = 42.5;
        planet.phase = Phase::Drift;
        planet.columns.deposit(0, RockType::Basalt, 1000.0, 0.0);
        planet.columns.deposit(0, RockType::Shale, 200.0, 10.0);
        planet
    }

    #[test]
    fn round_trip_planet() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().to_path_buf()).unwrap();
        let planet = small_planet();
        store.save("test", &planet).unwrap();
        let loaded = store.load("test").unwrap();
        assert_eq!(loaded.elevation_m, planet.elevation_m);
        assert_eq!(loaded.plate_id, planet.plate_id);
        assert_eq!(loaded.sea_level_m, planet.sea_level_m);
        assert_eq!(loaded.time_myr, planet.time_myr);
        assert_eq!(loaded.phase, planet.phase);
        assert_eq!(loaded.columns.col(0), planet.columns.col(0));
    }

    #[test]
    fn corrupted_magic_errors() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().to_path_buf()).unwrap();
        store.save("test", &small_planet()).unwrap();
        let path = dir.path().join("test.iwp");
        let mut buf = std::fs::read(&path).unwrap();
        buf[0] = b'X';
        std::fs::write(&path, &buf).unwrap();
        assert!(store.load("test").is_err());
    }

    #[test]
    fn truncated_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().to_path_buf()).unwrap();
        store.save("test", &small_planet()).unwrap();
        let path = dir.path().join("test.iwp");
        std::fs::write(&path, b"short").unwrap();
        assert!(store.load("test").is_err());
    }

    #[test]
    fn version_mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().to_path_buf()).unwrap();
        store.save("test", &small_planet()).unwrap();
        let path = dir.path().join("test.iwp");
        let mut buf = std::fs::read(&path).unwrap();
        buf[8..12].copy_from_slice(&99u32.to_le_bytes());
        std::fs::write(&path, &buf).unwrap();
        assert!(store.load("test").is_err());
    }

    #[test]
    fn save_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().to_path_buf()).unwrap();
        store.save("test", &small_planet()).unwrap();
        let tmp = dir.path().join("test.iwp.tmp");
        assert!(!tmp.exists());
        assert!(dir.path().join("test.iwp").exists());
    }

    #[test]
    fn list_sorted_by_mtime_then_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path().to_path_buf()).unwrap();
        store.save("b", &small_planet()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        store.save("a", &small_planet()).unwrap();
        assert_eq!(
            store.list().unwrap(),
            vec!["b".to_string(), "a".to_string()]
        );
    }

    fn fake_view(version: u64, time_myr: f64, n: usize) -> PlanetView {
        PlanetView {
            version,
            phase: Phase::Drift,
            time_myr,
            sea_level_m: 0.0,
            cells: Arc::new(ViewCells {
                elevation_m: vec![0.0; n],
                sediment_m: vec![0.0; n],
                biome: vec![Biome::Ocean; n],
                plate_id: vec![0; n],
                crust_type: vec![CrustType::Oceanic; n],
                crust_age_myr: vec![0.0; n],
                crust_thickness_m: vec![0.0; n],
                temperature_c: vec![0.0; n],
                precip_mm_yr: vec![0.0; n],
                ice_thickness_m: vec![0.0; n],
                water_flux_m3_yr: vec![0.0; n],
                flow_to: vec![u32::MAX; n],
                lake_depth_m: vec![0.0; n],
                top_rock: vec![None; n],
                plate_velocity_m_yr: vec![Vec3::ZERO; n],
                tectonic_flags: vec![0; n],
            }),
        }
    }

    #[test]
    fn history_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let hs = HistoryStore::new(dir.path().to_path_buf(), u64::MAX).unwrap();
        let view = fake_view(1, 3.5, 8);
        hs.push(&view).unwrap();
        let loaded = hs.load(1).unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.time_myr, 3.5);
        assert_eq!(loaded.elevation_m.len(), 8);
    }

    #[test]
    fn history_cap_enforced() {
        let n = 2000;
        // Probe a single snapshot's on-disk size, then size the cap for ~5.
        let probe_dir = tempfile::tempdir().unwrap();
        let probe = HistoryStore::new(probe_dir.path().to_path_buf(), u64::MAX).unwrap();
        probe.push(&fake_view(0, 0.0, n)).unwrap();
        let one_size = std::fs::metadata(probe_dir.path().join("history/0.iwh"))
            .unwrap()
            .len();
        let cap = one_size * 5;

        let dir = tempfile::tempdir().unwrap();
        let hs = HistoryStore::new(dir.path().to_path_buf(), cap).unwrap();
        for v in 0..20u64 {
            hs.push(&fake_view(v, v as f64, n)).unwrap();
        }
        let list = hs.list().unwrap();
        for w in list.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        assert!(!list.iter().any(|(v, _)| *v == 0));
        assert!(list.iter().any(|(v, _)| *v == 19));
        assert!(
            list.len() <= 8,
            "expected roughly 5 kept, got {}",
            list.len()
        );
    }
}
