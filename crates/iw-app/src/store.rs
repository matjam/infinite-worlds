//! Persistence wiring for the app: a checkpoint store whose directory can be
//! swapped at runtime, and a background writer for history snapshots.
//!
//! The simulation is handed one `Arc<dyn CheckpointStore>` when it is spawned
//! and keeps it for the lifetime of the worker, but regenerating with a new
//! seed or level should write to a new planet directory. [`SwitchableStore`]
//! provides that indirection.
//!
//! History snapshots are written on their own thread: at level 8 a snapshot is
//! several MB before compression, and the UI thread must not pay for that.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::RwLock;
use std::thread::JoinHandle;

use iw_core::{CheckpointStore, Planet, PlanetView};
use iw_store_postcard::{FileStore, HistoryStore};

/// Default planet directory for a seed/level pair: `<root>/<seed>-L<level>`.
pub fn planet_dir(root: &Path, seed: u64, level: u8) -> PathBuf {
    root.join(format!("{seed}-L{level}"))
}

/// A [`CheckpointStore`] that can be pointed at a different directory without
/// disturbing the simulation holding it.
pub struct SwitchableStore {
    inner: RwLock<FileStore>,
    dir: RwLock<PathBuf>,
}

impl SwitchableStore {
    /// Open (creating if needed) `dir`.
    pub fn new(dir: PathBuf) -> anyhow::Result<SwitchableStore> {
        let store = FileStore::new(dir.clone())?;
        Ok(SwitchableStore {
            inner: RwLock::new(store),
            dir: RwLock::new(dir),
        })
    }

    /// Point subsequent saves and loads at `dir`. Existing files are untouched.
    pub fn retarget(&self, dir: PathBuf) -> anyhow::Result<()> {
        let store = FileStore::new(dir.clone())?;
        *self.inner.write().unwrap() = store;
        *self.dir.write().unwrap() = dir;
        Ok(())
    }

    /// The directory currently being written to.
    pub fn dir(&self) -> PathBuf {
        self.dir.read().unwrap().clone()
    }
}

impl CheckpointStore for SwitchableStore {
    fn save(&self, tag: &str, planet: &Planet) -> anyhow::Result<()> {
        self.inner.read().unwrap().save(tag, planet)
    }

    fn load(&self, tag: &str) -> anyhow::Result<Planet> {
        self.inner.read().unwrap().load(tag)
    }

    fn list(&self) -> anyhow::Result<Vec<String>> {
        self.inner.read().unwrap().list()
    }
}

enum HistoryMsg {
    Push(Box<PlanetView>),
    Retarget(PathBuf, u64),
    Shutdown,
}

/// Background writer for history snapshots.
///
/// Dropping it (or calling [`HistoryWriter::shutdown`]) drains the queue and
/// joins the thread, so no snapshot is lost at exit.
pub struct HistoryWriter {
    tx: Option<Sender<HistoryMsg>>,
    join: Option<JoinHandle<()>>,
}

impl HistoryWriter {
    /// Start the writer against `dir` with a `cap_bytes` disk budget.
    pub fn new(dir: PathBuf, cap_bytes: u64) -> HistoryWriter {
        let (tx, rx) = mpsc::channel::<HistoryMsg>();
        let join = std::thread::Builder::new()
            .name("iw-history".to_string())
            .spawn(move || {
                let mut store = HistoryStore::new(dir, cap_bytes).ok();
                while let Ok(msg) = rx.recv() {
                    match msg {
                        HistoryMsg::Push(view) => {
                            if let Some(store) = store.as_ref() {
                                if let Err(e) = store.push(&view) {
                                    log::warn!("history snapshot {}: {e:#}", view.version);
                                }
                            }
                        }
                        HistoryMsg::Retarget(dir, cap) => match HistoryStore::new(dir, cap) {
                            Ok(s) => store = Some(s),
                            Err(e) => {
                                log::error!("history store: {e:#}");
                                store = None;
                            }
                        },
                        HistoryMsg::Shutdown => break,
                    }
                }
            })
            .expect("spawning the history writer thread");
        HistoryWriter {
            tx: Some(tx),
            join: Some(join),
        }
    }

    /// Queue a snapshot of `view`.
    pub fn push(&self, view: &PlanetView) {
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.send(HistoryMsg::Push(Box::new(view.clone())));
        }
    }

    /// Write subsequent snapshots to `dir`.
    pub fn retarget(&self, dir: PathBuf, cap_bytes: u64) {
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.send(HistoryMsg::Retarget(dir, cap_bytes));
        }
    }

    /// Stop the writer and wait for it.
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(HistoryMsg::Shutdown);
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for HistoryWriter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iw_core::PlanetConfig;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("iw-app-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn planet_dirs_are_named_by_seed_and_level() {
        let d = planet_dir(Path::new("/tmp/planets"), 42, 6);
        assert!(d.ends_with("42-L6"));
    }

    #[test]
    fn retargeting_moves_where_checkpoints_land() {
        let root = temp_dir("store");
        let a = root.join("a");
        let b = root.join("b");
        let store = SwitchableStore::new(a.clone()).unwrap();
        assert_eq!(store.dir(), a);
        let planet = Planet::new(PlanetConfig::default(), 4);
        store.save("phase-drift", &planet).unwrap();
        assert_eq!(store.list().unwrap(), vec!["phase-drift".to_string()]);

        store.retarget(b.clone()).unwrap();
        assert_eq!(store.dir(), b);
        assert!(
            store.list().unwrap().is_empty(),
            "a fresh directory starts empty"
        );
        store.save("phase-drift", &planet).unwrap();
        assert!(store.load("phase-drift").is_ok());
        // The old directory is left alone.
        let back = SwitchableStore::new(a).unwrap();
        assert_eq!(back.list().unwrap(), vec!["phase-drift".to_string()]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn history_writer_drains_before_it_joins() {
        let root = temp_dir("history");
        let config = PlanetConfig::default();
        let planet = Planet::new(config, 4);
        let mesh = iw_sim::test_util::tiny_mesh();
        let view = PlanetView::capture(9, &planet, &mesh);
        let mut writer = HistoryWriter::new(root.clone(), 64 * 1024 * 1024);
        writer.push(&view);
        writer.shutdown();

        let store = HistoryStore::new(root.clone(), 64 * 1024 * 1024).unwrap();
        let list = store.list().unwrap();
        assert_eq!(list.len(), 1, "the queued snapshot was written at shutdown");
        assert_eq!(list[0].0, 9);
        let _ = std::fs::remove_dir_all(&root);
    }
}
