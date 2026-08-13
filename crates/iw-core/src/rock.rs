use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Rock taxonomy per DESIGN.md §6. Igneous / sedimentary / metamorphic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RockType {
    // igneous
    Basalt,
    Gabbro,
    Granite,
    Diorite,
    Andesite,
    Rhyolite,
    Tuff,
    // sedimentary
    Sandstone,
    Shale,
    Limestone,
    Conglomerate,
    Evaporite,
    // metamorphic
    Slate,
    Schist,
    Gneiss,
    Marble,
    Quartzite,
    Amphibolite,
}

impl RockType {
    pub const ALL: [RockType; 18] = [
        RockType::Basalt,
        RockType::Gabbro,
        RockType::Granite,
        RockType::Diorite,
        RockType::Andesite,
        RockType::Rhyolite,
        RockType::Tuff,
        RockType::Sandstone,
        RockType::Shale,
        RockType::Limestone,
        RockType::Conglomerate,
        RockType::Evaporite,
        RockType::Slate,
        RockType::Schist,
        RockType::Gneiss,
        RockType::Marble,
        RockType::Quartzite,
        RockType::Amphibolite,
    ];

    pub fn is_sedimentary(self) -> bool {
        matches!(
            self,
            RockType::Sandstone
                | RockType::Shale
                | RockType::Limestone
                | RockType::Conglomerate
                | RockType::Evaporite
        )
    }

    pub fn is_metamorphic(self) -> bool {
        matches!(
            self,
            RockType::Slate
                | RockType::Schist
                | RockType::Gneiss
                | RockType::Marble
                | RockType::Quartzite
                | RockType::Amphibolite
        )
    }

    pub fn is_igneous(self) -> bool {
        !self.is_sedimentary() && !self.is_metamorphic()
    }

    /// Bulk density used for isostasy and mass accounting.
    pub fn density_kg_m3(self) -> f32 {
        match self {
            RockType::Basalt => 2900.0,
            RockType::Gabbro => 3000.0,
            RockType::Granite => 2700.0,
            RockType::Diorite => 2800.0,
            RockType::Andesite => 2650.0,
            RockType::Rhyolite => 2500.0,
            RockType::Tuff => 2200.0,
            RockType::Sandstone => 2350.0,
            RockType::Shale => 2400.0,
            RockType::Limestone => 2550.0,
            RockType::Conglomerate => 2450.0,
            RockType::Evaporite => 2200.0,
            RockType::Slate => 2700.0,
            RockType::Schist => 2750.0,
            RockType::Gneiss => 2750.0,
            RockType::Marble => 2700.0,
            RockType::Quartzite => 2650.0,
            RockType::Amphibolite => 3000.0,
        }
    }

    /// Relative erodibility multiplier (1.0 = average). Higher erodes faster.
    pub fn erodibility(self) -> f32 {
        match self {
            RockType::Quartzite => 0.3,
            RockType::Gneiss | RockType::Granite | RockType::Gabbro => 0.5,
            RockType::Amphibolite | RockType::Diorite => 0.55,
            RockType::Basalt | RockType::Schist | RockType::Marble => 0.7,
            RockType::Andesite | RockType::Rhyolite | RockType::Slate => 0.8,
            RockType::Limestone | RockType::Sandstone | RockType::Conglomerate => 1.2,
            RockType::Shale => 1.5,
            RockType::Tuff => 1.8,
            RockType::Evaporite => 2.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum MetamorphicGrade {
    None,
    Low,
    Medium,
    High,
}

/// One layer in a cell's stratigraphic column.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stratum {
    pub rock: RockType,
    pub thickness_m: f32,
    /// Simulation time at deposition/emplacement (elapsed Myr).
    pub deposited_myr: f32,
    pub grade: MetamorphicGrade,
}

pub const MAX_STRATA: usize = 64;

type Column = SmallVec<[Stratum; 8]>;

/// Per-cell stratigraphic columns, bottom (index 0) to top.
///
/// All deposition/erosion in the simulation flows through these operations so
/// the record stays consistent and mass-accountable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrataColumns {
    cols: Vec<Column>,
}

impl StrataColumns {
    pub fn new(n_cells: usize) -> Self {
        StrataColumns {
            cols: vec![Column::new(); n_cells],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cols.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }

    /// Bottom-to-top strata of a cell.
    #[inline]
    pub fn col(&self, cell: u32) -> &[Stratum] {
        &self.cols[cell as usize]
    }

    /// Mutable access for in-place transformation (metamorphism, folding).
    /// Callers must preserve non-negative thicknesses.
    #[inline]
    pub fn col_mut(&mut self, cell: u32) -> &mut SmallVec<[Stratum; 8]> {
        &mut self.cols[cell as usize]
    }

    pub fn total_thickness_m(&self, cell: u32) -> f32 {
        self.cols[cell as usize].iter().map(|s| s.thickness_m).sum()
    }

    pub fn top_rock(&self, cell: u32) -> Option<RockType> {
        self.cols[cell as usize].last().map(|s| s.rock)
    }

    /// Add material on top. Merges into the top stratum when rock and grade
    /// match and ages are within 5 Myr; enforces the MAX_STRATA cap by merging
    /// the thinnest adjacent pair.
    pub fn deposit(&mut self, cell: u32, rock: RockType, thickness_m: f32, time_myr: f64) {
        if thickness_m <= 0.0 {
            return;
        }
        let col = &mut self.cols[cell as usize];
        if let Some(top) = col.last_mut() {
            if top.rock == rock
                && top.grade == MetamorphicGrade::None
                && (time_myr as f32 - top.deposited_myr).abs() < 5.0
            {
                top.thickness_m += thickness_m;
                return;
            }
        }
        col.push(Stratum {
            rock,
            thickness_m,
            deposited_myr: time_myr as f32,
            grade: MetamorphicGrade::None,
        });
        if col.len() > MAX_STRATA {
            merge_thinnest_adjacent(col);
        }
    }

    /// Remove up to `depth_m` from the top. Returns the thickness actually
    /// removed (may be less if the column is exhausted).
    pub fn erode(&mut self, cell: u32, depth_m: f32) -> f32 {
        let col = &mut self.cols[cell as usize];
        let mut remaining = depth_m.max(0.0);
        let mut removed = 0.0;
        while remaining > 0.0 {
            let Some(top) = col.last_mut() else { break };
            if top.thickness_m > remaining {
                top.thickness_m -= remaining;
                removed += remaining;
                remaining = 0.0;
            } else {
                removed += top.thickness_m;
                remaining -= top.thickness_m;
                col.pop();
            }
        }
        removed
    }

    /// Insert an intrusion at `depth_from_top_m` below the surface, splitting
    /// the host stratum if needed.
    pub fn intrude(&mut self, cell: u32, depth_from_top_m: f32, stratum: Stratum) {
        let col = &mut self.cols[cell as usize];
        let mut depth = depth_from_top_m.max(0.0);
        let mut idx = col.len();
        while idx > 0 {
            let t = col[idx - 1].thickness_m;
            if depth < t {
                break;
            }
            depth -= t;
            idx -= 1;
        }
        if idx == 0 {
            col.insert(0, stratum);
        } else if depth <= 0.0 {
            col.insert(idx, stratum);
        } else {
            // Split host stratum at `depth` below its top.
            let host = col[idx - 1];
            let upper = Stratum {
                thickness_m: depth,
                ..host
            };
            col[idx - 1].thickness_m = (host.thickness_m - depth).max(0.0);
            col.insert(idx, stratum);
            col.insert(idx + 1, upper);
        }
        while col.len() > MAX_STRATA {
            merge_thinnest_adjacent(col);
        }
    }

    /// Wipe a cell's record entirely (subduction destroys the column).
    /// Returns the destroyed thickness for mass accounting.
    pub fn clear(&mut self, cell: u32) -> f32 {
        let t = self.total_thickness_m(cell);
        self.cols[cell as usize].clear();
        t
    }

    /// Replace `cell`'s column with a copy of `source_cell`'s column from
    /// another `StrataColumns`. Transport, not creation/destruction: callers
    /// account for the ledger themselves (the drift-v2 remap moves columns
    /// wholesale and charges nothing for pure moves).
    pub fn copy_col_from(&mut self, cell: u32, source: &StrataColumns, source_cell: u32) {
        let dst = &mut self.cols[cell as usize];
        dst.clear();
        dst.extend_from_slice(source.col(source_cell));
    }

    /// Underthrust `source_cell`'s strata beneath `cell`'s existing column
    /// (continental collision: the losing plate's record slides under the
    /// winner's, Himalaya-style). Transport, not destruction — no ledger
    /// charge; the cap is enforced by merging.
    pub fn prepend_col_from(&mut self, cell: u32, source: &StrataColumns, source_cell: u32) {
        let src = source.col(source_cell);
        if src.is_empty() {
            return;
        }
        let dst = &mut self.cols[cell as usize];
        dst.insert_from_slice(0, src);
        while dst.len() > MAX_STRATA {
            merge_thinnest_adjacent(dst);
        }
    }
}

fn merge_thinnest_adjacent(col: &mut Column) {
    if col.len() < 2 {
        return;
    }
    let mut best = 0;
    let mut best_sum = f32::INFINITY;
    for i in 0..col.len() - 1 {
        let sum = col[i].thickness_m + col[i + 1].thickness_m;
        if sum < best_sum {
            best_sum = sum;
            best = i;
        }
    }
    // Keep the identity of the thicker layer, sum the thicknesses.
    let (a, b) = (col[best], col[best + 1]);
    let keep = if a.thickness_m >= b.thickness_m { a } else { b };
    col[best] = Stratum {
        thickness_m: a.thickness_m + b.thickness_m,
        ..keep
    };
    col.remove(best + 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deposit_erode_roundtrip() {
        let mut s = StrataColumns::new(4);
        s.deposit(0, RockType::Basalt, 1000.0, 0.0);
        s.deposit(0, RockType::Shale, 200.0, 10.0);
        assert_eq!(s.top_rock(0), Some(RockType::Shale));
        assert!((s.total_thickness_m(0) - 1200.0).abs() < 1e-3);
        let removed = s.erode(0, 300.0);
        assert!((removed - 300.0).abs() < 1e-3);
        assert_eq!(s.top_rock(0), Some(RockType::Basalt));
        assert!((s.total_thickness_m(0) - 900.0).abs() < 1e-3);
        // over-erosion is clamped
        let removed = s.erode(0, 5000.0);
        assert!((removed - 900.0).abs() < 1e-3);
        assert_eq!(s.top_rock(0), None);
    }

    #[test]
    fn same_rock_merges() {
        let mut s = StrataColumns::new(1);
        s.deposit(0, RockType::Sandstone, 10.0, 0.0);
        s.deposit(0, RockType::Sandstone, 10.0, 1.0);
        assert_eq!(s.col(0).len(), 1);
        // far apart in time -> separate strata
        s.deposit(0, RockType::Sandstone, 10.0, 100.0);
        assert_eq!(s.col(0).len(), 2);
    }

    #[test]
    fn cap_enforced() {
        let mut s = StrataColumns::new(1);
        for i in 0..200 {
            let rock = if i % 2 == 0 {
                RockType::Shale
            } else {
                RockType::Sandstone
            };
            s.deposit(0, rock, 10.0, i as f64 * 20.0);
        }
        assert!(s.col(0).len() <= MAX_STRATA);
        assert!((s.total_thickness_m(0) - 2000.0).abs() < 1e-2);
    }

    #[test]
    fn intrusion_splits_host() {
        let mut s = StrataColumns::new(1);
        s.deposit(0, RockType::Limestone, 1000.0, 0.0);
        s.intrude(
            0,
            400.0,
            Stratum {
                rock: RockType::Granite,
                thickness_m: 50.0,
                deposited_myr: 5.0,
                grade: MetamorphicGrade::None,
            },
        );
        let col = s.col(0);
        assert_eq!(col.len(), 3);
        assert_eq!(col[1].rock, RockType::Granite);
        assert!((s.total_thickness_m(0) - 1050.0).abs() < 1e-3);
        assert!((col[2].thickness_m - 400.0).abs() < 1e-3);
    }
}
