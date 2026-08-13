//! Plate bookkeeping on the cell graph: union-find, connected components,
//! multi-source Voronoi, and compaction of the `plates` vector.
//!
//! Every routine iterates cells in index order and neighbours in mesh order,
//! so results never depend on hash or thread ordering.

use std::collections::VecDeque;

use iw_core::{Planet, Plate};
use iw_mesh::Mesh;
use rustc_hash::FxHashMap;

/// Plate id used during CrustalFormation for cells that belong to no craton.
/// It is deliberately out of range of `planet.plates`, which is what makes
/// "has the phase-1 -> drift partition run yet?" a pure function of `Planet`.
pub(crate) const UNASSIGNED: u16 = u16::MAX;

/// Disjoint-set over small id spaces (cratons, plates).
pub(crate) struct UnionFind {
    parent: Vec<u16>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> UnionFind {
        UnionFind {
            parent: (0..n as u16).collect(),
        }
    }

    pub(crate) fn find(&mut self, mut x: u16) -> u16 {
        while self.parent[x as usize] != x {
            let g = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = g;
            x = g;
        }
        x
    }

    /// Merge the sets of `a` and `b`; the smaller id wins so the outcome does
    /// not depend on call order. Returns true when a merge actually happened.
    pub(crate) fn union(&mut self, a: u16, b: u16) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[hi as usize] = lo;
        true
    }
}

/// Label every cell with the index of its connected component within its own
/// plate. `out[c]` is a component id unique across the whole planet.
/// Returns the number of components.
pub(crate) fn plate_components(planet: &Planet, mesh: &Mesh, out: &mut [u32]) -> u32 {
    let n = planet.n_cells();
    out[..n].fill(u32::MAX);
    let mut next = 0u32;
    let mut queue: VecDeque<u32> = VecDeque::new();
    for c in 0..n as u32 {
        if out[c as usize] != u32::MAX {
            continue;
        }
        let plate = planet.plate_id[c as usize];
        let id = next;
        next += 1;
        out[c as usize] = id;
        queue.push_back(c);
        while let Some(x) = queue.pop_front() {
            for &m in mesh.neighbors_of(x) {
                if out[m as usize] == u32::MAX && planet.plate_id[m as usize] == plate {
                    out[m as usize] = id;
                    queue.push_back(m);
                }
            }
        }
    }
    next
}

/// Force every plate to be a single connected region.
///
/// For each plate the largest component is kept. Small strays are absorbed by
/// whichever neighbouring plate they touch most; strays of at least
/// `calve_min` cells become new plates (microplate calving) as long as the
/// plate count stays under `max_plates`.
///
/// Returns true when anything changed.
pub(crate) fn enforce_contiguity(
    planet: &mut Planet,
    mesh: &Mesh,
    scratch: &mut Vec<u32>,
    calve_min: usize,
    max_plates: usize,
) -> bool {
    let n = planet.n_cells();
    scratch.resize(n, u32::MAX);
    let mut changed_any = false;
    // Absorbing one stray can connect another; a couple of sweeps settles it.
    for _ in 0..4 {
        let ncomp = plate_components(planet, mesh, scratch) as usize;
        let mut size = vec![0u32; ncomp];
        let mut plate_of = vec![0u16; ncomp];
        for (c, k) in scratch[..n].iter().enumerate() {
            let k = *k as usize;
            size[k] += 1;
            plate_of[k] = planet.plate_id[c];
        }
        // Largest component per plate (ties: lowest component id).
        let mut best: FxHashMap<u16, usize> = FxHashMap::default();
        for k in 0..ncomp {
            let e = best.entry(plate_of[k]).or_insert(k);
            if size[k] > size[*e] {
                *e = k;
            }
        }
        let mut strays: Vec<usize> = (0..ncomp).filter(|k| best[&plate_of[*k]] != *k).collect();
        if strays.is_empty() {
            return changed_any;
        }
        // Deterministic order: largest first, then by component id.
        strays.sort_by(|a, b| size[*b].cmp(&size[*a]).then(a.cmp(b)));
        let mut target = vec![u16::MAX; ncomp];
        for &k in &strays {
            if size[k] as usize >= calve_min && planet.plates.len() < max_plates {
                let src = plate_of[k] as usize;
                let plate = planet.plates[src].clone();
                target[k] = planet.plates.len() as u16;
                planet.plates.push(Plate {
                    welded_to: None,
                    ..plate
                });
            } else {
                // Neighbouring plate with the most shared edges.
                let mut contact: FxHashMap<u16, u32> = FxHashMap::default();
                for (c, cid) in scratch[..n].iter().enumerate() {
                    if *cid as usize != k {
                        continue;
                    }
                    for &m in mesh.neighbors_of(c as u32) {
                        let p = planet.plate_id[m as usize];
                        if p != plate_of[k] {
                            *contact.entry(p).or_insert(0) += 1;
                        }
                    }
                }
                let mut keys: Vec<u16> = contact.keys().copied().collect();
                keys.sort_unstable();
                let mut pick = u16::MAX;
                let mut pick_n = 0;
                for p in keys {
                    if contact[&p] > pick_n {
                        pick_n = contact[&p];
                        pick = p;
                    }
                }
                target[k] = pick;
            }
        }
        let mut changed = false;
        for (c, k) in scratch[..n].iter().enumerate() {
            let t = target[*k as usize];
            if t != u16::MAX {
                planet.plate_id[c] = t;
                changed = true;
            }
        }
        changed_any |= changed;
        if !changed {
            return changed_any;
        }
    }
    changed_any
}

/// Drop plates with no cells and renumber the survivors, preserving relative
/// order. Returns true when the vector changed.
pub(crate) fn compact_plates(planet: &mut Planet) -> bool {
    let np = planet.plates.len();
    if np == 0 {
        return false;
    }
    let mut used = vec![false; np];
    for &p in &planet.plate_id {
        if (p as usize) < np {
            used[p as usize] = true;
        }
    }
    if used.iter().all(|u| *u) {
        return false;
    }
    let mut remap = vec![u16::MAX; np];
    let mut kept: Vec<Plate> = Vec::with_capacity(np);
    for (i, u) in used.iter().enumerate() {
        if *u {
            remap[i] = kept.len() as u16;
            kept.push(planet.plates[i].clone());
        }
    }
    for p in planet.plate_id.iter_mut() {
        if (*p as usize) < np && remap[*p as usize] != u16::MAX {
            *p = remap[*p as usize];
        }
    }
    // `welded_to` points at plate ids; keep it pointing somewhere valid.
    for plate in kept.iter_mut() {
        plate.welded_to = plate
            .welded_to
            .and_then(|w| remap.get(w as usize).copied())
            .filter(|w| *w != u16::MAX);
    }
    planet.plates = kept;
    true
}

/// Fold plates smaller than `min_cells` into whichever neighbour they share
/// the most edges with. Keeps the mosaic from silting up with slivers thrown
/// off by boundary migration. Returns true when anything changed.
pub(crate) fn absorb_tiny_plates(planet: &mut Planet, mesh: &Mesh, min_cells: usize) -> bool {
    let np = planet.plates.len();
    if np <= 2 {
        return false;
    }
    let mut count = vec![0usize; np];
    for &p in &planet.plate_id {
        if (p as usize) < np {
            count[p as usize] += 1;
        }
    }
    let mut changed = false;
    for p in 0..np {
        if count[p] == 0 || count[p] >= min_cells {
            continue;
        }
        let mut contact: FxHashMap<u16, u32> = FxHashMap::default();
        for c in 0..planet.n_cells() {
            if planet.plate_id[c] as usize != p {
                continue;
            }
            for &m in mesh.neighbors_of(c as u32) {
                let q = planet.plate_id[m as usize];
                if q as usize != p && count[q as usize] > 0 {
                    *contact.entry(q).or_insert(0) += 1;
                }
            }
        }
        let mut keys: Vec<u16> = contact.keys().copied().collect();
        keys.sort_unstable();
        let mut pick = u16::MAX;
        let mut pick_n = 0;
        for q in keys {
            if contact[&q] > pick_n {
                pick_n = contact[&q];
                pick = q;
            }
        }
        if pick == u16::MAX {
            continue;
        }
        for c in planet.plate_id.iter_mut() {
            if *c as usize == p {
                *c = pick;
            }
        }
        count[pick as usize] += count[p];
        count[p] = 0;
        changed = true;
    }
    changed
}

/// Multi-source BFS over the cell graph. `label[c] == u16::MAX` marks a cell
/// still to be claimed; every other cell seeds the frontier with its label.
///
/// Seeds are enqueued in cell index order and neighbours in mesh order, so the
/// resulting regions are a deterministic graph Voronoi diagram.
pub(crate) fn flood_labels(mesh: &Mesh, label: &mut [u16]) {
    let mut queue: VecDeque<u32> = VecDeque::new();
    for (c, l) in label.iter().enumerate() {
        if *l != u16::MAX {
            queue.push_back(c as u32);
        }
    }
    while let Some(c) = queue.pop_front() {
        let l = label[c as usize];
        for &m in mesh.neighbors_of(c) {
            if label[m as usize] == u16::MAX {
                label[m as usize] = l;
                queue.push_back(m);
            }
        }
    }
}

/// Graph distance (in hops) from the set `sources` to every cell.
pub(crate) fn bfs_distance(mesh: &Mesh, sources: &[u32], out: &mut [u32]) {
    out.fill(u32::MAX);
    let mut queue: VecDeque<u32> = VecDeque::new();
    for &s in sources {
        out[s as usize] = 0;
        queue.push_back(s);
    }
    while let Some(c) = queue.pop_front() {
        let d = out[c as usize] + 1;
        for &m in mesh.neighbors_of(c) {
            if out[m as usize] == u32::MAX {
                out[m as usize] = d;
                queue.push_back(m);
            }
        }
    }
}
