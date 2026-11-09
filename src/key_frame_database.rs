/*
The data model differs from ORB-SLAM3's C++ implementation in
a few deliberate ways:

- KeyFrames are addressed by id, not by pointer. This keeps the database
trivially serializable and free of lock cycles.

- Per-query scratch state is local. This uses per-call HashMap instead,
which removes bookkeeping and makes concurrent detection safe.

- One RwLock over the whole inverted file.

- Vec instead of linked list. Erase is still linear in bucket length
but iteration and insertion are dramatically cheaper.
*/

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::orb_vocabulary::{BowVector, OrbVocabulary};

/// Stable identifier for a KeyFrame
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct KeyFrameId(pub u64);

/// Stable identifier for a map within the atlas
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MapId(pub u64);

pub trait KeyFrameView {
    /// Cached bag-of-words vector for `kf`, or `None` if the id is unknown.
    fn bow_vec(&self, kf: KeyFrameId) -> Option<&BowVector>;

    /// Map that owns `kf`, or `None` if the id is unknown.
    fn map_id(&self, kf: KeyFrameId) -> Option<MapId>;

    /// Whether the KeyFrame has been flagged as bad (culled).
    fn is_bad(&self, kf: KeyFrameId) -> bool;

    /// Whether the map has been flagged as bad. Used to suppress merge
    /// candidates from dead maps.
    fn is_map_bad(&self, map: MapId) -> bool;

    /// Up to `n` best covisibility neighbours of `kf` (highest weights
    /// first). May return fewer if the KeyFrame has fewer neighbours.
    fn best_covisibility(&self, kf: KeyFrameId, n: usize) -> Vec<KeyFrameId>;

    /// Set of KeyFrames connected to `kf` in the covisibility graph; these
    /// are excluded as candidates.
    fn connected_keyframes(&self, kf: KeyFrameId) -> HashSet<KeyFrameId>;
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Candidates {
    pub loop_: Vec<KeyFrameId>,
    pub merge: Vec<KeyFrameId>,
}

#[derive(Debug)]
pub struct KeyFrameDatabase {
    voc: Arc<OrbVocabulary>,
    inverted: RwLock<Vec<Vec<KeyFrameId>>>,
}

impl KeyFrameDatabase {
    pub fn new(voc: Arc<OrbVocabulary>) -> Self {
        let n_words = voc.size();
        Self {
            voc,
            inverted: RwLock::new(vec![Vec::new(); n_words]),
        }
    }

    /// Vocabulary used to score candidates.
    pub fn vocabulary(&self) -> &Arc<OrbVocabulary> {
        &self.voc
    }

    /// Register `kf` under each word it contains. The caller passes the BoW
    /// rather than us pulling it from a `KeyFrameView` so this works during
    /// KeyFrame construction, before the KeyFrame is published.
    pub fn add(&self, kf: KeyFrameId, bow: &BowVector) {
        let mut inv = self.inverted.write().expect("poisoned");
        for &word in bow.0.keys() {
            if let Some(bucket) = inv.get_mut(word as usize) {
                bucket.push(kf);
            }
        }
    }

    /// Remove `kf` from each bucket it appears in. The BoW must match the
    /// one used at `add` time (same set of words).
    pub fn erase(&self, kf: KeyFrameId, bow: &BowVector) {
        let mut inv = self.inverted.write().expect("poisoned");
        for &word in bow.0.keys() {
            if let Some(bucket) = inv.get_mut(word as usize)
                && let Some(pos) = bucket.iter().position(|&k| k == kf)
            {
                // Order is not significant, swap-remove is O(1).
                bucket.swap_remove(pos);
            }
        }
    }

    /// Empty every bucket. Capacity (one bucket per word) is preserved.
    pub fn clear(&self) {
        let mut inv = self.inverted.write().expect("poisoned");
        for bucket in inv.iter_mut() {
            bucket.clear();
        }
    }

    /// Drop every KeyFrame that belongs to `map`.
    pub fn clear_map(&self, view: &dyn KeyFrameView, map: MapId) {
        let mut inv = self.inverted.write().expect("poisoned");
        for bucket in inv.iter_mut() {
            bucket.retain(|&kf| view.map_id(kf) != Some(map));
        }
    }

    /// Total number of (kf, word) pairs currently stored. Useful in tests.
    #[doc(hidden)]
    pub fn _entry_count(&self) -> usize {
        self.inverted
            .read()
            .expect("poisoned")
            .iter()
            .map(|b| b.len())
            .sum()
    }

    // Deprecated
    pub fn detect_loop_candidates(
        &self,
        view: &dyn KeyFrameView,
        query: KeyFrameId,
        min_score: f32,
    ) -> Vec<KeyFrameId> {
        let Some(query_bow) = view.bow_vec(query) else {
            return Vec::new();
        };
        let Some(query_map) = view.map_id(query) else {
            return Vec::new();
        };
        let connected = view.connected_keyframes(query);

        // Phase 1: count shared words for KFs in the same map and not connected.
        let sharing = self.scan_sharing_words(query_bow, |kfi| {
            kfi != query && view.map_id(kfi) == Some(query_map) && !connected.contains(&kfi)
        });
        if sharing.is_empty() {
            return Vec::new();
        }

        // Phase 2: score and threshold.
        let max_words = sharing.values().copied().max().unwrap_or(0);
        let min_words = ((max_words as f32) * 0.8) as u32;
        let scored = score_candidates(view, query_bow, &self.voc, &sharing, min_words, min_score);
        if scored.is_empty() {
            return Vec::new();
        }

        // Phase 3: covisibility accumulation, 0.75·best_acc cutoff.
        let scores_map: HashMap<KeyFrameId, f32> =
            scored.iter().map(|&(score, kf)| (kf, score)).collect();
        let (acc, best_acc) =
            accumulate_covisibility(view, &scored, &scores_map, query, min_words, min_score);
        select_above_cutoff(&acc, 0.75 * best_acc)
    }

    pub fn detect_loop_and_merge_candidates(
        &self,
        view: &dyn KeyFrameView,
        query: KeyFrameId,
        min_score: f32,
    ) -> Candidates {
        let mut out = Candidates::default();
        let Some(query_bow) = view.bow_vec(query) else {
            return out;
        };
        let Some(query_map) = view.map_id(query) else {
            return out;
        };
        let connected = view.connected_keyframes(query);

        // Two parallel sharing maps: same-map → loop, other-(live)-map → merge.
        let mut loop_share: HashMap<KeyFrameId, u32> = HashMap::new();
        let mut merge_share: HashMap<KeyFrameId, u32> = HashMap::new();
        {
            let inv = self.inverted.read().expect("poisoned");
            for &word in query_bow.0.keys() {
                let Some(bucket) = inv.get(word as usize) else {
                    continue;
                };
                for &kfi in bucket {
                    if kfi == query || connected.contains(&kfi) {
                        continue;
                    }
                    match view.map_id(kfi) {
                        Some(m) if m == query_map => {
                            *loop_share.entry(kfi).or_insert(0) += 1;
                        }
                        Some(m) if !view.is_map_bad(m) => {
                            *merge_share.entry(kfi).or_insert(0) += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        out.loop_ = self.finalize_with_cutoff(view, query, query_bow, &loop_share, min_score);
        out.merge = self.finalize_with_cutoff(view, query, query_bow, &merge_share, min_score);
        out
    }

    pub fn detect_best_candidates(
        &self,
        view: &dyn KeyFrameView,
        query: KeyFrameId,
        n_min_words: u32,
    ) -> Candidates {
        let mut out = Candidates::default();
        let Some(query_bow) = view.bow_vec(query) else {
            return out;
        };
        let Some(query_map) = view.map_id(query) else {
            return out;
        };
        let connected = view.connected_keyframes(query);

        let sharing =
            self.scan_sharing_words(query_bow, |kfi| kfi != query && !connected.contains(&kfi));
        if sharing.is_empty() {
            return out;
        }

        let max_words = sharing.values().copied().max().unwrap_or(0);
        let min_words = (((max_words as f32) * 0.8) as u32).max(n_min_words);
        let scored = score_candidates(view, query_bow, &self.voc, &sharing, min_words, 0.0);
        if scored.is_empty() {
            return out;
        }

        let scores_map: HashMap<KeyFrameId, f32> =
            scored.iter().map(|&(score, kf)| (kf, score)).collect();
        let (acc, best_acc) =
            accumulate_covisibility(view, &scored, &scores_map, query, min_words, 0.0);
        let cutoff = 0.75 * best_acc;
        let kept = select_above_cutoff(&acc, cutoff);

        for kf in kept {
            if view.map_id(kf) == Some(query_map) {
                out.loop_.push(kf);
            } else {
                out.merge.push(kf);
            }
        }
        out
    }

    pub fn detect_n_best_candidates(
        &self,
        view: &dyn KeyFrameView,
        query: KeyFrameId,
        n: usize,
    ) -> Candidates {
        let mut out = Candidates::default();
        if n == 0 {
            return out;
        }
        let Some(query_bow) = view.bow_vec(query) else {
            return out;
        };
        let Some(query_map) = view.map_id(query) else {
            return out;
        };
        let connected = view.connected_keyframes(query);

        let sharing =
            self.scan_sharing_words(query_bow, |kfi| kfi != query && !connected.contains(&kfi));
        if sharing.is_empty() {
            return out;
        }

        let max_words = sharing.values().copied().max().unwrap_or(0);
        let min_words = ((max_words as f32) * 0.8) as u32;
        let scored = score_candidates(view, query_bow, &self.voc, &sharing, min_words, 0.0);
        if scored.is_empty() {
            return out;
        }

        let scores_map: HashMap<KeyFrameId, f32> =
            scored.iter().map(|&(score, kf)| (kf, score)).collect();
        let (mut acc, _best_acc) =
            accumulate_covisibility(view, &scored, &scores_map, query, min_words, 0.0);

        // Top-N by accumulated score, ties broken by id for determinism.
        acc.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });

        let mut already = HashSet::<KeyFrameId>::new();
        for (_score, kf) in acc {
            if out.loop_.len() >= n && out.merge.len() >= n {
                break;
            }
            if !already.insert(kf) {
                continue;
            }
            if view.is_bad(kf) {
                continue;
            }
            match view.map_id(kf) {
                Some(m) if m == query_map => {
                    if out.loop_.len() < n {
                        out.loop_.push(kf);
                    }
                }
                Some(m) if !view.is_map_bad(m) => {
                    if out.merge.len() < n {
                        out.merge.push(kf);
                    }
                }
                _ => {}
            }
        }
        out
    }

    pub fn detect_relocalization_candidates(
        &self,
        view: &dyn KeyFrameView,
        query_bow: &BowVector,
        map: MapId,
    ) -> Vec<KeyFrameId> {
        // No connected-set filter — a Frame has no covisibility yet.
        let sharing = self.scan_sharing_words(query_bow, |_| true);
        if sharing.is_empty() {
            return Vec::new();
        }

        let max_words = sharing.values().copied().max().unwrap_or(0);
        let min_words = ((max_words as f32) * 0.8) as u32;
        let scored = score_candidates(view, query_bow, &self.voc, &sharing, min_words, 0.0);
        if scored.is_empty() {
            return Vec::new();
        }

        let scores_map: HashMap<KeyFrameId, f32> =
            scored.iter().map(|&(score, kf)| (kf, score)).collect();
        let (acc, best_acc) = accumulate_covisibility(
            view,
            &scored,
            &scores_map,
            // No "self" KF to exclude — use a sentinel that can't appear.
            KeyFrameId(u64::MAX),
            min_words,
            0.0,
        );

        let cutoff = 0.75 * best_acc;
        select_above_cutoff(&acc, cutoff)
            .into_iter()
            .filter(|&kf| view.map_id(kf) == Some(map))
            .collect()
    }

    fn scan_sharing_words(
        &self,
        query_bow: &BowVector,
        mut accept: impl FnMut(KeyFrameId) -> bool,
    ) -> HashMap<KeyFrameId, u32> {
        let mut sharing: HashMap<KeyFrameId, u32> = HashMap::new();
        let inv = self.inverted.read().expect("poisoned");
        for &word in query_bow.0.keys() {
            let Some(bucket) = inv.get(word as usize) else {
                continue;
            };
            for &kfi in bucket {
                let entry = sharing.entry(kfi);
                use std::collections::hash_map::Entry;
                match entry {
                    Entry::Occupied(mut e) => {
                        *e.get_mut() += 1;
                    }
                    Entry::Vacant(slot) => {
                        if accept(kfi) {
                            slot.insert(1);
                        }
                    }
                }
            }
        }
        sharing
    }

    fn finalize_with_cutoff(
        &self,
        view: &dyn KeyFrameView,
        query: KeyFrameId,
        query_bow: &BowVector,
        sharing: &HashMap<KeyFrameId, u32>,
        min_score: f32,
    ) -> Vec<KeyFrameId> {
        if sharing.is_empty() {
            return Vec::new();
        }
        let max_words = sharing.values().copied().max().unwrap_or(0);
        let min_words = ((max_words as f32) * 0.8) as u32;
        let scored = score_candidates(view, query_bow, &self.voc, sharing, min_words, min_score);
        if scored.is_empty() {
            return Vec::new();
        }
        let scores_map: HashMap<KeyFrameId, f32> =
            scored.iter().map(|&(score, kf)| (kf, score)).collect();
        let (acc, best_acc) = accumulate_covisibility(
            view,
            &scored,
            &scores_map,
            query,
            min_words,
            min_score.max(0.0),
        );
        select_above_cutoff(&acc, 0.75 * best_acc)
    }
}

fn score_candidates(
    view: &dyn KeyFrameView,
    query_bow: &BowVector,
    voc: &OrbVocabulary,
    sharing: &HashMap<KeyFrameId, u32>,
    min_words: u32,
    min_score: f32,
) -> Vec<(f32, KeyFrameId)> {
    let mut out = Vec::with_capacity(sharing.len());
    for (&kf, &words) in sharing {
        if words <= min_words {
            continue;
        }
        let Some(bow) = view.bow_vec(kf) else {
            continue;
        };
        let s = voc.score(query_bow, bow) as f32;
        if s >= min_score {
            out.push((s, kf));
        }
    }
    out
}

fn accumulate_covisibility(
    view: &dyn KeyFrameView,
    scored: &[(f32, KeyFrameId)],
    scores_by_kf: &HashMap<KeyFrameId, f32>,
    query: KeyFrameId,
    _min_words: u32,
    init_best: f32,
) -> (Vec<(f32, KeyFrameId)>, f32) {
    let mut acc_list: Vec<(f32, KeyFrameId)> = Vec::with_capacity(scored.len());
    let mut best_acc = init_best;

    for &(score, kf) in scored {
        let neighbours = view.best_covisibility(kf, 10);

        let mut acc_score = score;
        let mut best_score = score;
        let mut best_kf = kf;

        for n in neighbours {
            if n == query {
                continue;
            }
            // A neighbour contributes iff it is itself in the scored set —
            // i.e. it passed the `min_words` and `min_score` filters above.
            if let Some(&n_score) = scores_by_kf.get(&n) {
                acc_score += n_score;
                if n_score > best_score {
                    best_score = n_score;
                    best_kf = n;
                }
            }
        }

        acc_list.push((acc_score, best_kf));
        if acc_score > best_acc {
            best_acc = acc_score;
        }
    }

    (acc_list, best_acc)
}

fn select_above_cutoff(acc: &[(f32, KeyFrameId)], cutoff: f32) -> Vec<KeyFrameId> {
    let mut seen = HashSet::<KeyFrameId>::new();
    let mut out = Vec::with_capacity(acc.len());
    for &(score, kf) in acc {
        if score > cutoff && seen.insert(kf) {
            out.push(kf);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orb_vocabulary::{DESC_LEN, OrbVocabulary};

    #[derive(Default)]
    struct MockKf {
        bow: BowVector,
        map: MapId,
        bad: bool,
        covis: Vec<KeyFrameId>,
        connected: HashSet<KeyFrameId>,
    }

    #[derive(Default)]
    struct MockView {
        kfs: HashMap<KeyFrameId, MockKf>,
        bad_maps: HashSet<MapId>,
    }

    impl MockView {
        fn insert(&mut self, id: KeyFrameId, kf: MockKf) {
            self.kfs.insert(id, kf);
        }
    }

    impl KeyFrameView for MockView {
        fn bow_vec(&self, kf: KeyFrameId) -> Option<&BowVector> {
            self.kfs.get(&kf).map(|k| &k.bow)
        }
        fn map_id(&self, kf: KeyFrameId) -> Option<MapId> {
            self.kfs.get(&kf).map(|k| k.map)
        }
        fn is_bad(&self, kf: KeyFrameId) -> bool {
            self.kfs.get(&kf).map(|k| k.bad).unwrap_or(true)
        }
        fn is_map_bad(&self, map: MapId) -> bool {
            self.bad_maps.contains(&map)
        }
        fn best_covisibility(&self, kf: KeyFrameId, n: usize) -> Vec<KeyFrameId> {
            self.kfs
                .get(&kf)
                .map(|k| k.covis.iter().take(n).copied().collect())
                .unwrap_or_default()
        }
        fn connected_keyframes(&self, kf: KeyFrameId) -> HashSet<KeyFrameId> {
            self.kfs
                .get(&kf)
                .map(|k| k.connected.clone())
                .unwrap_or_default()
        }
    }

    /// 4-word vocabulary built by hand: `k=4, L=1`, four leaves whose
    /// descriptors are linearly separable (one bit set per word).
    fn tiny_voc() -> Arc<OrbVocabulary> {
        let mut s = String::from("4 1 0 0\n");
        for word in 0..4 {
            s.push_str("0 1 ");
            // descriptor: byte `word` set to 0xFF, everything else 0
            for b in 0..DESC_LEN {
                if b == word {
                    s.push_str("255 ");
                } else {
                    s.push_str("0 ");
                }
            }
            s.push_str("1.0\n");
        }
        Arc::new(OrbVocabulary::load_from_reader(s.as_bytes()).expect("voc load"))
    }

    /// Build a BoW that places weight only on the listed word ids (uniform).
    /// L1-normalized.
    fn bow(words: &[u32]) -> BowVector {
        let mut b = BowVector::default();
        for &w in words {
            b.add_weight(w, 1.0);
        }
        b.normalize_l1();
        b
    }

    #[test]
    fn add_then_erase_round_trips() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc);

        let kf = KeyFrameId(42);
        let b = bow(&[0, 2]);
        db.add(kf, &b);
        assert_eq!(db._entry_count(), 2);
        db.erase(kf, &b);
        assert_eq!(db._entry_count(), 0);
    }

    #[test]
    fn clear_empties_all_buckets() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc);
        db.add(KeyFrameId(1), &bow(&[0, 1, 2, 3]));
        db.add(KeyFrameId(2), &bow(&[0, 2]));
        assert_eq!(db._entry_count(), 6);
        db.clear();
        assert_eq!(db._entry_count(), 0);
    }

    #[test]
    fn clear_map_only_removes_target_map() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc);
        let mut view = MockView::default();
        view.insert(
            KeyFrameId(1),
            MockKf {
                bow: bow(&[0, 1]),
                map: MapId(7),
                ..Default::default()
            },
        );
        view.insert(
            KeyFrameId(2),
            MockKf {
                bow: bow(&[0, 1]),
                map: MapId(8),
                ..Default::default()
            },
        );
        db.add(KeyFrameId(1), &view.kfs[&KeyFrameId(1)].bow);
        db.add(KeyFrameId(2), &view.kfs[&KeyFrameId(2)].bow);
        assert_eq!(db._entry_count(), 4);

        db.clear_map(&view, MapId(7));
        // Only KF 2's two entries remain.
        assert_eq!(db._entry_count(), 2);
    }

    #[test]
    fn relocalization_returns_only_target_map() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc.clone());
        let mut view = MockView::default();

        // Three KFs, two in map 1, one in map 2. All share words with query.
        for &(id, m) in &[(1u64, 1u64), (2, 1), (3, 2)] {
            let b = bow(&[0, 1, 2, 3]);
            view.insert(
                KeyFrameId(id),
                MockKf {
                    bow: b.clone(),
                    map: MapId(m),
                    ..Default::default()
                },
            );
            db.add(KeyFrameId(id), &b);
        }
        let query = bow(&[0, 1, 2, 3]);
        let cands = db.detect_relocalization_candidates(&view, &query, MapId(1));
        let set: HashSet<_> = cands.into_iter().collect();
        assert!(set.contains(&KeyFrameId(1)));
        assert!(set.contains(&KeyFrameId(2)));
        assert!(!set.contains(&KeyFrameId(3)));
    }

    #[test]
    fn loop_detection_skips_connected_keyframes() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc);
        let mut view = MockView::default();

        let q = KeyFrameId(100);
        let b = bow(&[0, 1, 2, 3]);

        // Query
        view.insert(
            q,
            MockKf {
                bow: b.clone(),
                map: MapId(1),
                connected: [KeyFrameId(2)].into_iter().collect(),
                ..Default::default()
            },
        );

        // Candidate 1: not connected, in same map → loop candidate.
        // Candidate 2: connected → must be excluded.
        for id in [1u64, 2] {
            view.insert(
                KeyFrameId(id),
                MockKf {
                    bow: b.clone(),
                    map: MapId(1),
                    ..Default::default()
                },
            );
            db.add(KeyFrameId(id), &b);
        }

        let cands = db.detect_loop_candidates(&view, q, 0.0);
        let set: HashSet<_> = cands.into_iter().collect();
        assert!(set.contains(&KeyFrameId(1)));
        assert!(!set.contains(&KeyFrameId(2)));
    }

    #[test]
    fn loop_and_merge_split_by_map() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc);
        let mut view = MockView::default();

        let q = KeyFrameId(100);
        let b = bow(&[0, 1, 2, 3]);

        view.insert(
            q,
            MockKf {
                bow: b.clone(),
                map: MapId(1),
                ..Default::default()
            },
        );

        // KF 1, 2 → same map (loop). KF 3 → different live map (merge).
        // KF 4 → different map but the map is bad → must be dropped.
        for &(id, m) in &[(1u64, 1u64), (2, 1), (3, 2), (4, 3)] {
            view.insert(
                KeyFrameId(id),
                MockKf {
                    bow: b.clone(),
                    map: MapId(m),
                    ..Default::default()
                },
            );
            db.add(KeyFrameId(id), &b);
        }
        view.bad_maps.insert(MapId(3));

        let res = db.detect_loop_and_merge_candidates(&view, q, 0.0);
        let l: HashSet<_> = res.loop_.into_iter().collect();
        let m: HashSet<_> = res.merge.into_iter().collect();
        assert!(l.contains(&KeyFrameId(1)));
        assert!(l.contains(&KeyFrameId(2)));
        assert!(m.contains(&KeyFrameId(3)));
        assert!(!m.contains(&KeyFrameId(4)));
    }

    #[test]
    fn n_best_caps_each_category_at_n_and_skips_bad() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc);
        let mut view = MockView::default();

        let q = KeyFrameId(100);
        let b = bow(&[0, 1, 2, 3]);
        view.insert(
            q,
            MockKf {
                bow: b.clone(),
                map: MapId(1),
                ..Default::default()
            },
        );

        // 4 same-map candidates and 4 other-map candidates, one bad in each.
        for &(id, map, bad) in &[
            (1u64, 1u64, false),
            (2, 1, false),
            (3, 1, true),
            (4, 1, false),
            (5, 2, false),
            (6, 2, true),
            (7, 2, false),
            (8, 2, false),
        ] {
            view.insert(
                KeyFrameId(id),
                MockKf {
                    bow: b.clone(),
                    map: MapId(map),
                    bad,
                    ..Default::default()
                },
            );
            db.add(KeyFrameId(id), &b);
        }

        let res = db.detect_n_best_candidates(&view, q, 2);
        assert!(res.loop_.len() <= 2);
        assert!(res.merge.len() <= 2);
        assert!(!res.loop_.contains(&KeyFrameId(3)), "bad KF leaked");
        assert!(!res.merge.contains(&KeyFrameId(6)), "bad KF leaked");
        for kf in &res.loop_ {
            assert_eq!(view.map_id(*kf), Some(MapId(1)));
        }
        for kf in &res.merge {
            assert_eq!(view.map_id(*kf), Some(MapId(2)));
        }
    }

    #[test]
    fn erase_then_detect_excludes_keyframe() {
        let voc = tiny_voc();
        let db = KeyFrameDatabase::new(voc);
        let mut view = MockView::default();

        let b = bow(&[0, 1, 2, 3]);
        for id in [1u64, 2] {
            view.insert(
                KeyFrameId(id),
                MockKf {
                    bow: b.clone(),
                    map: MapId(1),
                    ..Default::default()
                },
            );
            db.add(KeyFrameId(id), &b);
        }

        let cands = db.detect_relocalization_candidates(&view, &b, MapId(1));
        assert_eq!(cands.len(), 2);

        db.erase(KeyFrameId(1), &b);
        let cands = db.detect_relocalization_candidates(&view, &b, MapId(1));
        assert_eq!(cands, vec![KeyFrameId(2)]);
    }
}
