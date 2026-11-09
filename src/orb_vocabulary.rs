//! ORB vocabulary loader and matcher, compatible with DBoW2's text format
//!
//! Scope: ORB-SLAM3's exact configuration only — 256-bit (32-byte) ORB
//! descriptors, TF-IDF weighting, L1-norm scoring. We do *not* implement
//! arbitrary scoring/weighting combinations from DBoW2.
//!
//! ## File format
//!
//! ```text
//! <k> <L> <scoring> <weighting>
//! <parent_id> <is_leaf> <32 ints 0..255> <weight>
//! ...
//! ```
//!
//! - First line: `k` = branching factor, `L` = max depth, `scoring` = 0
//!   (L1_NORM), `weighting` = 0 (TF_IDF). ORB-SLAM3 uses `10 6 0 0`.
//! - Following lines: one node per line, in topological order so each
//!   `parent_id` already exists. Node id = `m_nodes.len()` at insertion time;
//!   word ids are assigned to leaves in insertion order. Node 0 is the
//!   implicit root and has no line.
//!
//! ## Transform
//!
//! For each descriptor we descend the tree picking the child with minimum
//! Hamming distance, accumulating TF weight at the resulting leaf and
//! recording the path-node at level `L - levelsup` for the
//! [`FeatureVector`]. The final [`BowVector`] is L1-normalized.
//!
//! ## Score
//!
//! L1 score from Nister 2006: `1 - 0.5 * Σ_i |v_i - w_i|`, computed via the
//! identity `score = -0.5 * Σ_{i ∈ both} (|v_i - w_i| - |v_i| - |w_i|)`
//! exploited over sorted sparse vectors.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use tracing::info;

/// Length of a single ORB descriptor in bytes (256 bits).
pub const DESC_LEN: usize = 32;

/// 256-bit ORB descriptor, byte-packed.
pub type Descriptor = [u8; DESC_LEN];

/// Identifier of a leaf (word) in the vocabulary tree.
pub type WordId = u32;

/// Identifier of any node in the vocabulary tree (internal or leaf).
pub type NodeId = u32;

/// Sparse bag-of-words vector: `word_id -> weight`. Kept sorted by id so the
/// L1 scoring routine can do a single linear merge.
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BowVector(pub BTreeMap<WordId, f64>);

impl BowVector {
    /// Add `v` to the entry for `id`, inserting if absent.
    pub fn add_weight(&mut self, id: WordId, v: f64) {
        *self.0.entry(id).or_insert(0.0) += v;
    }

    /// L1-normalize the vector in place. No-op if the vector is empty or all
    /// zeros.
    pub fn normalize_l1(&mut self) {
        let norm: f64 = self.0.values().map(|w| w.abs()).sum();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for w in self.0.values_mut() {
                *w *= inv;
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// Sparse feature vector: `node_id -> indices of input features that
/// descended through this node at the chosen level`. Used by ORB-SLAM3 to
/// restrict descriptor matching to features under the same vocabulary node.
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FeatureVector(pub BTreeMap<NodeId, Vec<u32>>);

impl FeatureVector {
    /// Append `i_feature` to the bucket for `node_id`. Indices for a given
    /// node end up sorted because callers feed them in order.
    pub fn add_feature(&mut self, node_id: NodeId, i_feature: u32) {
        self.0.entry(node_id).or_default().push(i_feature);
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// A single node in the vocabulary tree. The root (id 0) is implicit and has
/// no descriptor / weight.
#[derive(Debug, Clone)]
struct Node {
    parent: NodeId,
    /// Children node ids. Empty iff this node is a leaf.
    children: Vec<NodeId>,
    /// Closest cluster centroid descriptor for this node.
    descriptor: Descriptor,
    /// Weight; for ORB-SLAM3 (TF-IDF) this is the IDF for leaves and unused
    /// for internal nodes.
    weight: f64,
    /// `Some` iff this is a leaf.
    word_id: Option<WordId>,
}

impl Node {
    fn root() -> Self {
        Self {
            parent: 0,
            children: Vec::new(),
            descriptor: [0; DESC_LEN],
            weight: 0.0,
            word_id: None,
        }
    }
    fn is_leaf(&self) -> bool {
        self.word_id.is_some()
    }
}

/// Errors returned by [`OrbVocabulary::load_from_text_file`].
#[derive(Debug)]
pub enum VocabularyError {
    /// The file could not be opened or read.
    Io(io::Error),
    /// The header line did not parse.
    InvalidHeader(String),
    /// A node line did not parse.
    InvalidNode { line_no: usize, reason: String },
    /// Header values fall outside DBoW2's accepted ranges.
    OutOfRange(String),
    /// `parent_id` referred to a node that has not been inserted yet.
    DanglingParent { line_no: usize, parent: NodeId },
}

impl std::fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::InvalidHeader(s) => write!(f, "invalid vocabulary header: {s}"),
            Self::InvalidNode { line_no, reason } => {
                write!(f, "invalid node on line {line_no}: {reason}")
            }
            Self::OutOfRange(s) => write!(f, "header out of range: {s}"),
            Self::DanglingParent { line_no, parent } => {
                write!(f, "line {line_no}: parent {parent} not yet defined")
            }
        }
    }
}

impl std::error::Error for VocabularyError {}

impl From<io::Error> for VocabularyError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// ORB vocabulary tree. After loading, `nodes[0]` is the root and `words[i]`
/// points back to the leaf node that owns word id `i`.
#[derive(Debug, Clone)]
pub struct OrbVocabulary {
    k: u32,
    l: u32,
    /// All nodes, indexed by [`NodeId`]. `nodes[0]` is the implicit root.
    nodes: Vec<Node>,
    /// `words[wid]` is the [`NodeId`] of the leaf with that word id.
    words: Vec<NodeId>,
}

impl OrbVocabulary {
    /// Load the vocabulary from a DBoW2 text file (e.g. ORB-SLAM3's
    /// `orbvoc.txt`).
    pub fn load_from_text_file(path: impl AsRef<Path>) -> Result<Self, VocabularyError> {
        let path = path.as_ref();
        info!("Loading ORB vocabulary from {}", path.display());
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        Self::load_from_reader(reader)
    }

    /// Same as [`load_from_text_file`](Self::load_from_text_file) but takes
    /// any [`BufRead`]. Useful for tests with synthetic inputs.
    pub fn load_from_reader<R: BufRead>(mut reader: R) -> Result<Self, VocabularyError> {
        // Header
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            return Err(VocabularyError::InvalidHeader("empty file".into()));
        }
        let mut hp = header.split_whitespace();
        let k: i32 = hp
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| VocabularyError::InvalidHeader(header.trim().into()))?;
        let l: i32 = hp
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| VocabularyError::InvalidHeader(header.trim().into()))?;
        let scoring: i32 = hp
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| VocabularyError::InvalidHeader(header.trim().into()))?;
        let weighting: i32 = hp
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| VocabularyError::InvalidHeader(header.trim().into()))?;

        // DBoW2 enforces these ranges; we mirror them so a corrupted file is
        // detected up-front rather than producing nonsense words.
        if !(0..=20).contains(&k)
            || !(1..=10).contains(&l)
            || !(0..=5).contains(&scoring)
            || !(0..=3).contains(&weighting)
        {
            return Err(VocabularyError::OutOfRange(format!(
                "k={k} L={l} scoring={scoring} weighting={weighting}"
            )));
        }

        // ORB-SLAM3 ships TF-IDF + L1; warn but do not reject otherwise — the
        // transform/score implementation here only matches that combination.
        if scoring != 0 || weighting != 0 {
            tracing::warn!(
                "vocabulary uses scoring={scoring}, weighting={weighting}; \
                 only L1 / TF-IDF is implemented — results may differ from DBoW2"
            );
        }

        // Pre-size nodes optimistically: full k-ary tree of depth L has
        // (k^(L+1) - 1) / (k - 1) nodes. Cap to keep allocation reasonable
        // for misbehaved files.
        let expected_nodes = if k > 1 {
            // Saturating arithmetic in f64 avoids overflow for k=20, L=10.
            let approx = ((k as f64).powi(l + 1) - 1.0) / ((k - 1) as f64);
            (approx as usize).min(8_000_000)
        } else {
            l as usize + 1
        };
        let mut nodes: Vec<Node> = Vec::with_capacity(expected_nodes);
        nodes.push(Node::root());
        let mut words: Vec<NodeId> = Vec::new();

        let mut line_buf = String::new();
        let mut line_no = 1; // header was line 1
        loop {
            line_buf.clear();
            line_no += 1;
            let n = reader.read_line(&mut line_buf)?;
            if n == 0 {
                break;
            }
            if line_buf.trim().is_empty() {
                continue;
            }
            parse_node_line(&line_buf, line_no, &mut nodes, &mut words)?;
        }

        info!(
            "ORB vocabulary loaded: k={k}, L={l}, nodes={}, words={}",
            nodes.len(),
            words.len()
        );

        Ok(Self {
            k: k as u32,
            l: l as u32,
            nodes,
            words,
        })
    }

    /// Branching factor.
    pub fn k(&self) -> u32 {
        self.k
    }
    /// Maximum tree depth.
    pub fn l(&self) -> u32 {
        self.l
    }
    /// Number of words (leaves).
    pub fn size(&self) -> usize {
        self.words.len()
    }
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Descend the tree for a single descriptor. Returns the matching word
    /// id, its IDF weight, and the node id encountered at level
    /// `L - levelsup` (or 0 if `levelsup >= L`).
    pub fn transform_one(&self, desc: &Descriptor, levelsup: u32) -> (WordId, f64, NodeId) {
        // Capture the path node at this level. levelsup=0 → leaf; levelsup=L → root.
        let nid_level = self.l.saturating_sub(levelsup) as i32;
        let mut current_level: i32 = 0;
        let mut nid_out: NodeId = 0;
        let mut current: NodeId = 0;

        loop {
            current_level += 1;
            let node = &self.nodes[current as usize];
            debug_assert!(!node.children.is_empty(), "non-leaf has no children");

            let mut best_id = node.children[0];
            let mut best_d = hamming(desc, &self.nodes[best_id as usize].descriptor);
            for &cid in &node.children[1..] {
                let d = hamming(desc, &self.nodes[cid as usize].descriptor);
                if d < best_d {
                    best_d = d;
                    best_id = cid;
                }
            }
            current = best_id;

            if nid_level > 0 && current_level == nid_level {
                nid_out = current;
            }

            if self.nodes[current as usize].is_leaf() {
                break;
            }
        }

        let leaf = &self.nodes[current as usize];
        // Safe: we just confirmed `is_leaf()`.
        let wid = leaf.word_id.expect("leaf without word id");
        // If the requested level is at-or-below the leaf, the C++ code leaves
        // nid at root (0). We mirror that.
        if nid_level <= 0 {
            nid_out = 0;
        }
        (wid, leaf.weight, nid_out)
    }

    /// Transform a batch of descriptors into a [`BowVector`] +
    /// [`FeatureVector`] using TF-IDF weighting and L1 normalization, the
    /// configuration used by ORB-SLAM3.
    ///
    /// `levelsup` matches DBoW2's parameter: how many levels above the leaf
    /// the FeatureVector node is captured. ORB-SLAM3 typically uses 4
    /// (with `L=6`, that captures level 2).
    pub fn transform(
        &self,
        descriptors: &[Descriptor],
        levelsup: u32,
    ) -> (BowVector, FeatureVector) {
        let mut bow = BowVector::default();
        let mut fv = FeatureVector::default();

        if self.is_empty() {
            return (bow, fv);
        }

        for (i, d) in descriptors.iter().enumerate() {
            let (wid, w, nid) = self.transform_one(d, levelsup);
            if w > 0.0 {
                bow.add_weight(wid, w);
                fv.add_feature(nid, i as u32);
            }
        }
        bow.normalize_l1();
        (bow, fv)
    }

    /// L1 score in `[0, 1]` between two BoW vectors, matching DBoW2's
    /// `L1Scoring::score`. Both vectors should already be L1-normalized
    /// (which [`transform`](Self::transform) guarantees).
    pub fn score(&self, a: &BowVector, b: &BowVector) -> f64 {
        l1_score(a, b)
    }
}

/// L1 score `1 - 0.5 * ||a - b||_1` over two sorted sparse vectors. This is
/// a free function so callers can score without holding the vocabulary.
pub fn l1_score(a: &BowVector, b: &BowVector) -> f64 {
    let mut it_a = a.0.iter().peekable();
    let mut it_b = b.0.iter().peekable();
    let mut acc = 0.0f64;
    while let (Some(&(&ka, &va)), Some(&(&kb, &vb))) = (it_a.peek(), it_b.peek()) {
        if ka == kb {
            // |va-vb| - |va| - |vb|
            acc += (va - vb).abs() - va.abs() - vb.abs();
            it_a.next();
            it_b.next();
        } else if ka < kb {
            it_a.next();
        } else {
            it_b.next();
        }
    }
    -acc / 2.0
}

/// Hamming distance between two 32-byte descriptors. Operates on packed
/// `u64`s for ~4× speedup over a byte loop.
#[inline]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    let mut d = 0u32;
    // 4 × u64 = 32 bytes
    for i in 0..4 {
        let off = i * 8;
        let aa = u64::from_le_bytes(a[off..off + 8].try_into().unwrap());
        let bb = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
        d += (aa ^ bb).count_ones();
    }
    d
}

/// Parse one node line and append it to `nodes` (and `words` if it is a
/// leaf).
fn parse_node_line(
    line: &str,
    line_no: usize,
    nodes: &mut Vec<Node>,
    words: &mut Vec<NodeId>,
) -> Result<(), VocabularyError> {
    let mut it = line.split_whitespace();
    let invalid = |reason: String| VocabularyError::InvalidNode { line_no, reason };

    let parent: NodeId = it
        .next()
        .ok_or_else(|| invalid("missing parent id".into()))?
        .parse()
        .map_err(|e: std::num::ParseIntError| invalid(format!("bad parent id: {e}")))?;
    let is_leaf: i32 = it
        .next()
        .ok_or_else(|| invalid("missing is_leaf".into()))?
        .parse()
        .map_err(|e: std::num::ParseIntError| invalid(format!("bad is_leaf: {e}")))?;

    let mut desc = [0u8; DESC_LEN];
    for (i, slot) in desc.iter_mut().enumerate() {
        let v: u32 = it
            .next()
            .ok_or_else(|| invalid(format!("missing descriptor byte {i}")))?
            .parse()
            .map_err(|e: std::num::ParseIntError| {
                invalid(format!("bad descriptor byte {i}: {e}"))
            })?;
        if v > 255 {
            return Err(invalid(format!("descriptor byte {i} out of range: {v}")));
        }
        *slot = v as u8;
    }

    let weight: f64 = it
        .next()
        .ok_or_else(|| invalid("missing weight".into()))?
        .parse()
        .map_err(|e: std::num::ParseFloatError| invalid(format!("bad weight: {e}")))?;

    if (parent as usize) >= nodes.len() {
        return Err(VocabularyError::DanglingParent { line_no, parent });
    }

    let nid = nodes.len() as NodeId;
    let word_id = if is_leaf > 0 {
        let wid = words.len() as WordId;
        words.push(nid);
        Some(wid)
    } else {
        None
    };
    nodes.push(Node {
        parent,
        children: Vec::new(),
        descriptor: desc,
        weight,
        word_id,
    });
    nodes[parent as usize].children.push(nid);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build a tiny synthetic 2-word vocabulary by hand for deterministic
    /// transform/score tests. `k=2, L=1`, root has two leaf children.
    fn tiny_vocab_text() -> String {
        let mut s = String::new();
        s.push_str("2 1 0 0\n");
        // Leaf word 0: descriptor = all zeros, weight = 1.0
        s.push_str("0 1 ");
        for _ in 0..DESC_LEN {
            s.push_str("0 ");
        }
        s.push_str("1.0\n");
        // Leaf word 1: descriptor = all 0xFF, weight = 2.0
        s.push_str("0 1 ");
        for _ in 0..DESC_LEN {
            s.push_str("255 ");
        }
        s.push_str("2.0\n");
        s
    }

    #[test]
    fn parse_header_and_two_leaves() {
        let voc = OrbVocabulary::load_from_reader(tiny_vocab_text().as_bytes()).expect("loads");
        assert_eq!(voc.k(), 2);
        assert_eq!(voc.l(), 1);
        assert_eq!(voc.size(), 2);
        assert_eq!(voc.nodes.len(), 3); // root + 2 leaves
        assert_eq!(voc.nodes[0].children, vec![1, 2]);
        assert!(voc.nodes[1].is_leaf());
        assert_eq!(voc.nodes[1].word_id, Some(0));
        assert_eq!(voc.nodes[2].word_id, Some(1));
    }

    #[test]
    fn rejects_out_of_range_header() {
        // k=21 exceeds DBoW2's accepted range of 0..=20.
        let bad = "21 1 0 0\nignored\n";
        let err = OrbVocabulary::load_from_reader(bad.as_bytes()).unwrap_err();
        assert!(matches!(err, VocabularyError::OutOfRange(_)));
    }

    #[test]
    fn rejects_dangling_parent() {
        // parent=99, no such node exists yet.
        let mut s = String::from("2 1 0 0\n99 1 ");
        for _ in 0..DESC_LEN {
            s.push_str("0 ");
        }
        s.push_str("1.0\n");
        let err = OrbVocabulary::load_from_reader(s.as_bytes()).unwrap_err();
        assert!(matches!(err, VocabularyError::DanglingParent { .. }));
    }

    #[test]
    fn transform_picks_nearest_word() {
        let voc = OrbVocabulary::load_from_reader(tiny_vocab_text().as_bytes()).expect("loads");

        // Pure-zero descriptor → word 0.
        let zero = [0u8; DESC_LEN];
        let (wid, w, _nid) = voc.transform_one(&zero, 0);
        assert_eq!(wid, 0);
        assert!((w - 1.0).abs() < 1e-12);

        // Pure-ones descriptor → word 1.
        let ones = [0xFFu8; DESC_LEN];
        let (wid, w, _nid) = voc.transform_one(&ones, 0);
        assert_eq!(wid, 1);
        assert!((w - 2.0).abs() < 1e-12);
    }

    #[test]
    fn transform_normalizes_l1_and_groups_features() {
        let voc = OrbVocabulary::load_from_reader(tiny_vocab_text().as_bytes()).expect("loads");

        // Two zero descriptors and one all-ones descriptor:
        //   word 0 raw weight = 1.0 * 2 = 2.0
        //   word 1 raw weight = 2.0 * 1 = 2.0
        //   L1 norm = 4.0  →  both → 0.5
        let descs = vec![[0u8; DESC_LEN], [0u8; DESC_LEN], [0xFFu8; DESC_LEN]];
        let (bow, fv) = voc.transform(&descs, 1);
        assert_eq!(bow.len(), 2);
        assert!((bow.0[&0] - 0.5).abs() < 1e-12);
        assert!((bow.0[&1] - 0.5).abs() < 1e-12);

        // levelsup=L → all features bucket under root (nid=0).
        assert_eq!(fv.0.len(), 1);
        assert_eq!(fv.0[&0], vec![0u32, 1, 2]);
    }

    #[test]
    fn l1_score_self_is_one_disjoint_is_zero() {
        let voc = OrbVocabulary::load_from_reader(tiny_vocab_text().as_bytes()).expect("loads");

        let only_zero = vec![[0u8; DESC_LEN]];
        let only_one = vec![[0xFFu8; DESC_LEN]];
        let (a, _) = voc.transform(&only_zero, 0);
        let (b, _) = voc.transform(&only_one, 0);

        assert!((voc.score(&a, &a) - 1.0).abs() < 1e-12);
        assert!(voc.score(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn hamming_packs_correctly() {
        let a = [0u8; DESC_LEN];
        let mut b = [0u8; DESC_LEN];
        b[0] = 0xFF; // 8 bits set
        b[15] = 0x0F; // 4 bits set
        b[31] = 0x80; // 1 bit set
        assert_eq!(hamming(&a, &b), 8 + 4 + 1);
    }

    /// Smoke test against the real ORB-SLAM3 vocabulary, when available.
    /// Skipped on systems where the build step has not unpacked it.
    #[test]
    fn loads_real_orbvoc_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vocabulary/ORBvoc.txt");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let voc = OrbVocabulary::load_from_text_file(&path).expect("load");
        assert_eq!(voc.k(), 10);
        assert_eq!(voc.l(), 6);
        // The standard ORB-SLAM2/3 vocabulary has ~1M words.
        assert!(
            voc.size() > 100_000,
            "expected a large vocabulary, got {} words",
            voc.size()
        );

        // Transform a couple of synthetic descriptors and verify the BoW is
        // L1-normalized.
        let descs = vec![[0u8; DESC_LEN], [0xAAu8; DESC_LEN], [0x5Au8; DESC_LEN]];
        let (bow, _fv) = voc.transform(&descs, 4);
        assert!(!bow.is_empty());
        let sum: f64 = bow.0.values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "BoW not L1-normalized: sum={sum}");
    }
}
