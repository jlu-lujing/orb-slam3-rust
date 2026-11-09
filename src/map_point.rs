//! [`MapPoint`]: a 3-D landmark observed by one or more [`KeyFrame`]s.
//!
//! ## Threading model
//!
//! Unlike [`crate::frame::Frame`] (single-owner, lock-free), a `MapPoint` is a
//! genuinely shared, mutably-aliased graph node: Tracking, LocalMapping and
//! LoopClosing all read and write it concurrently through `Arc<MapPoint>`. The
//! cleanest faithful model is therefore interior mutability.
//!
//! * `pos`  — world position, mean viewing normal and scale-invariance
//!   distances.
//! * `feat` — observations, reference keyframe, best descriptor, visible/found
//!   counters and the bad/replaced flags.
//! * `map`  — owning map pointer.
//!
//! Cross-object operations (`set_bad_flag`, `replace`) follow the
//! discipline of *snapshotting under the lock, releasing, then* calling into
//! other objects — this keeps the lock hold times short and avoids re-entrant
//! deadlocks.
//!
//! Graph edges that would otherwise create `Arc` cycles
//! (`MapPoint → ref_kf → KeyFrame → map_points → MapPoint`) are stored as
//! [`Weak`] references.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};

use nalgebra::Vector3;
use opencv::core::{KeyPointTraitConst, Mat, MatTraitConst};

use crate::frame::Frame;
use crate::key_frame::KeyFrame;
use crate::key_frame_database::KeyFrameId;
use crate::map::Map;
use crate::orb_matcher::descriptor_distance;
use crate::serialization_utils::SerializableMat;
use serde::{Deserialize, Serialize};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Current value of the global map-point-id counter (for serialization).
pub(crate) fn peek_next_id() -> usize {
    NEXT_ID.load(Ordering::SeqCst)
}
/// Restore the global map-point-id counter (after deserialization).
pub(crate) fn set_next_id(v: usize) {
    NEXT_ID.store(v, Ordering::SeqCst);
}

/// A live observation: an observing keyframe and its `(left, right)` indices.
type LiveObservation = (Arc<KeyFrame>, (i32, i32));

/// A single observation of a map point inside a keyframe.
struct Observation {
    /// Weak handle to the observing keyframe (avoids an `Arc` cycle).
    kf: Weak<KeyFrame>,
    /// Index in the left image (or `-1`).
    left: i32,
    /// Index in the right image (or `-1`).
    right: i32,
}

/// Position-group state
struct PosState {
    world_pos: Vector3<f32>,
    normal_vector: Vector3<f32>,
    min_distance: f32,
    max_distance: f32,
}

/// Feature-group state
struct FeatState {
    observations: HashMap<KeyFrameId, Observation>,
    n_obs: i32,
    ref_kf: Weak<KeyFrame>,
    descriptor: Mat,
    visible: i32,
    found: i32,
    bad: bool,
    replaced: Option<Arc<MapPoint>>,
}

pub struct MapPoint {
    pub id: usize,
    pub first_kf_id: Option<u64>,
    pub first_frame: usize,
    pub origin_map_id: u32,

    // Single-thread (Tracking) scratch. Populated by the tracker
    // kept as plain fields so the matcher can read them. They are
    // read-only through `Arc` until Tracking wires them up.
    pub track_proj_x: f32,
    pub track_proj_y: f32,
    pub track_proj_xr: f32,
    pub track_proj_yr: f32,
    pub track_depth: f32,
    pub track_depth_r: f32,
    pub track_in_view: bool,
    pub track_in_view_r: bool,
    pub track_scale_level: i32,
    pub track_scale_level_r: i32,
    pub track_view_cos: f32,
    pub track_view_cos_r: f32,
    pub track_reference_for_frame: u64,
    pub last_frame_seen: u64,

    // Shared mutable state
    pos: RwLock<PosState>,
    feat: RwLock<FeatState>,
    // Weak back-pointer: the owning `Map` strongly holds this point, so a strong
    // ref here would form an `Arc` cycle and leak the whole map.
    map: RwLock<Weak<Map>>,
    /// Global-BA staging (loop-closing path); mirrors C++ `mPosGBA` /
    /// `mnBAGlobalForKF`. Not serialized.
    gba: RwLock<MpGba>,
}

/// Global-BA staging for a map point.
#[derive(Clone, Default)]
struct MpGba {
    pos_gba: Option<Vector3<f32>>,
    mn_ba_global_for_kf: i64,
}

impl PartialEq for MapPoint {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for MapPoint {}
impl Hash for MapPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl MapPoint {
    /// Build the common shared-state skeleton used by every constructor.
    #[allow(clippy::too_many_arguments)]
    fn build(
        id: usize,
        first_kf_id: Option<u64>,
        first_frame: usize,
        origin_map_id: u32,
        world_pos: Vector3<f32>,
        normal_vector: Vector3<f32>,
        min_distance: f32,
        max_distance: f32,
        ref_kf: Weak<KeyFrame>,
        descriptor: Mat,
        map: Option<Arc<Map>>,
    ) -> Self {
        MapPoint {
            id,
            first_kf_id,
            first_frame,
            origin_map_id,
            track_proj_x: 0.0,
            track_proj_y: 0.0,
            track_proj_xr: 0.0,
            track_proj_yr: 0.0,
            track_depth: 0.0,
            track_depth_r: 0.0,
            track_in_view: false,
            track_in_view_r: false,
            track_scale_level: -1,
            track_scale_level_r: -1,
            track_view_cos: 0.0,
            track_view_cos_r: 0.0,
            track_reference_for_frame: 0,
            last_frame_seen: 0,
            pos: RwLock::new(PosState {
                world_pos,
                normal_vector,
                min_distance,
                max_distance,
            }),
            feat: RwLock::new(FeatState {
                observations: HashMap::new(),
                n_obs: 0,
                ref_kf,
                descriptor,
                visible: 1,
                found: 1,
                bad: false,
                replaced: None,
            }),
            map: RwLock::new(map.as_ref().map(Arc::downgrade).unwrap_or_default()),
            gba: RwLock::new(MpGba::default()),
        }
    }

    /// Store global-BA result without applying it: `mPosGBA` + `mnBAGlobalForKF`.
    pub fn set_pos_gba(&self, pos_gba: Vector3<f32>, loop_kf: i64) {
        let mut g = self.gba.write().unwrap();
        g.pos_gba = Some(pos_gba);
        g.mn_ba_global_for_kf = loop_kf;
    }
    /// Read the staged global-BA position, if any.
    pub fn get_pos_gba(&self) -> Option<Vector3<f32>> {
        self.gba.read().unwrap().pos_gba
    }
    /// Read `mnBAGlobalForKF`.
    pub fn get_ba_global_for_kf(&self) -> i64 {
        self.gba.read().unwrap().mn_ba_global_for_kf
    }

    /// Default-constructed map point (no position, no reference).
    pub fn new() -> Self {
        Self::build(
            NEXT_ID.fetch_add(1, Ordering::SeqCst),
            Some(0),
            0,
            0,
            Vector3::zeros(),
            Vector3::zeros(),
            0.0,
            0.0,
            Weak::new(),
            Mat::default(),
            None,
        )
    }

    /// Construct from a world position and a reference keyframe
    pub fn from_pos_ref(pos: &Vector3<f32>, ref_kf: Arc<KeyFrame>, map: Arc<Map>) -> Arc<Self> {
        let mp = Self::build(
            NEXT_ID.fetch_add(1, Ordering::SeqCst),
            Some(ref_kf.id),
            ref_kf.frame_id,
            map.get_id(),
            *pos,
            Vector3::zeros(),
            0.0,
            0.0,
            Arc::downgrade(&ref_kf),
            Mat::default(),
            Some(map),
        );
        Arc::new(mp)
    }

    /// Construct from a world position observed by a `Frame` keypoint
    pub fn from_frame(pos: &Vector3<f32>, map: Arc<Map>, frame: &Frame, idx_f: usize) -> Arc<Self> {
        // Camera centre that observed the keypoint.
        let ow = match frame.n_left {
            None => frame.get_camera_center(),
            Some(n_left) if idx_f < n_left => frame.get_camera_center(),
            Some(_) => {
                let rwl = frame.get_rwc();
                let tlr = frame.get_relative_pose_tlr().translation.vector;
                let twl = frame.get_ow();
                rwl * tlr + twl
            }
        };

        let normal = pos - ow;
        let normal = normal / normal.norm();

        let dist = (pos - ow).norm();
        let level = match frame.n_left {
            None => frame.keys_un.as_ref().expect("keys_un")[idx_f].octave(),
            Some(n_left) if idx_f < n_left => frame.keys[idx_f].octave(),
            Some(n_left) => frame.keys_right.as_ref().expect("keys_right")[idx_f - n_left].octave(),
        } as usize;
        let level_scale_factor = frame.scale_factors[level];
        let n_levels = frame.scale_levels;

        let max_distance = dist * level_scale_factor;
        let min_distance = max_distance / frame.scale_factors[n_levels - 1];

        let descriptor = frame
            .descriptors
            .row(idx_f as i32)
            .expect("descriptor row")
            .try_clone()
            .expect("clone descriptor");

        let mp = Self::build(
            NEXT_ID.fetch_add(1, Ordering::SeqCst),
            None,
            frame.id,
            map.get_id(),
            *pos,
            normal,
            min_distance,
            max_distance,
            Weak::new(),
            descriptor,
            Some(map),
        );
        Arc::new(mp)
    }

    pub fn set_world_pos(&self, pos: Vector3<f32>) {
        self.pos.write().unwrap().world_pos = pos;
    }
    pub fn get_world_pos(&self) -> Vector3<f32> {
        self.pos.read().unwrap().world_pos
    }
    pub fn get_normal(&self) -> Vector3<f32> {
        self.pos.read().unwrap().normal_vector
    }
    pub fn set_normal_vector(&self, normal: Vector3<f32>) {
        self.pos.write().unwrap().normal_vector = normal;
    }
    pub fn get_reference_keyframe(&self) -> Option<Arc<KeyFrame>> {
        self.feat.read().unwrap().ref_kf.upgrade()
    }
    pub fn set_reference_keyframe(&self, kf: &Arc<KeyFrame>) {
        self.feat.write().unwrap().ref_kf = Arc::downgrade(kf);
    }

    /// Add an observation of this point in `kf` at keypoint index `idx`.
    pub fn add_observation(&self, kf: &Arc<KeyFrame>, idx: i32) {
        let mut feat = self.feat.write().unwrap();
        let key = KeyFrameId(kf.id);
        let entry = feat.observations.entry(key).or_insert_with(|| Observation {
            kf: Arc::downgrade(kf),
            left: -1,
            right: -1,
        });
        entry.kf = Arc::downgrade(kf);

        let is_right = kf.n_left.is_some_and(|n_left| idx >= n_left as i32);
        if is_right {
            entry.right = idx;
        } else {
            entry.left = idx;
        }

        // Stereo (non-fisheye) observations with a valid right coord count twice.
        let stereo =
            kf.camera2.is_none() && kf.u_right.get(idx as usize).is_some_and(|&u| u >= 0.0);
        feat.n_obs += if stereo { 2 } else { 1 };
    }

    /// Remove the observation in `kf`; if too few remain, flag the point bad.
    pub fn erase_observation(self: &Arc<Self>, kf: &Arc<KeyFrame>) {
        let mut bad = false;
        {
            let mut feat = self.feat.write().unwrap();
            if let Some(obs) = feat.observations.remove(&KeyFrameId(kf.id)) {
                if obs.left != -1 {
                    let stereo = kf.camera2.is_none()
                        && kf.u_right.get(obs.left as usize).is_some_and(|&u| u >= 0.0);
                    feat.n_obs -= if stereo { 2 } else { 1 };
                }
                if obs.right != -1 {
                    feat.n_obs -= 1;
                }

                // Pick a new reference keyframe if the erased one was it.
                let was_ref = feat.ref_kf.upgrade().is_some_and(|r| r.id == kf.id);
                if was_ref
                    && let Some(next) = feat.observations.values().find_map(|o| o.kf.upgrade())
                {
                    feat.ref_kf = Arc::downgrade(&next);
                }

                if feat.n_obs <= 2 {
                    bad = true;
                }
            }
        }
        if bad {
            self.set_bad_flag();
        }
    }

    /// `(left_index, right_index)` of this point's observation in `kf`,
    /// each `-1` when not observed on that camera.
    pub fn get_index_in_keyframe(&self, kf: &KeyFrame) -> (i32, i32) {
        let feat = self.feat.read().unwrap();
        feat.observations
            .get(&KeyFrameId(kf.id))
            .map(|o| (o.left, o.right))
            .unwrap_or((-1, -1))
    }

    pub fn is_in_keyframe(&self, kf: &KeyFrame) -> bool {
        self.feat
            .read()
            .unwrap()
            .observations
            .contains_key(&KeyFrameId(kf.id))
    }

    /// Number of weighted observations (`nObs`).
    pub fn observations(&self) -> i32 {
        self.feat.read().unwrap().n_obs
    }

    /// `(keyframe, (left, right))` for every observing keyframe still alive.
    pub fn get_observations(&self) -> Vec<(Arc<KeyFrame>, (i32, i32))> {
        self.feat
            .read()
            .unwrap()
            .observations
            .values()
            .filter_map(|o| o.kf.upgrade().map(|kf| (kf, (o.left, o.right))))
            .collect()
    }

    // Bad flag / replacement
    pub fn is_bad(&self) -> bool {
        self.feat.read().unwrap().bad
    }

    // Mark the point bad and detach it from all observing keyframes and the map.
    pub fn set_bad_flag(self: &Arc<Self>) {
        // Snapshot observations under the lock, then release before calling
        // into the keyframes / map
        let obs: Vec<(Arc<KeyFrame>, (i32, i32))>;
        {
            let mut feat = self.feat.write().unwrap();
            feat.bad = true;
            obs = feat
                .observations
                .drain()
                .filter_map(|(_, o)| o.kf.upgrade().map(|kf| (kf, (o.left, o.right))))
                .collect();
        }
        for (kf, (left, right)) in obs {
            if left != -1 {
                kf.erase_map_point_match_idx(left as usize);
            }
            if right != -1 {
                kf.erase_map_point_match_idx(right as usize);
            }
        }
        if let Some(map) = self.map.read().unwrap().upgrade() {
            map.erase_map_point(self);
        }
    }

    pub fn get_replaced(&self) -> Option<Arc<MapPoint>> {
        self.feat.read().unwrap().replaced.clone()
    }

    // Replace this point with `other` in every observing keyframe.
    pub fn replace(self: &Arc<Self>, other: &Arc<MapPoint>) {
        if other.id == self.id {
            return;
        }

        let (obs, n_visible, n_found): (Vec<LiveObservation>, i32, i32);
        {
            let mut feat = self.feat.write().unwrap();
            obs = feat
                .observations
                .drain()
                .filter_map(|(_, o)| o.kf.upgrade().map(|kf| (kf, (o.left, o.right))))
                .collect();
            feat.bad = true;
            n_visible = feat.visible;
            n_found = feat.found;
            feat.replaced = Some(other.clone());
        }

        for (kf, (left, right)) in obs {
            if !other.is_in_keyframe(&kf) {
                if left != -1 {
                    kf.replace_map_point_match(left as usize, other.clone());
                    other.add_observation(&kf, left);
                }
                if right != -1 {
                    kf.replace_map_point_match(right as usize, other.clone());
                    other.add_observation(&kf, right);
                }
            } else {
                if left != -1 {
                    kf.erase_map_point_match_idx(left as usize);
                }
                if right != -1 {
                    kf.erase_map_point_match_idx(right as usize);
                }
            }
        }
        other.increase_found(n_found);
        other.increase_visible(n_visible);
        other.compute_distinctive_descriptors();

        if let Some(map) = self.map.read().unwrap().upgrade() {
            map.erase_map_point(self);
        }
    }

    // Visible / found

    pub fn increase_visible(&self, n: i32) {
        self.feat.write().unwrap().visible += n;
    }
    pub fn increase_found(&self, n: i32) {
        self.feat.write().unwrap().found += n;
    }
    pub fn get_found(&self) -> i32 {
        self.feat.read().unwrap().found
    }
    pub fn get_found_ratio(&self) -> f32 {
        let feat = self.feat.read().unwrap();
        feat.found as f32 / feat.visible as f32
    }

    // Descriptor

    pub fn get_descriptor(&self) -> Mat {
        self.feat
            .read()
            .unwrap()
            .descriptor
            .try_clone()
            .unwrap_or_default()
    }

    /// Recompute the representative descriptor as the one with the least
    /// median Hamming distance to all observed descriptors.
    pub fn compute_distinctive_descriptors(&self) {
        let observations = {
            let feat = self.feat.read().unwrap();
            if feat.bad {
                return;
            }
            feat.observations
                .values()
                .filter_map(|o| o.kf.upgrade().map(|kf| (kf, o.left, o.right)))
                .collect::<Vec<_>>()
        };
        if observations.is_empty() {
            return;
        }

        let mut descriptors: Vec<Mat> = Vec::new();
        for (kf, left, right) in observations {
            if kf.is_bad() {
                continue;
            }
            if left != -1
                && let Ok(row) = kf.descriptors.row(left)
            {
                descriptors.push(row.try_clone().expect("clone row"));
            }
            if right != -1
                && let Ok(row) = kf.descriptors.row(right)
            {
                descriptors.push(row.try_clone().expect("clone row"));
            }
        }
        if descriptors.is_empty() {
            return;
        }

        let n = descriptors.len();
        let mut distances = vec![vec![0i32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = descriptor_distance(&descriptors[i], &descriptors[j]);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }

        let mut best_median = i32::MAX;
        let mut best_idx = 0usize;
        for (i, row) in distances.iter().enumerate() {
            let mut sorted = row.clone();
            sorted.sort_unstable();
            let median = sorted[(n - 1) / 2];
            if median < best_median {
                best_median = median;
                best_idx = i;
            }
        }

        self.feat.write().unwrap().descriptor =
            descriptors[best_idx].try_clone().expect("clone descriptor");
    }

    // Scale invariance

    // Recompute the mean viewing direction and scale-invariance distances
    // from all current observations.
    pub fn update_normal_and_depth(&self) {
        let (observations, ref_kf, pos) = {
            let feat = self.feat.read().unwrap();
            if feat.bad {
                return;
            }
            let observations = feat
                .observations
                .values()
                .filter_map(|o| o.kf.upgrade().map(|kf| (kf, o.left, o.right)))
                .collect::<Vec<_>>();
            (observations, feat.ref_kf.upgrade(), self.get_world_pos())
        };
        if observations.is_empty() {
            return;
        }
        let Some(ref_kf) = ref_kf else {
            return;
        };

        let mut normal = Vector3::zeros();
        let mut n = 0;
        for (kf, left, right) in &observations {
            if *left != -1 {
                let owi = kf.get_camera_center();
                let normali = pos - owi;
                normal += normali / normali.norm();
                n += 1;
            }
            if *right != -1 {
                let owi = kf.get_right_camera_center();
                let normali = pos - owi;
                normal += normali / normali.norm();
                n += 1;
            }
        }

        let pc = pos - ref_kf.get_camera_center();
        let dist = pc.norm();

        let (ref_left, ref_right) = self.get_index_in_keyframe(&ref_kf);
        let level = match ref_kf.n_left {
            None => ref_kf.keys_un[ref_left as usize].octave(),
            _ if ref_left != -1 => ref_kf.keys[ref_left as usize].octave(),
            _ => {
                let n_left = ref_kf.n_left.unwrap_or(0) as i32;
                ref_kf.keys_right.as_ref().expect("keys_right")[(ref_right - n_left) as usize]
                    .octave()
            }
        } as usize;
        let level_scale_factor = ref_kf.scale_factors[level];
        let n_levels = ref_kf.scale_levels;

        let mut pos_state = self.pos.write().unwrap();
        pos_state.max_distance = dist * level_scale_factor;
        pos_state.min_distance = pos_state.max_distance / ref_kf.scale_factors[n_levels - 1];
        if n > 0 {
            pos_state.normal_vector = normal / n as f32;
        }
    }

    pub fn get_min_distance_invariance(&self) -> f32 {
        0.8 * self.pos.read().unwrap().min_distance
    }
    pub fn get_max_distance_invariance(&self) -> f32 {
        1.2 * self.pos.read().unwrap().max_distance
    }

    pub fn predict_scale(&self, current_dist: f32, frame: &Frame) -> usize {
        let ratio = self.pos.read().unwrap().max_distance / current_dist;
        let scale = (ratio.ln() / frame.log_scale_factor).ceil() as i32;
        scale.clamp(0, frame.scale_levels as i32 - 1) as usize
    }

    pub fn predict_scale_keyframe(&self, current_dist: f32, kf: &KeyFrame) -> usize {
        let ratio = self.pos.read().unwrap().max_distance / current_dist;
        let scale = (ratio.ln() / kf.log_scale_factor).ceil() as i32;
        scale.clamp(0, kf.scale_levels as i32 - 1) as usize
    }

    pub fn get_map(&self) -> Option<Arc<Map>> {
        self.map.read().unwrap().upgrade()
    }
    pub fn update_map(&self, map: Arc<Map>) {
        *self.map.write().unwrap() = Arc::downgrade(&map);
    }
}

/// Serde-friendly snapshot of a [`MapPoint`].
#[derive(Serialize, Deserialize)]
pub(crate) struct MapPointSnapshot {
    pub(crate) id: usize,
    first_kf_id: Option<u64>,
    first_frame: usize,
    n_obs: i32,
    world_pos: Vector3<f32>,
    normal_vector: Vector3<f32>,
    /// `(observing keyframe id, left index)` — mirrors `mBackupObservationsId1`.
    obs_left: Vec<(u64, i32)>,
    /// `(observing keyframe id, right index)` — mirrors `mBackupObservationsId2`.
    obs_right: Vec<(u64, i32)>,
    descriptor: SerializableMat,
    ref_kf_id: Option<u64>,
    bad: bool,
    replaced_id: Option<usize>,
    min_distance: f32,
    max_distance: f32,
}

impl MapPoint {
    /// Capture this point's serializable state (`MapPoint::PreSave`).
    pub(crate) fn to_snapshot(&self) -> MapPointSnapshot {
        let pos = self.pos.read().unwrap();
        let feat = self.feat.read().unwrap();
        let mut obs_left = Vec::with_capacity(feat.observations.len());
        let mut obs_right = Vec::with_capacity(feat.observations.len());
        for (kf_id, o) in feat.observations.iter() {
            obs_left.push((kf_id.0, o.left));
            obs_right.push((kf_id.0, o.right));
        }
        MapPointSnapshot {
            id: self.id,
            first_kf_id: self.first_kf_id,
            first_frame: self.first_frame,
            n_obs: feat.n_obs,
            world_pos: pos.world_pos,
            normal_vector: pos.normal_vector,
            obs_left,
            obs_right,
            descriptor: SerializableMat::from_mat(&feat.descriptor)
                .expect("serialize map point descriptor"),
            ref_kf_id: feat.ref_kf.upgrade().map(|k| k.id),
            bad: feat.bad,
            replaced_id: feat.replaced.as_ref().map(|m| m.id),
            min_distance: pos.min_distance,
            max_distance: pos.max_distance,
        }
    }

    /// Build a link-less map point shell from a backup. Observation / reference
    /// / replaced links are wired separately in [`MapPoint::restore_links`].
    pub(crate) fn from_snapshot(b: &MapPointSnapshot, origin_map_id: u32) -> Arc<MapPoint> {
        let mp = Self::build(
            b.id,
            b.first_kf_id,
            b.first_frame,
            origin_map_id,
            b.world_pos,
            b.normal_vector,
            b.min_distance,
            b.max_distance,
            Weak::new(),
            b.descriptor
                .to_mat()
                .expect("deserialize map point descriptor"),
            None,
        );
        {
            let mut feat = mp.feat.write().unwrap();
            feat.n_obs = b.n_obs;
            feat.bad = b.bad;
        }
        Arc::new(mp)
    }

    /// Re-establish observation / reference / replaced links from ids
    /// (`MapPoint::PostLoad`).
    pub(crate) fn restore_links(
        &self,
        b: &MapPointSnapshot,
        kfs: &HashMap<u64, Arc<KeyFrame>>,
        mps: &HashMap<usize, Arc<MapPoint>>,
    ) {
        let mut feat = self.feat.write().unwrap();

        // Merge the left/right id maps back into single observations.
        let mut merged: HashMap<u64, (i32, i32)> = HashMap::new();
        for (id, left) in &b.obs_left {
            merged.entry(*id).or_insert((-1, -1)).0 = *left;
        }
        for (id, right) in &b.obs_right {
            merged.entry(*id).or_insert((-1, -1)).1 = *right;
        }
        for (kf_id, (left, right)) in merged {
            if let Some(kf) = kfs.get(&kf_id) {
                feat.observations.insert(
                    KeyFrameId(kf_id),
                    Observation {
                        kf: Arc::downgrade(kf),
                        left,
                        right,
                    },
                );
            }
        }

        if let Some(rid) = b.ref_kf_id
            && let Some(kf) = kfs.get(&rid)
        {
            feat.ref_kf = Arc::downgrade(kf);
        }
        if let Some(rep) = b.replaced_id
            && let Some(mp) = mps.get(&rep)
        {
            feat.replaced = Some(mp.clone());
        }
    }
}

impl Default for MapPoint {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    use std::sync::Arc;

    use nalgebra::Vector3;
    use opencv::core::{KeyPointTrait, MatTraitConst};

    use super::*;
    use crate::key_frame::KeyFrame;
    use crate::test_helpers::*;

    #[test]
    fn world_pos_and_normal() {
        let mp = Arc::new(MapPoint::new());
        mp.set_world_pos(Vector3::new(1.5, -2.5, 3.5));
        assert_vec(mp.get_world_pos(), [1.5, -2.5, 3.5], 1e-6);

        mp.set_normal_vector(Vector3::new(0.0, 0.0, 1.0));
        assert_vec(mp.get_normal(), [0.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn from_frame_distances_normal_scale() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        assert!(f.n > 0);
        f.keys_un.as_mut().unwrap()[0].set_octave(0);

        let map = Arc::new(Map::new());
        let cam_pt = Vector3::new(0.3, 0.1, 6.0);
        let world = f.get_rwc() * cam_pt + f.get_ow();
        let mp = MapPoint::from_frame(&world, map, &f, 0);

        assert!(approx(mp.get_max_distance_invariance(), 7.20999336, 1e-3));
        assert!(approx(mp.get_min_distance_invariance(), 1.34145093, 1e-3));
        assert_vec(
            mp.get_normal(),
            [-0.128525048, 0.128840521, 0.983301282],
            1e-4,
        );

        let dist = cam_pt.norm();
        assert_eq!(mp.predict_scale(dist / 1.5, &f), 3);
    }

    #[test]
    fn observations_add_count_index_erase() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map.clone());
        assert!(f.n > 5);

        f.keys_un.as_mut().unwrap()[0].set_octave(0);
        let world = f.get_rwc() * Vector3::new(0.0, 0.0, 5.0) + f.get_ow();
        let mp = MapPoint::from_frame(&world, map, &f, 0);

        assert_eq!(mp.observations(), 0);
        assert!(!mp.is_in_keyframe(&kf));

        mp.add_observation(&kf, 5);
        assert_eq!(mp.observations(), 1); // monocular -> +1
        assert!(mp.is_in_keyframe(&kf));
        assert_eq!(mp.get_index_in_keyframe(&kf), (5, -1));

        mp.erase_observation(&kf);
        assert_eq!(mp.observations(), 0);
        assert!(!mp.is_in_keyframe(&kf));
        assert_eq!(mp.get_index_in_keyframe(&kf), (-1, -1));
    }

    #[test]
    fn visible_found_counters() {
        let mp = Arc::new(MapPoint::new());
        assert_eq!(mp.get_found(), 1);
        assert!(approx(mp.get_found_ratio(), 1.0, 1e-6));

        mp.increase_visible(3); // visible = 4
        mp.increase_found(1); // found = 2
        assert_eq!(mp.get_found(), 2);
        assert!(approx(mp.get_found_ratio(), 2.0 / 4.0, 1e-6));
    }

    #[test]
    fn compute_distinctive_descriptors_single_obs() {
        let f = build_frame();
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);
        assert!(kf.n > 0);

        let mp = Arc::new(MapPoint::new());
        mp.add_observation(&kf, 0);
        mp.compute_distinctive_descriptors();

        let desc = mp.get_descriptor();
        assert_eq!(desc.rows(), 1);
        assert_eq!(desc.cols(), 32);
        let row0 = kf.descriptors.row(0).unwrap();
        let d =
            opencv::core::norm2(&desc, &row0, opencv::core::NORM_HAMMING, &Mat::default()).unwrap();
        assert_eq!(d, 0.0);
    }

    #[test]
    fn replace_transfers_observations_and_marks_bad() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map.clone());
        assert!(kf.n > 2);

        f.keys_un.as_mut().unwrap()[0].set_octave(0);
        let world = f.get_rwc() * Vector3::new(0.0, 0.0, 5.0) + f.get_ow();
        let mp1 = MapPoint::from_frame(&world, map.clone(), &f, 0);
        let mp2 = MapPoint::from_frame(&world, map, &f, 2);

        mp1.add_observation(&kf, 1);
        kf.add_map_point(mp1.clone(), 1);

        assert!(!mp1.is_bad());
        mp1.replace(&mp2);

        assert!(mp1.is_bad());
        assert_eq!(mp1.get_replaced().unwrap().id, mp2.id);
        assert!(mp2.is_in_keyframe(&kf));
        assert_eq!(kf.get_map_point(1).unwrap().id, mp2.id);
    }
}
