//! [`Map`]: the set of [`KeyFrame`]s and [`MapPoint`]s that make up one
//! reconstruction, plus the bookkeeping flags shared across Tracking,
//! LocalMapping and LoopClosing.
//!
//! ## Threading model
//!
//! A `Map` is shared as `Arc<Map>` and mutated concurrently. Identity (`id`)
//! lives in an [`AtomicU32`] so `get_id` stays cheap; everything else lives
//! behind a single `RwLock<MapInner>`, mirroring the one `std::mutex` of the
//! C++ class.
//!
//! ## Ownership
//!
//! The map is the *strong* owner of its keyframes and map points (it stores
//! `Arc<KeyFrame>` / `Arc<MapPoint>` keyed by id, where the original C++ keeps
//! raw `std::set` pointers). The keyframe/map-point graph edges that point back
//! at a `Map` (or at each other) are [`std::sync::Weak`] to avoid leaking,
//! except for `KeyFrame::map` / `MapPoint::map`, whose `Arc<Map>` back-pointer
//! forms a cycle that is a deferred-cleanup concern — matching the C++ `~Map`
//! `TODO: erase all points from memory`.
//!
//! Cross-object operations (`apply_scaled_rotation`, `clear`) follow the same
//! discipline as [`crate::map_point`]: snapshot the membership under the lock,
//! release it, then call into the keyframes / map points.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use nalgebra::Isometry3;

use serde::{Deserialize, Serialize};

use crate::camera_models::GeometricCamera;
use crate::camera_models::pinhole::Pinhole;
use crate::key_frame::{KeyFrame, KeyFrameSnapshot};
use crate::key_frame_database::{KeyFrameDatabase, KeyFrameId};
use crate::map_point::{MapPoint, MapPointSnapshot};
use crate::orb_vocabulary::OrbVocabulary;

static NEXT_MAP_ID: AtomicU32 = AtomicU32::new(0);

/// Current value of the global map-id counter (for serialization).
pub(crate) fn peek_next_map_id() -> u32 {
    NEXT_MAP_ID.load(Ordering::SeqCst)
}
/// Restore the global map-id counter (after deserialization).
pub(crate) fn set_next_map_id(v: u32) {
    NEXT_MAP_ID.store(v, Ordering::SeqCst);
}

/// Thumbnail size used by the atlas viewer (always a power of 2).
pub const THUMB_WIDTH: usize = 512;
pub const THUMB_HEIGHT: usize = 512;

/// Mutable membership and flags, guarded by [`Map::inner`].
#[derive(Default)]
struct MapInner {
    /// All map points, keyed by [`MapPoint::id`].
    map_points: HashMap<usize, Arc<MapPoint>>,
    /// All keyframes, keyed by [`KeyFrame::id`].
    key_frames: HashMap<u64, Arc<KeyFrame>>,
    /// Reference map points (those drawn / used for relocalisation).
    reference_map_points: Vec<Arc<MapPoint>>,

    /// First keyframe inserted, i.e. the map origin.
    kf_initial: Option<Arc<KeyFrame>>,
    /// Keyframe with the lowest id currently in the map
    kf_lower: Option<Arc<KeyFrame>>,

    /// ID of the first keyframe of this map
    init_kf_id: u64,
    /// Largest keyframe ID seen so far
    max_kf_id: u64,

    /// Index bumped on every big change (loop closure / global BA)
    big_change_idx: u32,
    /// Local map-change counter and the last value notified to the viewer
    map_change: u32,
    map_change_notified: u32,

    imu_initialized: bool,
    is_inertial: bool,
    imu_ba1: bool,
    imu_ba2: bool,

    is_in_use: bool,
    has_thumbnail: bool,
    bad: bool,
}

/// Serde-friendly snapshot of a [`Map`].
#[derive(Serialize, Deserialize)]
pub(crate) struct MapSnapshot {
    id: u32,
    init_kf_id: u64,
    max_kf_id: u64,
    big_change_idx: u32,
    keyframes: Vec<KeyFrameSnapshot>,
    map_points: Vec<MapPointSnapshot>,
    kf_origins_id: Vec<u64>,
    kf_initial_id: Option<u64>,
    kf_lower_id: Option<u64>,
    imu_initialized: bool,
    is_inertial: bool,
    imu_ba1: bool,
    imu_ba2: bool,
}

/// One reconstruction: its keyframes, map points and shared state
pub struct Map {
    id: AtomicU32,
    inner: RwLock<MapInner>,

    /// Keyframes that seeded this map
    key_frame_origins: RwLock<Vec<Arc<KeyFrame>>>,
    /// First keyframe of the active region
    first_region_kf: RwLock<Option<Arc<KeyFrame>>>,

    /// Held by LocalMapping/LoopClosing around multi-step map edits
    pub map_update: Mutex<()>,
    /// Serialises map-point creation so two threads cannot mint the same id
    pub point_creation: Mutex<()>,
}

impl Hash for Map {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get_id().hash(state);
    }
}
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.get_id() == other.get_id()
    }
}
impl Eq for Map {}

impl Default for Map {
    fn default() -> Self {
        Self::new()
    }
}

impl Map {
    /// Create a new map with a fresh, process-unique id
    pub fn new() -> Self {
        Map {
            id: AtomicU32::new(NEXT_MAP_ID.fetch_add(1, Ordering::SeqCst)),
            inner: RwLock::new(MapInner::default()),
            key_frame_origins: RwLock::new(Vec::new()),
            first_region_kf: RwLock::new(None),
            map_update: Mutex::new(()),
            point_creation: Mutex::new(()),
        }
    }

    /// Create a new map whose initial keyframe id is `init_kf_id`
    pub fn with_init_kf_id(init_kf_id: u64) -> Self {
        let map = Map::new();
        {
            let mut inner = map.inner.write().unwrap();
            inner.init_kf_id = init_kf_id;
            inner.max_kf_id = init_kf_id;
        }
        map
    }

    // Identity

    pub fn get_id(&self) -> u32 {
        self.id.load(Ordering::SeqCst)
    }

    /// Reassign the map id (used when merging maps in the atlas).
    pub fn change_id(&self, id: u32) {
        self.id.store(id, Ordering::SeqCst);
    }

    // Keyframes

    /// Insert a keyframe, updating the origin/lowest-id/max-id bookkeeping.
    pub fn add_key_frame(&self, kf: Arc<KeyFrame>) {
        let mut inner = self.inner.write().unwrap();
        if inner.key_frames.is_empty() {
            inner.init_kf_id = kf.id;
            inner.kf_initial = Some(kf.clone());
            inner.kf_lower = Some(kf.clone());
        }
        if kf.id > inner.max_kf_id {
            inner.max_kf_id = kf.id;
        }
        if inner
            .kf_lower
            .as_ref()
            .is_some_and(|lower| kf.id < lower.id)
        {
            inner.kf_lower = Some(kf.clone());
        }
        inner.key_frames.insert(kf.id, kf);
    }

    /// Remove a keyframe; recompute the lowest-id keyframe if it was erased.
    pub fn erase_key_frame(&self, kf: &KeyFrame) {
        let mut inner = self.inner.write().unwrap();
        inner.key_frames.remove(&kf.id);
        if inner.key_frames.is_empty() {
            inner.kf_lower = None;
        } else if inner
            .kf_lower
            .as_ref()
            .is_some_and(|lower| lower.id == kf.id)
        {
            inner.kf_lower = inner.key_frames.values().min_by_key(|k| k.id).cloned();
        }
        // NOTE: this only detaches the pointer; deleting the keyframe is a
        // deferred concern (matching the C++ TODO).
    }

    pub fn get_all_key_frames(&self) -> Vec<Arc<KeyFrame>> {
        self.inner
            .read()
            .unwrap()
            .key_frames
            .values()
            .cloned()
            .collect()
    }

    /// Number of keyframes currently registered with the map.
    pub fn key_frames_in_map(&self) -> usize {
        self.inner.read().unwrap().key_frames.len()
    }

    /// The map origin keyframe
    pub fn get_origin_kf(&self) -> Option<Arc<KeyFrame>> {
        self.inner.read().unwrap().kf_initial.clone()
    }

    /// Id of the lowest-id keyframe still in the map (`0` if empty).
    pub fn get_lower_kf_id(&self) -> u64 {
        self.inner
            .read()
            .unwrap()
            .kf_lower
            .as_ref()
            .map(|kf| kf.id)
            .unwrap_or(0)
    }

    pub fn get_init_kf_id(&self) -> u64 {
        self.inner.read().unwrap().init_kf_id
    }

    pub fn set_init_kf_id(&self, init_kf_id: u64) {
        self.inner.write().unwrap().init_kf_id = init_kf_id;
    }

    pub fn get_max_kf_id(&self) -> u64 {
        self.inner.read().unwrap().max_kf_id
    }

    // --- Map points -----------------------------------------------------

    pub fn add_map_point(&self, mp: Arc<MapPoint>) {
        self.inner.write().unwrap().map_points.insert(mp.id, mp);
    }

    /// Remove a map point from the map (only detaches the pointer).
    pub fn erase_map_point(&self, mp: &MapPoint) {
        self.inner.write().unwrap().map_points.remove(&mp.id);
    }

    pub fn get_all_map_points(&self) -> Vec<Arc<MapPoint>> {
        self.inner
            .read()
            .unwrap()
            .map_points
            .values()
            .cloned()
            .collect()
    }

    /// Number of map points currently registered with the map.
    pub fn map_points_in_map(&self) -> usize {
        self.inner.read().unwrap().map_points.len()
    }

    pub fn set_reference_map_points(&self, mps: Vec<Arc<MapPoint>>) {
        self.inner.write().unwrap().reference_map_points = mps;
    }

    pub fn get_reference_map_points(&self) -> Vec<Arc<MapPoint>> {
        self.inner.read().unwrap().reference_map_points.clone()
    }

    // Keyframe origins / region

    pub fn add_key_frame_origin(&self, kf: Arc<KeyFrame>) {
        self.key_frame_origins.write().unwrap().push(kf);
    }

    pub fn get_key_frame_origins(&self) -> Vec<Arc<KeyFrame>> {
        self.key_frame_origins.read().unwrap().clone()
    }

    pub fn get_first_region_kf(&self) -> Option<Arc<KeyFrame>> {
        self.first_region_kf.read().unwrap().clone()
    }

    pub fn set_first_region_kf(&self, kf: Option<Arc<KeyFrame>>) {
        *self.first_region_kf.write().unwrap() = kf;
    }

    // Big change / map change tracking

    pub fn inform_new_big_change(&self) {
        self.inner.write().unwrap().big_change_idx += 1;
    }

    pub fn get_last_big_change_idx(&self) -> u32 {
        self.inner.read().unwrap().big_change_idx
    }

    pub fn get_map_change_index(&self) -> u32 {
        self.inner.read().unwrap().map_change
    }

    pub fn increase_change_index(&self) {
        self.inner.write().unwrap().map_change += 1;
    }

    pub fn get_last_map_change(&self) -> u32 {
        self.inner.read().unwrap().map_change_notified
    }

    pub fn set_last_map_change(&self, current_change_id: u32) {
        self.inner.write().unwrap().map_change_notified = current_change_id;
    }

    /// Prepare the map for serialization (`Map::PreSave`, graph-maintenance part).
    ///
    /// Every observation that references a keyframe belonging to a *different*
    /// map (or a bad keyframe) is dropped, so the snapshot produced by
    /// [`Map::to_snapshot`] is self-contained. The pointer→id conversion itself is
    /// done lazily by `to_snapshot`, so there is no separate backup-vector step
    /// (the C++ `mvpBackupKeyFrames` / `mvpBackupMapPoints`).
    pub fn pre_save(&self, _cams: &[Arc<dyn GeometricCamera>]) {
        let this_id = self.get_id();
        for mp in self.get_all_map_points() {
            if mp.is_bad() {
                continue;
            }
            for (kf, _) in mp.get_observations() {
                let foreign = kf.get_map().map(|m| m.get_id()) != Some(this_id);
                if foreign || kf.is_bad() {
                    mp.erase_observation(&kf);
                }
            }
        }
    }

    /// Finalize a freshly deserialized map (`Map::PostLoad`, external-resource part).
    ///
    /// The keyframe / map-point graph and every back-pointer are already wired
    /// by [`Map::from_snapshot`]; this step only registers each keyframe with the
    /// keyframe database (C++ `pKFDB->add(pKFi)`). The ORB vocabulary is accepted
    /// for parity with the C++ signature but is already embedded in the database.
    pub fn post_load(
        &self,
        key_frame_database: Arc<KeyFrameDatabase>,
        _orb_voc: Arc<OrbVocabulary>,
    ) {
        for kf in self.get_all_key_frames() {
            key_frame_database.add(KeyFrameId(kf.id), &kf.bow_vec);
        }
    }

    /// Capture this map's full serializable state (`Map::PreSave` snapshot).
    ///
    /// Bad keyframes and map points are skipped; the rest are converted to their
    /// id-based [`KeyFrameSnapshot`] / [`MapPointSnapshot`] forms.
    pub(crate) fn to_snapshot(&self) -> MapSnapshot {
        let inner = self.inner.read().unwrap();
        let keyframes = inner
            .key_frames
            .values()
            .filter(|k| !k.is_bad())
            .map(|k| k.to_snapshot())
            .collect();
        let map_points = inner
            .map_points
            .values()
            .filter(|m| !m.is_bad())
            .map(|m| m.to_snapshot())
            .collect();
        let kf_origins_id = self
            .key_frame_origins
            .read()
            .unwrap()
            .iter()
            .map(|k| k.id)
            .collect();
        MapSnapshot {
            id: self.get_id(),
            init_kf_id: inner.init_kf_id,
            max_kf_id: inner.max_kf_id,
            big_change_idx: inner.big_change_idx,
            keyframes,
            map_points,
            kf_origins_id,
            kf_initial_id: inner.kf_initial.as_ref().map(|k| k.id),
            kf_lower_id: inner.kf_lower.as_ref().map(|k| k.id),
            imu_initialized: inner.imu_initialized,
            is_inertial: inner.is_inertial,
            imu_ba1: inner.imu_ba1,
            imu_ba2: inner.imu_ba2,
        }
    }

    /// Reconstruct a live map from a backup (`Map::PostLoad` pointer rebuild).
    ///
    /// Keyframe / map-point shells are built first, then their links (and the
    /// `Weak<Map>` back-pointers) are wired up, mirroring ORB-SLAM3's two-pass
    /// `PostLoad`.
    pub(crate) fn from_snapshot(
        b: &MapSnapshot,
        cams: &HashMap<u64, Arc<dyn GeometricCamera>>,
    ) -> Arc<Map> {
        let map = Map::new();
        map.change_id(b.id);
        {
            let mut inner = map.inner.write().unwrap();
            inner.init_kf_id = b.init_kf_id;
            inner.max_kf_id = b.max_kf_id;
            inner.big_change_idx = b.big_change_idx;
            inner.imu_initialized = b.imu_initialized;
            inner.is_inertial = b.is_inertial;
            inner.imu_ba1 = b.imu_ba1;
            inner.imu_ba2 = b.imu_ba2;
        }
        let map = Arc::new(map);

        // Resolve a camera id against the loaded camera set, falling back to any
        // available camera (and finally a default) so reconstruction never panics.
        let resolve_cam = |id: Option<u64>| -> Arc<dyn GeometricCamera> {
            id.and_then(|id| cams.get(&id).cloned())
                .or_else(|| cams.values().next().cloned())
                .unwrap_or_else(|| Arc::new(Pinhole::new()))
        };

        // Phase 1: build link-less shells, indexed by id.
        let mut kfs: HashMap<u64, Arc<KeyFrame>> = HashMap::new();
        for kb in &b.keyframes {
            let camera = resolve_cam(kb.camera_id);
            let camera2 = kb.camera2_id.and_then(|id| cams.get(&id).cloned());
            let kf = KeyFrame::from_snapshot(kb, camera, camera2);
            kfs.insert(kf.id, kf);
        }
        let mut mps: HashMap<usize, Arc<MapPoint>> = HashMap::new();
        for mb in &b.map_points {
            let mp = MapPoint::from_snapshot(mb, b.id);
            mps.insert(mp.id, mp);
        }

        // Phase 2: wire links and re-point the map back-pointers.
        for kb in &b.keyframes {
            if let Some(kf) = kfs.get(&kb.id) {
                kf.restore_links(kb, &kfs, &mps);
                kf.update_map(map.clone());
            }
        }
        for mb in &b.map_points {
            if let Some(mp) = mps.get(&mb.id) {
                mp.restore_links(mb, &kfs, &mps);
                mp.update_map(map.clone());
            }
        }

        // Register everything with the map.
        {
            let mut inner = map.inner.write().unwrap();
            inner.key_frames = kfs.clone();
            inner.map_points = mps.clone();
            inner.kf_initial = b.kf_initial_id.and_then(|id| kfs.get(&id).cloned());
            inner.kf_lower = b.kf_lower_id.and_then(|id| kfs.get(&id).cloned());
        }
        {
            let mut origins = map.key_frame_origins.write().unwrap();
            for id in &b.kf_origins_id {
                if let Some(kf) = kfs.get(id) {
                    origins.push(kf.clone());
                }
            }
        }
        map
    }

    // IMU / inertial flags

    pub fn is_imu_initialized(&self) -> bool {
        self.inner.read().unwrap().imu_initialized
    }

    pub fn set_imu_initialized(&self) {
        self.inner.write().unwrap().imu_initialized = true;
    }

    pub fn is_inertial(&self) -> bool {
        self.inner.read().unwrap().is_inertial
    }

    pub fn set_inertial_sensor(&self) {
        self.inner.write().unwrap().is_inertial = true;
    }

    pub fn get_inertial_ba1(&self) -> bool {
        self.inner.read().unwrap().imu_ba1
    }

    pub fn set_inertial_ba1(&self) {
        self.inner.write().unwrap().imu_ba1 = true;
    }

    pub fn get_inertial_ba2(&self) -> bool {
        self.inner.read().unwrap().imu_ba2
    }

    pub fn set_inertial_ba2(&self) {
        self.inner.write().unwrap().imu_ba2 = true;
    }

    // Lifecycle flags

    pub fn set_current_map(&self) {
        self.inner.write().unwrap().is_in_use = true;
    }

    pub fn set_stored_map(&self) {
        self.inner.write().unwrap().is_in_use = false;
    }

    pub fn is_in_use(&self) -> bool {
        self.inner.read().unwrap().is_in_use
    }

    pub fn has_thumbnail(&self) -> bool {
        self.inner.read().unwrap().has_thumbnail
    }

    pub fn set_bad(&self) {
        self.inner.write().unwrap().bad = true;
    }

    pub fn is_bad(&self) -> bool {
        self.inner.read().unwrap().bad
    }

    // Bulk operations

    /// Apply a similarity transform `T` (with scale `s`) to every keyframe and
    /// map point. When `scaled_vel` is set the keyframe velocities are scaled
    /// as well (`ApplyScaledRotation`).
    pub fn apply_scaled_rotation(&self, t: Isometry3<f32>, s: f32, scaled_vel: bool) {
        let ryw = t.rotation.to_rotation_matrix().into_inner();
        let tyw = t.translation.vector;

        // Snapshot membership under the lock, then operate without holding it.
        let (key_frames, map_points) = {
            let inner = self.inner.read().unwrap();
            (
                inner.key_frames.values().cloned().collect::<Vec<_>>(),
                inner.map_points.values().cloned().collect::<Vec<_>>(),
            )
        };

        for kf in key_frames {
            let mut twc = kf.get_pose_inverse();
            twc.translation.vector *= s;
            let tcy = (t * twc).inverse();
            kf.set_pose(tcy);
            let vw = kf.get_velocity();
            kf.set_velocity(if scaled_vel { ryw * vw * s } else { ryw * vw });
        }

        for mp in map_points {
            mp.set_world_pos(s * (ryw * mp.get_world_pos()) + tyw);
            mp.update_normal_and_depth();
        }

        self.inner.write().unwrap().map_change += 1;
    }

    /// Empty the map and reset its IMU/scale state, keeping the init id.
    pub fn clear(&self) {
        // NOTE: the C++ `clear` also calls `pKF->UpdateMap(nullptr)` on every
        // keyframe; `KeyFrame::update_map` cannot yet detach a map, so that
        // step is deferred.
        let mut inner = self.inner.write().unwrap();
        inner.map_points.clear();
        inner.key_frames.clear();
        inner.max_kf_id = inner.init_kf_id;
        inner.imu_initialized = false;
        inner.reference_map_points.clear();
        inner.kf_initial = None;
        inner.kf_lower = None;
        inner.imu_ba1 = false;
        inner.imu_ba2 = false;
        drop(inner);
        self.key_frame_origins.write().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    use super::*;
    use crate::key_frame::KeyFrame;
    use crate::map_point::MapPoint;
    use crate::test_helpers::build_frame;

    fn make_kf(map: &Arc<Map>) -> Arc<KeyFrame> {
        KeyFrame::from_frame(&build_frame(), map.clone())
    }

    #[test]
    fn keyframe_bookkeeping() {
        let map = Arc::new(Map::new());
        let a = make_kf(&map);
        let b = make_kf(&map);
        assert!(a.id < b.id);

        map.add_key_frame(b.clone());
        map.add_key_frame(a.clone());

        assert_eq!(map.key_frames_in_map(), 2);
        // Origin is the first one inserted (b), lower-id is a, max is b.
        assert_eq!(map.get_origin_kf().unwrap().id, b.id);
        assert_eq!(map.get_lower_kf_id(), a.id);
        assert_eq!(map.get_max_kf_id(), b.id);
        assert_eq!(map.get_init_kf_id(), b.id);

        // Erasing the lowest-id keyframe re-elects the next lowest.
        map.erase_key_frame(&a);
        assert_eq!(map.key_frames_in_map(), 1);
        assert_eq!(map.get_lower_kf_id(), b.id);

        map.erase_key_frame(&b);
        assert_eq!(map.get_lower_kf_id(), 0);
    }

    #[test]
    fn map_point_membership() {
        let map = Arc::new(Map::new());
        let mp = Arc::new(MapPoint::new());

        map.add_map_point(mp.clone());
        assert_eq!(map.map_points_in_map(), 1);
        assert_eq!(map.get_all_map_points()[0].id, mp.id);

        map.erase_map_point(&mp);
        assert_eq!(map.map_points_in_map(), 0);

        map.set_reference_map_points(vec![mp.clone()]);
        assert_eq!(map.get_reference_map_points().len(), 1);
    }

    #[test]
    fn flags_and_counters() {
        let map = Map::new();

        assert!(!map.is_imu_initialized());
        map.set_imu_initialized();
        assert!(map.is_imu_initialized());

        assert!(!map.is_inertial());
        map.set_inertial_sensor();
        assert!(map.is_inertial());

        map.set_inertial_ba1();
        map.set_inertial_ba2();
        assert!(map.get_inertial_ba1());
        assert!(map.get_inertial_ba2());

        assert_eq!(map.get_last_big_change_idx(), 0);
        map.inform_new_big_change();
        assert_eq!(map.get_last_big_change_idx(), 1);

        assert_eq!(map.get_map_change_index(), 0);
        map.increase_change_index();
        assert_eq!(map.get_map_change_index(), 1);
        map.set_last_map_change(7);
        assert_eq!(map.get_last_map_change(), 7);

        assert!(!map.is_in_use());
        map.set_current_map();
        assert!(map.is_in_use());
        map.set_stored_map();
        assert!(!map.is_in_use());

        assert!(!map.is_bad());
        map.set_bad();
        assert!(map.is_bad());
    }

    #[test]
    fn change_id_and_init_kf() {
        let map = Map::with_init_kf_id(42);
        assert_eq!(map.get_init_kf_id(), 42);
        assert_eq!(map.get_max_kf_id(), 42);

        let id = map.get_id();
        map.change_id(id + 100);
        assert_eq!(map.get_id(), id + 100);

        map.set_init_kf_id(5);
        assert_eq!(map.get_init_kf_id(), 5);
    }

    #[test]
    fn clear_resets_state() {
        let map = Arc::new(Map::with_init_kf_id(3));
        map.add_key_frame(make_kf(&map));
        map.add_map_point(Arc::new(MapPoint::new()));
        map.set_imu_initialized();
        map.set_inertial_ba1();

        map.clear();

        assert_eq!(map.key_frames_in_map(), 0);
        assert_eq!(map.map_points_in_map(), 0);
        // `clear` resets the max keyframe id back to the (current) init id.
        assert_eq!(map.get_max_kf_id(), map.get_init_kf_id());
        assert!(!map.is_imu_initialized());
        assert!(!map.get_inertial_ba1());
        assert!(map.get_origin_kf().is_none());
    }

    #[test]
    fn apply_scaled_rotation_scales_points() {
        let map = Arc::new(Map::new());
        let mp = Arc::new(MapPoint::new());
        mp.set_world_pos(Vector3::new(1.0, 0.0, 0.0));
        map.add_map_point(mp.clone());

        // 90° about Z, scale 2, no translation.
        let t = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f32::consts::FRAC_PI_2),
        );
        map.apply_scaled_rotation(t, 2.0, false);

        let p = mp.get_world_pos();
        assert!((p.x - 0.0).abs() < 1e-5, "x={}", p.x);
        assert!((p.y - 2.0).abs() < 1e-5, "y={}", p.y);
        assert_eq!(map.get_map_change_index(), 1);
    }
}
