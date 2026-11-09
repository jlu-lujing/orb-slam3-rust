use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tracing::info;

use crate::{
    camera_models::{GeometricCamera, GeometricCameraSnapshot},
    key_frame::KeyFrame,
    key_frame_database::KeyFrameDatabase,
    map::{Map, MapSnapshot},
    map_point::MapPoint,
    orb_vocabulary::OrbVocabulary,
    viewer::Viewer,
};

/// Serde-friendly snapshot of an [`Atlas`].
#[derive(Serialize, Deserialize)]
struct AtlasSnapshot {
    maps: Vec<MapSnapshot>,
    cameras: Vec<GeometricCameraSnapshot>,
    last_init_kf_id_map: u64,
    next_map_id: u32,
    next_frame_id: usize,
    next_key_frame_id: u64,
    next_map_point_id: usize,
    next_camera_id: u64,
}

impl Serialize for Atlas {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let inner = self.inner.lock().unwrap();
        let maps = inner
            .maps
            .values()
            .filter(|m| !m.is_bad())
            .map(|m| m.to_snapshot())
            .collect();
        let cameras = inner
            .cameras
            .iter()
            .filter_map(|c| GeometricCameraSnapshot::try_from(c.as_ref()).ok())
            .collect();
        AtlasSnapshot {
            maps,
            cameras,
            last_init_kf_id_map: inner.last_init_kf_id_map,
            next_map_id: crate::map::peek_next_map_id(),
            next_frame_id: crate::frame::peek_next_frame_id(),
            next_key_frame_id: crate::key_frame::peek_next_id(),
            next_map_point_id: crate::map_point::peek_next_id(),
            next_camera_id: crate::camera_models::peek_next_camera_id(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Atlas {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let snapshot = AtlasSnapshot::deserialize(deserializer)?;

        // Restore the global id counters so newly minted ids never collide with
        // loaded ones.
        crate::map::set_next_map_id(snapshot.next_map_id);
        crate::frame::set_next_frame_id(snapshot.next_frame_id);
        crate::key_frame::set_next_id(snapshot.next_key_frame_id);
        crate::map_point::set_next_id(snapshot.next_map_point_id);
        crate::camera_models::set_next_camera_id(snapshot.next_camera_id);

        let cameras: Vec<Arc<dyn GeometricCamera>> = snapshot
            .cameras
            .into_iter()
            .map(|c| Arc::from(c.into_boxed()))
            .collect();
        let cam_map: HashMap<u64, Arc<dyn GeometricCamera>> =
            cameras.iter().map(|c| (c.get_id(), c.clone())).collect();

        let mut maps = HashMap::new();
        for mb in &snapshot.maps {
            let map = Map::from_snapshot(mb, &cam_map);
            maps.insert(map.get_id(), map);
        }

        Ok(Atlas {
            inner: Mutex::new(AtlasInner {
                maps,
                bad_maps: HashMap::new(),
                current_map: None,
                cameras,
                last_init_kf_id_map: snapshot.last_init_kf_id_map,
            }),
            viewer: OnceLock::new(),
            key_frame_database: OnceLock::new(),
            orb_vocabulary: OnceLock::new(),
        })
    }
}

/// Magic bytes prefixing a serialized atlas blob (`"ORBA"`).
const ATLAS_MAGIC: [u8; 4] = *b"ORBA";
/// On-disk schema version. Bump whenever a `*Snapshot` layout changes so that
/// older blobs fail loudly instead of decoding to garbage (the wire format is
/// not self-describing).
const ATLAS_FORMAT_VERSION: u32 = 1;

/// Error returned by [`Atlas::to_bytes`] / [`Atlas::from_bytes`].
#[derive(Debug)]
pub enum AtlasIoError {
    /// The blob is shorter than the 8-byte header.
    Truncated,
    /// The leading magic bytes did not match [`ATLAS_MAGIC`].
    BadMagic,
    /// The header version is not [`ATLAS_FORMAT_VERSION`].
    UnsupportedVersion(u32),
    /// The postcard codec failed.
    Postcard(postcard::Error),
}

impl std::fmt::Display for AtlasIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtlasIoError::Truncated => write!(f, "atlas blob is truncated"),
            AtlasIoError::BadMagic => write!(f, "atlas blob has invalid magic"),
            AtlasIoError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported atlas format version {v} (expected {ATLAS_FORMAT_VERSION})"
                )
            }
            AtlasIoError::Postcard(e) => write!(f, "postcard codec error: {e}"),
        }
    }
}
impl std::error::Error for AtlasIoError {}

pub struct Atlas {
    /// All mutable atlas state lives behind a single mutex
    /// This lets the atlas be shared as `Arc<Atlas>` across the
    /// tracking / local-mapping / loop-closing threads while every method takes
    /// `&self`. Critical sections are intentionally short: the heavy lifting
    /// happens inside `Map`, which carries its own interior locks.
    inner: Mutex<AtlasInner>,

    /// Set-once dependency wiring, injected after construction and never
    /// cleared. Kept outside `inner` so reads are lock-free and don't contend
    /// on the atlas mutex (`OnceLock` instead of a mutex-guarded `Option`).
    viewer: OnceLock<Arc<Viewer>>,
    // Class references for the map reconstruction from the save field
    key_frame_database: OnceLock<Arc<KeyFrameDatabase>>,
    orb_vocabulary: OnceLock<Arc<OrbVocabulary>>,
}

struct AtlasInner {
    maps: HashMap<u32, Arc<Map>>,

    bad_maps: HashMap<u32, Arc<Map>>,

    current_map: Option<Arc<Map>>,

    cameras: Vec<Arc<dyn GeometricCamera>>,

    last_init_kf_id_map: u64,
}

impl AtlasInner {
    fn new() -> Self {
        AtlasInner {
            maps: HashMap::new(),
            bad_maps: HashMap::new(),
            current_map: None,
            cameras: Vec::new(),
            last_init_kf_id_map: 0,
        }
    }
}

impl Default for Atlas {
    fn default() -> Self {
        Self::new()
    }
}

impl Atlas {
    pub fn new() -> Self {
        Atlas {
            inner: Mutex::new(AtlasInner::new()),
            viewer: OnceLock::new(),
            key_frame_database: OnceLock::new(),
            orb_vocabulary: OnceLock::new(),
        }
    }

    pub fn from_kf_id(id: u64) -> Self {
        let r = Atlas::new();
        r.inner.lock().unwrap().last_init_kf_id_map = id;
        r.create_new_map();
        r
    }

    /// Clone the current map out under a short-lived lock. Used by the accessors
    /// that only operate on the current map, so the atlas lock is never held
    /// while doing actual work inside `Map`.
    fn current_map(&self) -> Option<Arc<Map>> {
        self.inner.lock().unwrap().current_map.clone()
    }

    pub fn create_new_map(&self) {
        let mut inner = self.inner.lock().unwrap();

        // Clone (Arc) so the borrow on `inner.current_map` is released before we
        // mutate `inner.last_init_kf_id_map` through the same guard.
        if let Some(current_map) = inner.current_map.clone() {
            let max = current_map.get_max_kf_id();
            if !inner.maps.is_empty() && inner.last_init_kf_id_map < max {
                inner.last_init_kf_id_map = max + 1;
            }

            current_map.set_stored_map();
            info!("Stored map with ID: {}", current_map.get_id());
        }

        info!(
            "Creation of new map with last KF id: {}",
            inner.last_init_kf_id_map
        );

        let new_map = Map::with_init_kf_id(inner.last_init_kf_id_map);
        new_map.set_current_map();
        let id = new_map.get_id();
        info!("Creation of new map with ID: {}", id);
        let m = Arc::new(new_map);
        inner.current_map = Some(m.clone());
        inner.maps.insert(id, m);
    }

    pub fn get_last_init_kf_id(&self) -> u64 {
        self.inner.lock().unwrap().last_init_kf_id_map
    }

    pub fn set_viewer(&self, viewer: Arc<Viewer>) {
        // Set-once wiring; ignore a second set (mirrors upstream's single call).
        let _ = self.viewer.set(viewer);
    }

    // Method to change components in the current map
    pub fn add_key_frame(&self, kf: Arc<KeyFrame>) {
        if let Some(map) = kf.get_map() {
            map.add_key_frame(kf);
        }
    }

    pub fn add_map_point(&self, mp: Arc<MapPoint>) {
        if let Some(map) = mp.get_map() {
            map.add_map_point(mp);
        }
    }

    pub fn add_camera(&self, cam: Arc<dyn GeometricCamera>) -> Arc<dyn GeometricCamera> {
        let mut inner = self.inner.lock().unwrap();
        // Check if the camera already exists
        if let Some(i) = inner.cameras.iter().position(|c| cam.is_equal(c)) {
            inner.cameras[i].clone()
        } else {
            inner.cameras.push(cam.clone());
            cam
        }
    }

    pub fn get_all_cameras(&self) -> Vec<Arc<dyn GeometricCamera>> {
        self.inner.lock().unwrap().cameras.clone()
    }

    // All methods without Map pointer work on current map
    pub fn set_reference_map_points(&self, mps: Vec<Arc<MapPoint>>) {
        if let Some(current_map) = self.current_map() {
            current_map.set_reference_map_points(mps);
        }
    }
    pub fn inform_new_big_change(&self) {
        if let Some(current_map) = self.current_map() {
            current_map.inform_new_big_change();
        }
    }
    pub fn get_last_big_change_idx(&self) -> Option<u32> {
        self.current_map().map(|m| m.get_last_big_change_idx())
    }

    pub fn map_points_in_map(&self) -> Option<usize> {
        self.current_map().map(|m| m.map_points_in_map())
    }
    pub fn key_frames_in_map(&self) -> Option<usize> {
        self.current_map().map(|m| m.key_frames_in_map())
    }

    // Method for get data in current map
    pub fn get_all_key_frames(&self) -> Option<Vec<Arc<KeyFrame>>> {
        self.current_map().map(|m| m.get_all_key_frames())
    }
    pub fn get_all_map_points(&self) -> Option<Vec<Arc<MapPoint>>> {
        self.current_map().map(|m| m.get_all_map_points())
    }
    pub fn get_reference_map_points(&self) -> Option<Vec<Arc<MapPoint>>> {
        self.current_map().map(|m| m.get_reference_map_points())
    }

    pub fn get_all_maps(&self) -> Vec<Arc<Map>> {
        let mut maps: Vec<_> = self.inner.lock().unwrap().maps.values().cloned().collect();
        maps.sort_by_key(|m| m.get_id());
        maps
    }

    pub fn count_maps(&self) -> usize {
        self.inner.lock().unwrap().maps.len()
    }

    pub fn clear_map(&self) {
        if let Some(current_map) = self.current_map() {
            current_map.clear();
        }
    }

    pub fn clear_atlas(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.maps.clear();
        inner.bad_maps.clear();
        inner.current_map.take();
        inner.last_init_kf_id_map = 0;
    }

    pub fn get_current_map(&self) -> Arc<Map> {
        // Fast path: a current map already exists.
        if let Some(current_map) = self.current_map() {
            return current_map;
        }
        self.create_new_map();
        loop {
            // Re-clone under a short lock each iteration; never sleep while the
            // guard is held.
            let current_map = self
                .current_map()
                .expect("create_new_map always sets current_map");
            if !current_map.is_bad() {
                return current_map;
            }
            // TODO: why do we spin lock here?
            std::thread::sleep(Duration::from_micros(3000));
        }
    }

    pub fn set_map_bad(&self, map: Arc<Map>) {
        let id = map.get_id();
        let mut inner = self.inner.lock().unwrap();
        inner.maps.remove(&id);
        map.set_bad();
        inner.bad_maps.insert(id, map);
    }

    pub fn remove_bad_maps(&self) {
        self.inner.lock().unwrap().bad_maps.clear();
    }

    pub fn is_inertial(&self) -> Option<bool> {
        self.current_map().map(|m| m.is_inertial())
    }
    pub fn set_inertial_sensor(&self) {
        if let Some(current_map) = self.current_map() {
            current_map.set_inertial_sensor();
        }
    }
    pub fn set_imu_initialized(&self) {
        if let Some(current_map) = self.current_map() {
            current_map.set_imu_initialized();
        }
    }
    pub fn is_imu_initialized(&self) -> Option<bool> {
        self.current_map().map(|m| m.is_imu_initialized())
    }

    /// Prepare the atlas for serialization (`Atlas::PreSave`).
    ///
    /// Advances `last_init_kf_id_map` past the current map's keyframes, drops
    /// empty maps, and runs each surviving map's observation cleanup
    /// ([`Map::pre_save`]). The actual pointer→id conversion happens lazily in
    /// the [`Serialize`] implementation via [`Map::to_snapshot`]. Call this before
    /// serializing the atlas.
    pub fn pre_save(&self) {
        // Collect the maps to discard while holding the lock, then release it
        // before calling `set_map_bad` / `remove_bad_maps` (which re-lock).
        let maps_to_mark_bad: Vec<_> = {
            let mut inner = self.inner.lock().unwrap();

            let max = inner.current_map.as_ref().map(|m| m.get_max_kf_id());
            if let Some(max) = max
                && !inner.maps.is_empty()
                && inner.last_init_kf_id_map < max
            {
                inner.last_init_kf_id_map = max + 1;
            }

            let cams = inner.cameras.clone();

            inner
                .maps
                .values()
                .filter_map(|map| {
                    if map.is_bad() {
                        return None;
                    }
                    if map.get_all_key_frames().is_empty() {
                        return Some(map.clone());
                    }
                    map.pre_save(&cams);
                    None
                })
                .collect()
        };

        for map in maps_to_mark_bad {
            self.set_map_bad(map);
        }

        self.remove_bad_maps();
    }

    /// Finalize the atlas after deserialization (`Atlas::PostLoad`).
    ///
    /// The map / keyframe / map-point graph is rebuilt during deserialization
    /// (see [`Map::from_snapshot`]); this only registers each map's keyframes with
    /// the keyframe database. Requires [`set_key_frame_database`] and
    /// [`set_orb_vocabulary`] to have been called first.
    ///
    /// [`set_key_frame_database`]: Atlas::set_key_frame_database
    /// [`set_orb_vocabulary`]: Atlas::set_orb_vocabulary
    pub fn post_load(&self) {
        let (Some(db), Some(voc)) = (self.key_frame_database.get(), self.orb_vocabulary.get())
        else {
            return;
        };
        let inner = self.inner.lock().unwrap();
        for map in inner.maps.values() {
            map.post_load(db.clone(), voc.clone());
        }
    }

    /// Serialize the atlas to a self-contained binary blob (magic + version
    /// header followed by a postcard-encoded [`AtlasSnapshot`]).
    ///
    /// Call [`pre_save`](Atlas::pre_save) first to drop empty maps and clean up
    /// cross-map observations.
    pub fn to_bytes(&self) -> Result<Vec<u8>, AtlasIoError> {
        let mut out = Vec::new();
        out.extend_from_slice(&ATLAS_MAGIC);
        out.extend_from_slice(&ATLAS_FORMAT_VERSION.to_le_bytes());
        let body = postcard::to_allocvec(self).map_err(AtlasIoError::Postcard)?;
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Reconstruct an atlas from a blob produced by [`to_bytes`](Atlas::to_bytes).
    ///
    /// The keyframe database and ORB vocabulary are *not* restored here; set them
    /// with [`set_key_frame_database`](Atlas::set_key_frame_database) /
    /// [`set_orb_vocabulary`](Atlas::set_orb_vocabulary) and call
    /// [`post_load`](Atlas::post_load) to finish wiring.
    pub fn from_bytes(bytes: &[u8]) -> Result<Atlas, AtlasIoError> {
        let (header, body) = bytes.split_at_checked(8).ok_or(AtlasIoError::Truncated)?;
        if header[..4] != ATLAS_MAGIC {
            return Err(AtlasIoError::BadMagic);
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != ATLAS_FORMAT_VERSION {
            return Err(AtlasIoError::UnsupportedVersion(version));
        }
        postcard::from_bytes(body).map_err(AtlasIoError::Postcard)
    }

    pub fn get_atlas_key_frames(&self) -> HashMap<u64, Arc<KeyFrame>> {
        let inner = self.inner.lock().unwrap();
        let mut ret = HashMap::new();
        for map in inner.maps.values() {
            let keyframes = map.get_all_key_frames();
            for kf in keyframes {
                ret.insert(kf.id, kf.clone());
            }
        }
        ret
    }

    pub fn set_key_frame_database(&self, db: Arc<KeyFrameDatabase>) {
        let _ = self.key_frame_database.set(db);
    }
    pub fn get_key_frame_database(&self) -> Option<Arc<KeyFrameDatabase>> {
        self.key_frame_database.get().cloned()
    }

    pub fn set_orb_vocabulary(&self, voc: Arc<OrbVocabulary>) {
        let _ = self.orb_vocabulary.set(voc);
    }
    pub fn get_orb_vocabulary(&self) -> Option<Arc<OrbVocabulary>> {
        self.orb_vocabulary.get().cloned()
    }

    pub fn get_num_lived_kdf(&self) -> usize {
        self.inner
            .lock()
            .unwrap()
            .maps
            .values()
            .map(|map| map.get_all_key_frames().len())
            .sum()
    }
    pub fn get_num_lived_mp(&self) -> usize {
        self.inner
            .lock()
            .unwrap()
            .maps
            .values()
            .map(|map| map.get_all_map_points().len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_models::pinhole::Pinhole;
    use crate::test_helpers::build_frame;

    fn kf_in(map: &Arc<Map>) -> Arc<KeyFrame> {
        KeyFrame::from_frame(&build_frame(), map.clone())
    }

    fn pinhole(params: Vec<f32>) -> Arc<dyn GeometricCamera> {
        Arc::new(Pinhole::with_params(params))
    }

    #[test]
    fn new_atlas_is_empty() {
        let atlas = Atlas::new();
        assert_eq!(atlas.count_maps(), 0);
        assert!(atlas.get_all_maps().is_empty());
        assert_eq!(atlas.get_last_init_kf_id(), 0);
        // Accessors on the (absent) current map return None.
        assert!(atlas.get_all_key_frames().is_none());
        assert!(atlas.is_inertial().is_none());
    }

    #[test]
    fn create_new_map_sets_current() {
        let atlas = Atlas::new();
        atlas.create_new_map();
        assert_eq!(atlas.count_maps(), 1);
        let m = atlas.get_current_map();
        assert!(!m.is_bad());
        assert!(m.is_in_use()); // set_current_map() was called
    }

    #[test]
    fn from_kf_id_creates_first_map() {
        let id: u64 = 41;
        let atlas = Atlas::from_kf_id(id);
        assert_eq!(atlas.count_maps(), 1);
        assert_eq!(atlas.get_last_init_kf_id(), id);
        assert_eq!(atlas.get_current_map().get_init_kf_id(), id);
    }

    #[test]
    fn create_new_map_advances_last_init_kf_id() {
        let atlas = Atlas::new();
        atlas.create_new_map();
        let m = atlas.get_current_map();
        // Two keyframes guarantee max_kf_id >= 1, so it is strictly greater than
        // the initial last_init_kf_id (0) regardless of the global id counter.
        let k1 = kf_in(&m);
        let k2 = kf_in(&m);
        m.add_key_frame(k1.clone());
        m.add_key_frame(k2.clone());
        let max = m.get_max_kf_id();
        assert_eq!(max, k1.id.max(k2.id));
        assert!(max >= 1);

        atlas.create_new_map();
        assert_eq!(atlas.count_maps(), 2);
        // The next map's init id is one past the previous map's max KF id.
        assert_eq!(atlas.get_last_init_kf_id(), max + 1);
    }

    #[test]
    fn get_current_map_creates_when_absent() {
        let atlas = Atlas::new();
        assert_eq!(atlas.count_maps(), 0);
        let m = atlas.get_current_map();
        assert!(!m.is_bad());
        assert_eq!(atlas.count_maps(), 1);
    }

    #[test]
    fn set_map_bad_moves_out_of_live_set() {
        let atlas = Atlas::new();
        atlas.create_new_map();
        let first = atlas.get_current_map();
        atlas.create_new_map();
        assert_eq!(atlas.count_maps(), 2);

        atlas.set_map_bad(first.clone());
        assert_eq!(atlas.count_maps(), 1);
        assert!(first.is_bad());

        atlas.remove_bad_maps();
        assert_eq!(atlas.count_maps(), 1);
    }

    #[test]
    fn clear_atlas_resets_everything() {
        let atlas = Atlas::from_kf_id(10);
        atlas.create_new_map();
        assert!(atlas.count_maps() >= 1);

        atlas.clear_atlas();
        assert_eq!(atlas.count_maps(), 0);
        assert_eq!(atlas.get_last_init_kf_id(), 0);
        assert!(atlas.get_all_key_frames().is_none());
    }

    #[test]
    fn clear_map_empties_current_but_keeps_it() {
        let atlas = Atlas::new();
        atlas.create_new_map();
        let m = atlas.get_current_map();
        m.add_key_frame(kf_in(&m));
        assert_eq!(atlas.key_frames_in_map(), Some(1));

        atlas.clear_map();
        // Current map is still present, just emptied.
        assert_eq!(atlas.key_frames_in_map(), Some(0));
        assert_eq!(atlas.count_maps(), 1);
    }

    #[test]
    fn add_camera_dedups_equal_and_keeps_distinct() {
        let atlas = Atlas::new();
        let c1 = atlas.add_camera(pinhole(vec![1.0, 2.0, 3.0, 4.0]));
        let c2 = atlas.add_camera(pinhole(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(atlas.get_all_cameras().len(), 1);
        assert!(Arc::ptr_eq(&c1, &c2));

        let c3 = atlas.add_camera(pinhole(vec![9.0, 9.0, 9.0, 9.0]));
        assert_eq!(atlas.get_all_cameras().len(), 2);
        assert!(!Arc::ptr_eq(&c1, &c3));
    }

    #[test]
    fn imu_and_inertial_flags_route_to_current_map() {
        let atlas = Atlas::new();
        atlas.create_new_map();

        assert_eq!(atlas.is_inertial(), Some(false));
        atlas.set_inertial_sensor();
        assert_eq!(atlas.is_inertial(), Some(true));

        assert_eq!(atlas.is_imu_initialized(), Some(false));
        atlas.set_imu_initialized();
        assert_eq!(atlas.is_imu_initialized(), Some(true));
    }

    #[test]
    fn get_all_maps_sorted_by_id() {
        let atlas = Atlas::new();
        atlas.create_new_map();
        atlas.create_new_map();
        atlas.create_new_map();

        let maps = atlas.get_all_maps();
        assert_eq!(maps.len(), 3);
        for w in maps.windows(2) {
            assert!(w[0].get_id() < w[1].get_id());
        }
    }

    #[test]
    fn num_lived_and_atlas_keyframes_span_all_maps() {
        let atlas = Atlas::new();
        atlas.create_new_map();
        let m1 = atlas.get_current_map();
        let a = kf_in(&m1);
        m1.add_key_frame(a.clone());
        m1.add_map_point(Arc::new(MapPoint::new()));

        atlas.create_new_map();
        let m2 = atlas.get_current_map();
        let b = kf_in(&m2);
        m2.add_key_frame(b.clone());

        assert_eq!(atlas.get_num_lived_kdf(), 2);
        assert_eq!(atlas.get_num_lived_mp(), 1);

        let kfs = atlas.get_atlas_key_frames();
        assert_eq!(kfs.len(), 2);
        assert!(kfs.contains_key(&a.id));
        assert!(kfs.contains_key(&b.id));
    }

    /// Regression test for the `Map` <-> `KeyFrame` reference cycle: because the
    /// keyframe's back-pointer is now [`Weak`], dropping every strong handle to
    /// a map must actually free it (and its keyframes), and leave the keyframe's
    /// back-pointer dangling rather than keeping the map alive forever.
    #[test]
    fn weak_back_pointer_breaks_cycle() {
        let atlas = Atlas::new();
        atlas.create_new_map();
        let map = atlas.get_current_map();
        let kf = kf_in(&map);
        map.add_key_frame(kf.clone());

        // While the map is alive, the back-pointer resolves.
        assert!(kf.get_map().is_some());

        let weak_map = Arc::downgrade(&map);
        drop(map);
        atlas.clear_atlas(); // drops the atlas's strong handles to the map

        // No strong refs remain -> the map is freed (no Arc cycle leak).
        assert!(
            weak_map.upgrade().is_none(),
            "map leaked: the Arc cycle was not broken"
        );
        // The surviving keyframe's back-pointer is now dangling, not a leak.
        assert!(kf.get_map().is_none());
    }

    /// Round-trip a map with a non-trivial graph through the snapshot form
    /// ([`Map::to_snapshot`] / [`Map::from_snapshot`]) — the pointer↔id conversion
    /// that backs the atlas [`Serialize`] / [`Deserialize`] impls — and assert
    /// ids, the covisibility graph, the spanning tree, observations, origins and
    /// the camera link are preserved.
    ///
    /// The serde wire format itself is exercised at compile time by the
    /// `#[derive(Serialize, Deserialize)]` on the snapshot structs; this test
    /// covers the substantive reconstruction logic without a format dependency.
    #[test]
    fn snapshot_round_trip_preserves_graph() {
        use crate::map_point::MapPoint;
        use nalgebra::Vector3;

        let map = Arc::new(Map::new());

        // Two keyframes built from a real frame (so descriptors / keypoints
        // exist), sharing the frame's camera.
        let frame = build_frame();
        let cam = frame.camera.clone();
        let a = KeyFrame::from_frame(&frame, map.clone());
        let b = KeyFrame::from_frame(&frame, map.clone());
        a.set_pose(crate::test_helpers::make_pose());
        map.add_key_frame(a.clone());
        map.add_key_frame(b.clone());
        map.add_key_frame_origin(a.clone());

        // Covisibility weight + spanning tree edge.
        a.add_connection(&b, 42);
        b.add_connection(&a, 42);
        b.change_parent(&a);

        // A map point observed by `a`.
        let mp = MapPoint::from_frame(&Vector3::new(1.0, 2.0, 3.0), map.clone(), &frame, 0);
        map.add_map_point(mp.clone());
        mp.add_observation(&a, 0);

        // Round-trip through the serializable snapshot form.
        let snapshot = map.to_snapshot();
        let cam_map: HashMap<u64, Arc<dyn GeometricCamera>> =
            HashMap::from([(cam.get_id(), cam.clone())]);
        let rmap = Map::from_snapshot(&snapshot, &cam_map);

        // Map / keyframe / map-point counts and id.
        assert_eq!(rmap.get_id(), map.get_id());
        assert_eq!(rmap.key_frames_in_map(), 2);
        assert_eq!(rmap.map_points_in_map(), 1);

        // Keyframe ids preserved; look them up in the restored map.
        let r_kfs: HashMap<u64, _> = rmap
            .get_all_key_frames()
            .into_iter()
            .map(|k| (k.id, k))
            .collect();
        let ra = r_kfs.get(&a.id).expect("keyframe a restored").clone();
        let rb = r_kfs.get(&b.id).expect("keyframe b restored").clone();

        // Covisibility weight + spanning tree restored.
        assert_eq!(ra.get_weight(&rb), 42);
        assert_eq!(rb.get_parent().expect("parent restored").id, a.id);
        assert!(ra.get_children().iter().any(|c| c.id == b.id));

        // Origins, initial keyframe.
        assert_eq!(rmap.get_key_frame_origins()[0].id, a.id);
        assert_eq!(rmap.get_origin_kf().expect("origin kf").id, a.id);

        // Map-point observation restored, pointing at restored keyframe `a`.
        let rmp = rmap.get_all_map_points()[0].clone();
        assert_eq!(rmp.id, mp.id);
        let obs = rmp.get_observations();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].0.id, a.id);

        // Back-pointers re-point at the restored map (Weak upgrade).
        assert_eq!(ra.get_map().expect("kf map").get_id(), rmap.get_id());
        assert_eq!(rmp.get_map().expect("mp map").get_id(), rmap.get_id());

        // Camera link + pose preserved.
        assert_eq!(ra.camera.get_id(), cam.get_id());
        let dt = (ra.get_pose().translation.vector - a.get_pose().translation.vector).norm();
        assert!(dt < 1e-5, "pose drift {dt}");
    }

    /// True end-to-end test of the atlas `Serialize` / `Deserialize` impls:
    /// `Atlas -> bytes -> Atlas` through the postcard wire format.
    #[test]
    fn atlas_serde_end_to_end() {
        use crate::map_point::MapPoint;
        use nalgebra::Vector3;

        let atlas = Atlas::new();
        atlas.create_new_map();
        let map = atlas.get_current_map();

        let frame = build_frame();
        let cam = atlas.add_camera(frame.camera.clone());
        let a = KeyFrame::from_frame(&frame, map.clone());
        let b = KeyFrame::from_frame(&frame, map.clone());
        a.set_pose(crate::test_helpers::make_pose());
        map.add_key_frame(a.clone());
        map.add_key_frame(b.clone());
        map.add_key_frame_origin(a.clone());
        a.add_connection(&b, 7);
        b.add_connection(&a, 7);
        b.change_parent(&a);
        let mp = MapPoint::from_frame(&Vector3::new(1.0, 2.0, 3.0), map.clone(), &frame, 0);
        map.add_map_point(mp.clone());
        mp.add_observation(&a, 0);

        atlas.pre_save();

        // Full wire round-trip through postcard (magic + version header + body).
        let bytes = atlas.to_bytes().expect("serialize atlas");
        assert_eq!(&bytes[..4], b"ORBA");
        let restored = Atlas::from_bytes(&bytes).expect("deserialize atlas");

        // Corrupting the magic / version is rejected.
        let mut bad = bytes.clone();
        bad[0] ^= 0xFF;
        assert!(matches!(
            Atlas::from_bytes(&bad),
            Err(AtlasIoError::BadMagic)
        ));

        // Top-level structure + cameras + counters.
        assert_eq!(restored.count_maps(), 1);
        assert_eq!(restored.get_all_cameras().len(), 1);
        assert_eq!(restored.get_all_cameras()[0].get_id(), cam.get_id());
        assert_eq!(restored.get_last_init_kf_id(), atlas.get_last_init_kf_id());

        // Map contents.
        let rmap = restored.get_all_maps()[0].clone();
        assert_eq!(rmap.get_id(), map.get_id());
        assert_eq!(rmap.key_frames_in_map(), 2);
        assert_eq!(rmap.map_points_in_map(), 1);

        // Graph: covisibility weight, spanning tree, origin.
        let kfs = restored.get_atlas_key_frames();
        let ra = kfs.get(&a.id).expect("a restored").clone();
        let rb = kfs.get(&b.id).expect("b restored").clone();
        assert_eq!(ra.get_weight(&rb), 7);
        assert_eq!(rb.get_parent().expect("parent").id, a.id);
        assert_eq!(rmap.get_origin_kf().expect("origin").id, a.id);

        // Observation + camera + pose.
        let rmp = rmap.get_all_map_points()[0].clone();
        assert_eq!(rmp.get_observations()[0].0.id, a.id);
        assert_eq!(ra.camera.get_id(), cam.get_id());
        let dt = (ra.get_pose().translation.vector - a.get_pose().translation.vector).norm();
        assert!(dt < 1e-5, "pose drift {dt}");
    }
}
