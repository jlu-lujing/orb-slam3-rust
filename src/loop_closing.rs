//! Loop detection and correction.
//!
//! Detection is split into a vocabulary candidate phase and a geometric
//! verification phase. Only the latter is allowed to alter the map.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use nalgebra::Translation3;

use crate::atlas::Atlas;
use crate::g2o_core::Sim3;
use crate::key_frame::KeyFrame;
use crate::key_frame_database::{KeyFrameDatabase, KeyFrameId, KeyFrameView, MapId};
use crate::local_mapping::LocalMapping;
use crate::map::Map;
use crate::map_point::MapPoint;
use crate::optimizer::{self, KeyFrameAndPose, LoopConnections};
use crate::orb_matcher::OrbMatcher;
use crate::orb_vocabulary::{BowVector, OrbVocabulary};

pub struct LoopClosing {
    atlas: Arc<Atlas>,
    database: Arc<KeyFrameDatabase>,
    vocabulary: Arc<OrbVocabulary>,
    fix_scale: bool,
    active_loop_closing: bool,
    local_mapping: OnceLock<Weak<LocalMapping>>,
    key_frames: Mutex<VecDeque<Arc<KeyFrame>>>,
    reset_all: AtomicBool,
    reset_active_map: Mutex<Option<Arc<Map>>>,
    finish_requested: AtomicBool,
    finished: AtomicBool,
    running_loop: AtomicBool,
    last_processed: Mutex<Option<u64>>,
    pending_match: Mutex<Option<PendingMatch>>,
    gba_running: Arc<AtomicBool>,
    gba_stop: Arc<AtomicBool>,
    gba_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

#[derive(Debug, Clone)]
struct PendingMatch {
    candidate_group: HashSet<u64>,
    last_current_id: u64,
    count: u8,
}

impl LoopClosing {
    pub fn new(
        atlas: Arc<Atlas>,
        database: Arc<KeyFrameDatabase>,
        vocabulary: Arc<OrbVocabulary>,
        fix_scale: bool,
        active_loop_closing: bool,
    ) -> Self {
        Self {
            atlas,
            database,
            vocabulary,
            fix_scale,
            active_loop_closing,
            local_mapping: OnceLock::new(),
            key_frames: Mutex::new(VecDeque::new()),
            reset_all: AtomicBool::new(false),
            reset_active_map: Mutex::new(None),
            finish_requested: AtomicBool::new(false),
            finished: AtomicBool::new(true),
            running_loop: AtomicBool::new(false),
            last_processed: Mutex::new(None),
            pending_match: Mutex::new(None),
            gba_running: Arc::new(AtomicBool::new(false)),
            gba_stop: Arc::new(AtomicBool::new(false)),
            gba_thread: Mutex::new(None),
        }
    }

    pub fn set_local_mapper(&self, mapping: Arc<LocalMapping>) {
        let _ = self.local_mapping.set(Arc::downgrade(&mapping));
    }

    pub fn insert_key_frame(&self, keyframe: Arc<KeyFrame>) {
        self.key_frames.lock().unwrap().push_back(keyframe);
    }

    pub fn keyframes_in_queue(&self) -> usize {
        self.key_frames.lock().unwrap().len()
    }

    pub fn is_running_loop(&self) -> bool {
        self.running_loop.load(Ordering::SeqCst)
    }

    pub fn is_running_gba(&self) -> bool {
        self.gba_running.load(Ordering::SeqCst)
    }

    pub fn request_finish(&self) {
        self.finish_requested.store(true, Ordering::SeqCst);
        self.stop_and_join_gba();
    }

    pub fn request_reset(&self) {
        self.reset_all.store(true, Ordering::SeqCst);
        while self.reset_all.load(Ordering::SeqCst) {
            if self.is_finished() {
                self.reset_if_requested();
                break;
            }
            std::thread::sleep(Duration::from_micros(3000));
        }
    }

    /// Drop queued keyframes from `map` and block until the loop-closing
    /// worker has acknowledged the request.
    pub fn request_reset_active_map(&self, map: Arc<Map>) {
        *self.reset_active_map.lock().unwrap() = Some(map);
        loop {
            if self.reset_active_map.lock().unwrap().is_none() {
                break;
            }
            if self.is_finished() {
                self.reset_if_requested();
                break;
            }
            std::thread::sleep(Duration::from_micros(3000));
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    pub fn run(&self) {
        self.finished.store(false, Ordering::SeqCst);
        loop {
            self.reset_if_requested();
            let next = self.key_frames.lock().unwrap().pop_front();
            if let Some(current) = next {
                if !current.is_bad() {
                    *self.last_processed.lock().unwrap() = Some(current.id);
                    self.process_keyframe(current);
                }
            } else {
                std::thread::sleep(Duration::from_micros(3000));
            }
            if self.finish_requested.load(Ordering::SeqCst) {
                break;
            }
        }
        self.finished.store(true, Ordering::SeqCst);
    }

    fn reset_if_requested(&self) {
        if self.reset_all.swap(false, Ordering::SeqCst) {
            self.key_frames.lock().unwrap().clear();
            self.pending_match.lock().unwrap().take();
        }
        let reset_map = self.reset_active_map.lock().unwrap().take();
        if let Some(map) = reset_map {
            self.key_frames.lock().unwrap().retain(|keyframe| {
                keyframe
                    .get_map()
                    .is_none_or(|owner| owner.get_id() != map.get_id())
            });
        }
    }

    fn process_keyframe(&self, current: Arc<KeyFrame>) {
        let Some(map) = current.get_map() else {
            return;
        };
        if map.key_frames_in_map() < 10 {
            return;
        }

        let view = AtlasKeyFrameView::from_atlas(&self.atlas);
        let candidates = self
            .database
            .detect_n_best_candidates(&view, KeyFrameId(current.id), 5);
        if self.active_loop_closing {
            for candidate in candidates.loop_ {
                if let Some(loop_kf) = view.key_frames.get(&candidate).cloned()
                    && let Some(sim3) = self.detect_sim3(&current, &loop_kf)
                    && self.confirm_candidate(
                        current.id,
                        &Self::loop_neighborhood(&loop_kf)
                            .into_iter()
                            .map(|keyframe| keyframe.id)
                            .collect(),
                    )
                {
                    tracing::info!(
                        "Loop detected: current KF {} matched KF {}",
                        current.id,
                        loop_kf.id
                    );
                    self.correct_loop(&current, &loop_kf, sim3);
                    break;
                }
            }
            for candidate in candidates.merge {
                if let Some(merge_kf) = view.key_frames.get(&candidate).cloned()
                    && let Some(sim3) = self.detect_sim3(&current, &merge_kf)
                    && self.confirm_candidate(
                        current.id,
                        &Self::loop_neighborhood(&merge_kf)
                            .into_iter()
                            .map(|keyframe| keyframe.id)
                            .collect(),
                    )
                {
                    self.merge_maps(&current, &merge_kf, sim3);
                    break;
                }
            }
        }
    }

    fn confirm_candidate(&self, current_id: u64, candidate_group: &HashSet<u64>) -> bool {
        let mut pending = self.pending_match.lock().unwrap();
        let count = match pending.as_ref() {
            Some(previous)
                if !previous.candidate_group.is_disjoint(candidate_group)
                    && current_id > previous.last_current_id =>
            {
                previous.count.saturating_add(1)
            }
            _ => 1,
        };
        if count >= 3 {
            *pending = None;
            true
        } else {
            *pending = Some(PendingMatch {
                candidate_group: candidate_group.clone(),
                last_current_id: current_id,
                count,
            });
            false
        }
    }

    fn detect_sim3(&self, current: &Arc<KeyFrame>, candidate: &Arc<KeyFrame>) -> Option<Sim3> {
        if current.id == candidate.id || current.is_bad() || candidate.is_bad() {
            return None;
        }
        let pose = current.get_pose();
        let current_sim_f32 =
            nalgebra::Similarity3::from_parts(pose.translation, pose.rotation, 1.0_f32);
        let mut sim3 = Sim3::new(
            pose.rotation.cast::<f64>(),
            pose.translation.vector.cast::<f64>(),
            1.0,
        );
        let points = Self::loop_points(&Self::loop_neighborhood(candidate));
        let mut matches = vec![None; current.n];
        let matcher = OrbMatcher::new(0.75, true);
        if matcher.search_by_projection_similarity_loop_detection(
            current,
            &current_sim_f32,
            &points,
            &mut matches,
            8.0,
            0.75,
        ) < 20
        {
            return None;
        }
        let inliers = optimizer::optimize_sim3(
            current,
            candidate,
            &mut matches,
            &mut sim3,
            10.0,
            self.fix_scale,
            true,
        );
        (inliers >= 20).then_some(sim3)
    }

    fn merge_maps(&self, current: &Arc<KeyFrame>, candidate: &Arc<KeyFrame>, sim3: Sim3) {
        let (Some(source), Some(target)) = (current.get_map(), candidate.get_map()) else {
            return;
        };
        if source.get_id() == target.get_id() {
            return;
        }
        self.running_loop.store(true, Ordering::SeqCst);
        if !self.stop_mapper() {
            self.running_loop.store(false, Ordering::SeqCst);
            return;
        }

        let transform = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(sim3.translation().cast::<f32>()),
            sim3.rotation().cast::<f32>(),
        );
        source.apply_scaled_rotation(transform, sim3.scale() as f32, true);
        let (keyframes, points) = source.take_contents();
        for keyframe in &keyframes {
            keyframe.update_map(target.clone());
            target.add_key_frame(keyframe.clone());
        }
        for point in &points {
            point.update_map(target.clone());
            target.add_map_point(point.clone());
        }
        current.add_loop_edge(candidate);
        candidate.add_loop_edge(current);
        self.atlas.merge_map_into(&source, &target);
        target.increase_change_index();

        let corrected = if target.is_inertial() {
            optimizer::merge_inertial_ba(current, candidate, None, &target)
        } else {
            Self::loop_neighborhood(current)
                .into_iter()
                .map(|keyframe| (keyframe.id, optimizer::sim3_from_pose(&keyframe.get_pose())))
                .collect()
        };
        let fixed_merge = Self::loop_neighborhood(candidate);
        let fixed_current = Self::loop_neighborhood(current);
        optimizer::optimize_essential_graph_merge(
            current,
            &fixed_merge,
            &fixed_current,
            &keyframes,
            &points,
        );
        let fuse_neighbors = Self::loop_neighborhood(candidate);
        let fuse_points = Self::loop_points(&fuse_neighbors);
        self.search_and_fuse(&corrected, &fuse_points);

        for keyframe in target.get_all_key_frames() {
            keyframe.update_connections();
        }
        if let Some(mapping) = self.local_mapping.get().and_then(Weak::upgrade) {
            mapping.release();
        }
        self.running_loop.store(false, Ordering::SeqCst);
    }

    fn correct_loop(&self, current: &Arc<KeyFrame>, candidate: &Arc<KeyFrame>, sim3: Sim3) {
        self.running_loop.store(true, Ordering::SeqCst);
        if !self.stop_mapper() {
            self.running_loop.store(false, Ordering::SeqCst);
            return;
        }

        current.add_loop_edge(candidate);
        candidate.add_loop_edge(current);

        let Some(map) = current.get_map() else {
            self.running_loop.store(false, Ordering::SeqCst);
            return;
        };

        let candidate_pose = candidate.get_pose();
        let candidate_sim = Sim3::new(
            candidate_pose.rotation.cast::<f64>(),
            candidate_pose.translation.vector.cast::<f64>(),
            1.0,
        );
        let corrected_current = sim3.mul(&candidate_sim);
        let current_neighbors = Self::loop_neighborhood(current);
        let loop_neighbors = Self::loop_neighborhood(candidate);
        let loop_points = Self::loop_points(&loop_neighbors);
        let mut corrected: KeyFrameAndPose = HashMap::new();
        for keyframe in &current_neighbors {
            corrected.insert(
                keyframe.id,
                sim3.mul(&optimizer::sim3_from_pose(&keyframe.get_pose())),
            );
        }
        corrected.insert(current.id, corrected_current);
        let non_corrected: KeyFrameAndPose = current_neighbors
            .iter()
            .chain(loop_neighbors.iter())
            .map(|keyframe| (keyframe.id, optimizer::sim3_from_pose(&keyframe.get_pose())))
            .collect();
        let mut connections: LoopConnections = HashMap::new();
        connections
            .entry(current.id)
            .or_default()
            .insert(candidate.id);
        connections
            .entry(candidate.id)
            .or_default()
            .insert(current.id);
        for current_kf in &current_neighbors {
            for loop_kf in &loop_neighbors {
                if Self::common_map_point_count(current_kf, loop_kf) < 100 {
                    continue;
                }
                connections
                    .entry(current_kf.id)
                    .or_default()
                    .insert(loop_kf.id);
                connections
                    .entry(loop_kf.id)
                    .or_default()
                    .insert(current_kf.id);
                current_kf.add_loop_edge(loop_kf);
                loop_kf.add_loop_edge(current_kf);
            }
        }

        let fused = self.search_and_fuse(&corrected, &loop_points);
        if fused > 0 {
            tracing::info!("Loop closure fused {fused} duplicate map points");
        }

        optimizer::optimize_essential_graph(
            &map,
            candidate,
            current,
            &non_corrected,
            &corrected,
            &connections,
            self.fix_scale,
        );
        // ORB-SLAM3 launches global BA in a dedicated thread after the
        // essential-graph correction and commits its staged result later.
        self.start_global_bundle_adjustment(map.clone(), current.id);

        for point in map.get_all_map_points() {
            if !point.is_bad() {
                point.compute_distinctive_descriptors();
                point.update_normal_and_depth();
            }
        }
        for keyframe in map.get_all_key_frames() {
            if !keyframe.is_bad() {
                keyframe.update_connections();
            }
        }
        map.increase_change_index();

        if let Some(mapping) = self.local_mapping.get().and_then(Weak::upgrade) {
            mapping.release();
        }
        self.running_loop.store(false, Ordering::SeqCst);
    }

    fn stop_mapper(&self) -> bool {
        let Some(mapping) = self.local_mapping.get().and_then(Weak::upgrade) else {
            return true;
        };
        mapping.request_stop();
        for _ in 0..1000 {
            if mapping.is_stopped() || mapping.is_finished() {
                mapping.empty_queue();
                return true;
            }
            std::thread::sleep(Duration::from_micros(3000));
        }
        tracing::error!("local mapper did not stop; aborting loop-closing update");
        false
    }

    fn start_global_bundle_adjustment(&self, map: Arc<Map>, loop_kf: u64) {
        self.stop_and_join_gba();
        self.gba_stop.store(false, Ordering::SeqCst);
        self.gba_running.store(true, Ordering::SeqCst);
        let stop = self.gba_stop.clone();
        let running = self.gba_running.clone();
        let handle = std::thread::Builder::new()
            .name("orb-global-ba".to_owned())
            .spawn(move || {
                optimizer::global_bundle_adjustment(&map, 10, Some(stop.clone()), loop_kf, false);
                if !stop.load(Ordering::SeqCst) {
                    optimizer::commit_global_bundle_adjustment(&map, loop_kf);
                }
                running.store(false, Ordering::SeqCst);
            });
        if let Ok(handle) = handle {
            *self.gba_thread.lock().unwrap() = Some(handle);
            if self.finish_requested.load(Ordering::SeqCst) {
                self.stop_and_join_gba();
            }
        } else {
            self.gba_running.store(false, Ordering::SeqCst);
            tracing::error!("failed to start global-BA thread");
        }
    }

    fn stop_and_join_gba(&self) {
        self.gba_stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.gba_thread.lock().unwrap().take() {
            let _ = handle.join();
        }
        self.gba_running.store(false, Ordering::SeqCst);
    }

    fn search_and_fuse(&self, corrected: &KeyFrameAndPose, points: &[Arc<MapPoint>]) -> usize {
        let matcher = OrbMatcher::new(0.8, true);
        let keyframes: HashMap<_, _> = self.atlas.get_atlas_key_frames().into_iter().collect();
        let mut total = 0;
        let mut keyframe_ids: Vec<_> = corrected.keys().copied().collect();
        keyframe_ids.sort_unstable();
        for keyframe_id in keyframe_ids {
            let Some(scw) = corrected.get(&keyframe_id) else {
                continue;
            };
            let Some(keyframe) = keyframes.get(&keyframe_id) else {
                continue;
            };
            let rotation = scw.rotation().cast::<f32>();
            let similarity = nalgebra::Similarity3::from_parts(
                Translation3::from(scw.translation().cast::<f32>()),
                rotation,
                scw.scale() as f32,
            );
            let mut replacements = vec![None; points.len()];
            matcher.fuse_by_sim3(keyframe, &similarity, points, 4.0, &mut replacements);
            for (source, replacement) in points.iter().zip(replacements) {
                if let Some(replacement) = replacement
                    && !source.is_bad()
                    && source.id != replacement.id
                {
                    source.replace(&replacement);
                    total += 1;
                }
            }
        }
        total
    }

    fn loop_neighborhood(keyframe: &Arc<KeyFrame>) -> Vec<Arc<KeyFrame>> {
        let mut keyframes = vec![keyframe.clone()];
        keyframes.extend(keyframe.get_best_covisibility_keyframes(10));
        keyframes
    }

    fn loop_points(keyframes: &[Arc<KeyFrame>]) -> Vec<Arc<MapPoint>> {
        let mut seen = HashSet::new();
        keyframes
            .iter()
            .flat_map(|keyframe| keyframe.get_map_points())
            .filter(|point| !point.is_bad())
            .filter(|point| seen.insert(point.id))
            .collect()
    }

    fn common_map_point_count(left: &KeyFrame, right: &KeyFrame) -> usize {
        let left: HashSet<_> = left.get_map_points().into_iter().map(|mp| mp.id).collect();
        right
            .get_map_points()
            .into_iter()
            .filter(|mp| left.contains(&mp.id))
            .count()
    }
}

pub(crate) struct AtlasKeyFrameView {
    key_frames: HashMap<KeyFrameId, Arc<KeyFrame>>,
    bad_maps: HashSet<MapId>,
}

impl AtlasKeyFrameView {
    pub(crate) fn from_atlas(atlas: &Atlas) -> Self {
        let key_frames = atlas
            .get_atlas_key_frames()
            .into_iter()
            .map(|(id, kf)| (KeyFrameId(id), kf))
            .collect();
        let bad_maps = atlas
            .get_all_maps()
            .into_iter()
            .filter(|map| map.is_bad())
            .map(|map| MapId(map.get_id() as u64))
            .collect();
        Self {
            key_frames,
            bad_maps,
        }
    }
}

impl KeyFrameView for AtlasKeyFrameView {
    fn bow_vec(&self, kf: KeyFrameId) -> Option<&BowVector> {
        self.key_frames.get(&kf).map(|kf| &kf.bow_vec)
    }

    fn map_id(&self, kf: KeyFrameId) -> Option<MapId> {
        self.key_frames
            .get(&kf)
            .and_then(|kf| kf.get_map())
            .map(|map| MapId(map.get_id() as u64))
    }

    fn is_bad(&self, kf: KeyFrameId) -> bool {
        self.key_frames
            .get(&kf)
            .is_none_or(|keyframe| keyframe.is_bad())
    }

    fn is_map_bad(&self, map: MapId) -> bool {
        self.bad_maps.contains(&map)
    }

    fn best_covisibility(&self, kf: KeyFrameId, n: usize) -> Vec<KeyFrameId> {
        self.key_frames
            .get(&kf)
            .map(|kf| {
                kf.get_best_covisibility_keyframes(n)
                    .into_iter()
                    .map(|kf| KeyFrameId(kf.id))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn connected_keyframes(&self, kf: KeyFrameId) -> HashSet<KeyFrameId> {
        self.key_frames
            .get(&kf)
            .map(|kf| {
                kf.get_connected_keyframes()
                    .into_iter()
                    .map(|kf| KeyFrameId(kf.id))
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_frame::KeyFrame;
    use crate::test_helpers::{build_frame, dummy_voc};

    #[test]
    fn reset_active_map_drops_only_queued_keyframes_from_that_map() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let vocabulary = dummy_voc();
        let database = Arc::new(KeyFrameDatabase::new(vocabulary.clone()));
        let loop_closing = LoopClosing::new(atlas, database, vocabulary, true, true);
        let map_a = Arc::new(Map::new());
        let map_b = Arc::new(Map::new());
        let keyframe_a = KeyFrame::from_frame(&build_frame(), map_a.clone());
        let keyframe_b = KeyFrame::from_frame(&build_frame(), map_b);
        loop_closing.insert_key_frame(keyframe_a);
        loop_closing.insert_key_frame(keyframe_b.clone());

        *loop_closing.reset_active_map.lock().unwrap() = Some(map_a);
        loop_closing.reset_if_requested();

        let queue = loop_closing.key_frames.lock().unwrap();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].id, keyframe_b.id);
    }

    #[test]
    fn loop_confirmation_requires_three_distinct_consecutive_current_keyframes() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let vocabulary = dummy_voc();
        let database = Arc::new(KeyFrameDatabase::new(vocabulary.clone()));
        let loop_closing = LoopClosing::new(atlas, database, vocabulary, true, true);
        let group = HashSet::from([2]);

        assert!(!loop_closing.confirm_candidate(10, &group));
        assert!(!loop_closing.confirm_candidate(10, &group));
        assert!(!loop_closing.confirm_candidate(11, &group));
        assert!(loop_closing.confirm_candidate(12, &group));
        assert!(!loop_closing.confirm_candidate(13, &group));
    }

    #[test]
    fn loop_confirmation_accepts_overlapping_candidate_neighborhoods() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let vocabulary = dummy_voc();
        let database = Arc::new(KeyFrameDatabase::new(vocabulary.clone()));
        let loop_closing = LoopClosing::new(atlas, database, vocabulary, true, true);
        let first = HashSet::from([2, 3, 4]);
        let overlapping = HashSet::from([4, 5, 6]);
        let disjoint = HashSet::from([20, 21, 22]);

        assert!(!loop_closing.confirm_candidate(10, &first));
        assert!(!loop_closing.confirm_candidate(11, &overlapping));
        assert!(!loop_closing.confirm_candidate(12, &disjoint));
        assert!(!loop_closing.confirm_candidate(13, &disjoint));
        assert!(loop_closing.confirm_candidate(14, &disjoint));
    }

    #[test]
    fn reset_request_completes_even_if_loop_thread_is_not_running() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let vocabulary = dummy_voc();
        let database = Arc::new(KeyFrameDatabase::new(vocabulary.clone()));
        let loop_closing = LoopClosing::new(atlas, database, vocabulary, true, true);
        let map = Arc::new(Map::new());
        let keyframe = KeyFrame::from_frame(&build_frame(), map.clone());
        loop_closing.insert_key_frame(keyframe);

        assert!(loop_closing.is_finished());
        loop_closing.request_reset_active_map(map);
        assert_eq!(loop_closing.keyframes_in_queue(), 0);
    }

    #[test]
    fn merge_maps_moves_source_contents_into_target() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let vocabulary = dummy_voc();
        let database = Arc::new(KeyFrameDatabase::new(vocabulary.clone()));
        let loop_closing = LoopClosing::new(atlas.clone(), database, vocabulary, true, true);
        let source = Arc::new(Map::new());
        let target = Arc::new(Map::new());
        let source_id = source.get_id();
        let current = KeyFrame::from_frame(&build_frame(), source.clone());
        let candidate = KeyFrame::from_frame(&build_frame(), target.clone());
        source.add_key_frame(current.clone());
        target.add_key_frame(candidate.clone());

        loop_closing.merge_maps(
            &current,
            &candidate,
            Sim3::new(
                nalgebra::UnitQuaternion::identity(),
                nalgebra::Vector3::zeros(),
                1.0,
            ),
        );

        assert_eq!(source.key_frames_in_map(), 0);
        assert_eq!(target.key_frames_in_map(), 2);
        assert_eq!(target.get_id(), source_id);
        assert_eq!(
            current.get_map().map(|map| map.get_id()),
            Some(target.get_id())
        );
        assert_eq!(
            atlas.get_current_map().get_id(),
            target.get_id(),
            "merged target should become the active atlas map"
        );
    }

    #[test]
    fn staged_global_ba_results_commit_atomically() {
        let map = Arc::new(Map::new());
        let keyframe = KeyFrame::from_frame(&build_frame(), map.clone());
        map.add_key_frame(keyframe.clone());
        let point = Arc::new(MapPoint::new());
        map.add_map_point(point.clone());
        let staged_pose = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, 2.0, 3.0),
            nalgebra::UnitQuaternion::identity(),
        );
        keyframe.set_tcw_gba(staged_pose, 9);
        point.set_pos_gba(nalgebra::Vector3::new(4.0, 5.0, 6.0), 9);

        assert_eq!(optimizer::commit_global_bundle_adjustment(&map, 9), 1);
        assert!(
            (keyframe.get_pose().translation.vector - staged_pose.translation.vector).norm() < 1e-6
        );
        assert!((point.get_world_pos() - nalgebra::Vector3::new(4.0, 5.0, 6.0)).norm() < 1e-6);
    }

    #[test]
    fn background_global_ba_completes_and_shutdown_joins_it() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let vocabulary = dummy_voc();
        let database = Arc::new(KeyFrameDatabase::new(vocabulary.clone()));
        let loop_closing = LoopClosing::new(atlas, database, vocabulary, true, true);
        let map = Arc::new(Map::new());
        let keyframe = KeyFrame::from_frame(&build_frame(), map.clone());
        map.add_key_frame(keyframe);

        loop_closing.start_global_bundle_adjustment(map, 1);
        for _ in 0..1000 {
            if !loop_closing.is_running_gba() {
                break;
            }
            std::thread::sleep(Duration::from_micros(3000));
        }
        assert!(!loop_closing.is_running_gba());
        loop_closing.request_finish();
    }
}
