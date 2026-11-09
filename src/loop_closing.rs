//! Loop detection and correction.
//!
//! Detection is split into a vocabulary candidate phase and a geometric
//! verification phase. Only the latter is allowed to alter the map.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use crate::atlas::Atlas;
use crate::g2o_core::Sim3;
use crate::key_frame::KeyFrame;
use crate::key_frame_database::{KeyFrameDatabase, KeyFrameId, KeyFrameView, MapId};
use crate::local_mapping::LocalMapping;
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
    finish_requested: AtomicBool,
    finished: AtomicBool,
    running_loop: AtomicBool,
    last_processed: Mutex<Option<u64>>,
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
            finish_requested: AtomicBool::new(false),
            finished: AtomicBool::new(true),
            running_loop: AtomicBool::new(false),
            last_processed: Mutex::new(None),
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

    pub fn request_finish(&self) {
        self.finish_requested.store(true, Ordering::SeqCst);
    }

    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    pub fn run(&self) {
        self.finished.store(false, Ordering::SeqCst);
        loop {
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
                    && self.detect_and_correct(&current, &loop_kf)
                {
                    break;
                }
            }
            for candidate in candidates.merge {
                if let Some(merge_kf) = view.key_frames.get(&candidate).cloned()
                    && self.detect_and_merge(&current, &merge_kf)
                {
                    break;
                }
            }
        }
    }

    fn detect_and_correct(&self, current: &Arc<KeyFrame>, candidate: &Arc<KeyFrame>) -> bool {
        if current.id == candidate.id || current.is_bad() || candidate.is_bad() {
            return false;
        }

        let pose = current.get_pose();
        let current_sim_f32 =
            nalgebra::Similarity3::from_parts(pose.translation, pose.rotation, 1.0_f32);
        let current_sim = Sim3::new(
            pose.rotation.cast::<f64>(),
            pose.translation.vector.cast::<f64>(),
            1.0,
        );
        let points: Vec<_> = candidate.get_map_points().into_iter().collect();
        let mut matches = vec![None; current.n];
        let matcher = OrbMatcher::new(0.75, true);
        let n_matches = matcher.search_by_projection_similarity_loop_detection(
            current,
            &current_sim_f32,
            &points,
            &mut matches,
            8.0,
            0.75,
        );
        if n_matches < 20 {
            return false;
        }

        let mut sim3 = current_sim;
        let inliers = optimizer::optimize_sim3(
            current,
            candidate,
            &mut matches,
            &mut sim3,
            10.0,
            self.fix_scale,
            true,
        );
        if inliers < 20 {
            return false;
        }

        tracing::info!(
            "Loop detected: current KF {} matched KF {} ({} inliers)",
            current.id,
            candidate.id,
            inliers
        );
        self.correct_loop(current, candidate, sim3);
        true
    }

    fn detect_and_merge(&self, current: &Arc<KeyFrame>, candidate: &Arc<KeyFrame>) -> bool {
        let Some(detected) = self.detect_sim3(current, candidate) else {
            return false;
        };
        self.merge_maps(current, candidate, detected);
        true
    }

    fn detect_sim3(&self, current: &Arc<KeyFrame>, candidate: &Arc<KeyFrame>) -> Option<Sim3> {
        let pose = current.get_pose();
        let current_sim_f32 =
            nalgebra::Similarity3::from_parts(pose.translation, pose.rotation, 1.0_f32);
        let mut sim3 = Sim3::new(
            pose.rotation.cast::<f64>(),
            pose.translation.vector.cast::<f64>(),
            1.0,
        );
        let points: Vec<_> = candidate.get_map_points().into_iter().collect();
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
        if let Some(mapping) = self.local_mapping.get().and_then(Weak::upgrade) {
            mapping.request_stop();
            mapping.empty_queue();
        }

        let transform = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(sim3.translation().cast::<f32>()),
            sim3.rotation().cast::<f32>(),
        );
        source.apply_scaled_rotation(transform, sim3.scale() as f32, true);
        let (keyframes, points) = source.take_contents();
        for keyframe in keyframes {
            keyframe.update_map(target.clone());
            target.add_key_frame(keyframe);
        }
        for point in points {
            point.update_map(target.clone());
            target.add_map_point(point);
        }
        self.atlas.set_current_map(target.clone());
        self.atlas.set_map_bad(source);
        target.increase_change_index();

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
        if let Some(mapping) = self.local_mapping.get().and_then(Weak::upgrade) {
            mapping.request_stop();
            for _ in 0..1000 {
                if mapping.is_stopped() || mapping.is_finished() {
                    break;
                }
                std::thread::sleep(Duration::from_micros(3000));
            }
            mapping.empty_queue();
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
        let mut corrected: KeyFrameAndPose = HashMap::new();
        corrected.insert(current.id, corrected_current);
        let non_corrected: KeyFrameAndPose = HashMap::new();
        let mut connections: LoopConnections = HashMap::new();
        connections
            .entry(current.id)
            .or_default()
            .insert(candidate.id);
        connections
            .entry(candidate.id)
            .or_default()
            .insert(current.id);

        optimizer::optimize_essential_graph(
            &map,
            candidate,
            current,
            &non_corrected,
            &corrected,
            &connections,
            self.fix_scale,
        );

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
}

struct AtlasKeyFrameView {
    key_frames: HashMap<KeyFrameId, Arc<KeyFrame>>,
    bad_maps: HashSet<MapId>,
}

impl AtlasKeyFrameView {
    fn from_atlas(atlas: &Atlas) -> Self {
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
