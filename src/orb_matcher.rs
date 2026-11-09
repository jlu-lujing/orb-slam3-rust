use opencv::core::{KeyPoint, Point2f};
use opencv::prelude::*;
use std::sync::Arc;
use std::{cmp::Ordering, collections::HashSet};

use crate::frame::Frame;
use crate::key_frame::KeyFrame;
use crate::map_point::MapPoint;
use nalgebra::{Isometry3, Matrix3, Point3, Similarity3, Translation3};

pub(crate) const TH_HIGH: i32 = 100;
pub(crate) const TH_LOW: i32 = 50;
const HISTO_LENGTH: usize = 30;

pub struct OrbMatcher {
    nn_ratio: f32,
    check_orientation: bool,
}

impl Default for OrbMatcher {
    fn default() -> Self {
        OrbMatcher {
            nn_ratio: 0.6,
            check_orientation: true,
        }
    }
}
impl OrbMatcher {
    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    pub fn search_by_projection_keypoints_mappoints(
        &self,
        f: &mut Frame,
        map_points: &[Arc<MapPoint>],
        th: f32,            // default 3.0
        far_points: bool,   // default false
        th_far_points: f32, // default 50.0f
    ) -> i32 {
        let mut n_matches = 0;
        let factor = th != 1.0;

        for mp in map_points.iter() {
            if !mp.track_in_view && !mp.track_in_view_r {
                continue;
            }
            if far_points && mp.track_depth > th_far_points {
                continue;
            }
            if mp.is_bad() {
                continue;
            }
            if mp.track_in_view {
                let predicted_level = mp.track_scale_level;

                // The size of the window will depend on the viewing direction
                let mut r = radius_by_viewing_cos(mp.track_view_cos);

                if factor {
                    r *= th;
                }

                let indices = f.get_features_in_area(
                    mp.track_proj_x,
                    mp.track_proj_y,
                    r * f.scale_factors[predicted_level as usize],
                    predicted_level - 1,
                    predicted_level,
                    false,
                );

                if !indices.is_empty() {
                    let mp_descriptor = mp.get_descriptor();

                    let mut best_dist = 256;
                    let mut best_level = None;
                    let mut best_dist2 = 256;
                    let mut best_level2 = None;
                    let mut best_idx = None;

                    // Get best and second matches with near keypoints
                    for idx in indices.into_iter() {
                        if let Some(Some(it)) = f.map_points.get(idx) {
                            if it.observations() > 0 {
                                continue;
                            }
                        }

                        if f.n_left.is_none() && f.u_right.get(idx).is_some_and(|&v| v > 0.0) {
                            let er = (mp.track_proj_xr - f.u_right[idx]).abs();
                            if er > r * f.scale_factors[predicted_level as usize] {
                                continue;
                            }
                        }

                        let d = f.descriptors.row(idx as i32).expect("get row");

                        let dist = descriptor_distance(&mp_descriptor, &d);

                        let octave_at = |idx: usize| -> i32 {
                            if f.n_left.is_none() {
                                f.keys_un.as_ref().unwrap()[idx].octave()
                            } else {
                                if idx < f.n_left.unwrap() {
                                    f.keys[idx].octave()
                                } else {
                                    f.keys_right.as_ref().unwrap()[idx - f.n_left.unwrap()].octave()
                                }
                            }
                        };
                        if dist < best_dist {
                            best_dist2 = best_dist;
                            best_dist = dist;
                            best_level2 = best_level;
                            best_level = Some(octave_at(idx));
                            best_idx = Some(idx);
                        } else if dist < best_dist2 {
                            best_level2 = Some(octave_at(idx));
                            best_dist2 = dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if best_dist <= TH_HIGH {
                        if best_level == best_level2
                            && best_dist as f32 > self.nn_ratio * best_dist2 as f32
                        {
                            continue;
                        }

                        // bestLevel != bestLevel2 || bestDist <= nnratio*bestDist2
                        // (always holds here, since the inverse case `continue`d
                        // above). Assign the MapPoint to the matched keypoint.
                        let best_idx = best_idx.expect("best_idx set when best_dist <= TH_HIGH");
                        f.map_points[best_idx] = Some(mp.clone());

                        // Also match with the stereo observation at the right camera (stereo fisheye only).
                        let right_partner = match (f.n_left, f.left_to_right_match.as_ref()) {
                            (Some(n_left), Some(l2r)) if l2r[best_idx] != usize::MAX => {
                                Some(l2r[best_idx] + n_left)
                            }
                            _ => None,
                        };
                        if let Some(right_idx) = right_partner {
                            f.map_points[right_idx] = Some(mp.clone());
                            n_matches += 1;
                        }

                        n_matches += 1;
                    }
                }
            }

            // Right-camera projection (stereo fisheye only).
            if let Some(n_left) = f.n_left {
                if mp.track_in_view_r {
                    let predicted_level = mp.track_scale_level_r;
                    if predicted_level == -1 {
                        continue;
                    }

                    // The size of the window will depend on the viewing direction
                    let r = radius_by_viewing_cos(mp.track_view_cos_r);

                    let indices = f.get_features_in_area(
                        mp.track_proj_xr,
                        mp.track_proj_yr,
                        r * f.scale_factors[predicted_level as usize],
                        predicted_level - 1,
                        predicted_level,
                        true,
                    );

                    if indices.is_empty() {
                        continue;
                    }

                    let mp_descriptor = mp.get_descriptor();

                    let mut best_dist = 256;
                    let mut best_level = None;
                    let mut best_dist2 = 256;
                    let mut best_level2 = None;
                    let mut best_idx = None;

                    // Get best and second matches with near keypoints
                    for idx in indices.into_iter() {
                        if let Some(Some(it)) = f.map_points.get(idx + n_left) {
                            if it.observations() > 0 {
                                continue;
                            }
                        }

                        let d = f.descriptors.row((idx + n_left) as i32).expect("get row");
                        let dist = descriptor_distance(&mp_descriptor, &d);

                        let octave = f.keys_right.as_ref().unwrap()[idx].octave();
                        if dist < best_dist {
                            best_dist2 = best_dist;
                            best_dist = dist;
                            best_level2 = best_level;
                            best_level = Some(octave);
                            best_idx = Some(idx);
                        } else if dist < best_dist2 {
                            best_level2 = Some(octave);
                            best_dist2 = dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if best_dist <= TH_HIGH {
                        if best_level == best_level2
                            && best_dist as f32 > self.nn_ratio * best_dist2 as f32
                        {
                            continue;
                        }

                        let best_idx = best_idx.expect("best_idx set when best_dist <= TH_HIGH");

                        // Also match with the stereo observation at the left
                        // camera. `usize::MAX` is the "no match" sentinel.
                        let left_partner = match f.right_to_left_match.as_ref() {
                            Some(r2l) if r2l[best_idx] != usize::MAX => Some(r2l[best_idx]),
                            _ => None,
                        };
                        if let Some(left_idx) = left_partner {
                            f.map_points[left_idx] = Some(mp.clone());
                            n_matches += 1;
                        }

                        f.map_points[best_idx + n_left] = Some(mp.clone());
                        n_matches += 1;
                    }
                }
            }
        }
        n_matches
    }

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    pub fn search_by_projection_last_frame(
        &self,
        current_frame: &mut Frame,
        last_frame: &Frame,
        th: f32, // default 3.0
        mono: bool,
    ) -> i32 {
        let mut n_matches = 0;

        // Rotation Histogram (to check rotation consistency)
        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        let t_cw = current_frame.get_pose();
        let t_wc = t_cw.inverse();
        let twc = t_wc.translation.vector;

        let t_lw = last_frame.get_pose();
        let tlc = t_lw * twc;

        let forward = tlc.z > current_frame.b && !mono;
        let backward = -tlc.z > current_frame.b && !mono;

        for i in 0..last_frame.n {
            let mp = last_frame.map_points.get(i);
            if let Some(Some(mp)) = mp {
                if !last_frame.outlier[i] {
                    // Project
                    let x_3dw = mp.get_world_pos();
                    let x_3dc = t_cw * x_3dw;

                    let inv_zc = 1.0 / x_3dc[2];

                    if inv_zc < 0. {
                        continue;
                    }

                    let uv = current_frame.camera.project_n(&Point3::from(x_3dc));

                    let bounds = current_frame.constants.bounds;
                    if uv[0] < bounds.min_x || uv[0] > bounds.max_x {
                        continue;
                    }
                    if uv[1] < bounds.min_y || uv[1] > bounds.max_y {
                        continue;
                    }

                    let last_octave =
                        if last_frame.n_left.is_none() || i < last_frame.n_left.unwrap() {
                            last_frame.keys[i].octave()
                        } else {
                            last_frame.keys_right.as_ref().expect("missing keys")
                                [i - last_frame.n_left.expect("missing n left")]
                            .octave()
                        };

                    // Search in a window, size depends on scale
                    let radius = th * current_frame.scale_factors[last_octave as usize];

                    let indices2 = {
                        let (min_level, max_level) = if forward {
                            (last_octave, -1)
                        } else if backward {
                            (0, last_octave)
                        } else {
                            (last_octave - 1, last_octave + 1)
                        };
                        current_frame
                            .get_features_in_area(uv[0], uv[1], radius, min_level, max_level, false)
                    };

                    if indices2.is_empty() {
                        continue;
                    }

                    let d_mp = mp.get_descriptor();

                    let mut best_dist = 256;
                    let mut best_idx2 = 0;

                    for i2 in indices2 {
                        if let Some(Some(mp)) = current_frame.map_points.get(i2) {
                            if mp.observations() > 0 {
                                continue;
                            }
                        }
                        if current_frame.n_left.is_none() && current_frame.u_right[i2] > 0. {
                            let ur = uv[0] - current_frame.b_fx * inv_zc;
                            let er = (ur - current_frame.u_right[i2]).abs();
                            if er > radius {
                                continue;
                            }
                        }
                        let d = current_frame
                            .descriptors
                            .row(i2 as i32)
                            .expect("missing descriptor");
                        let dist = descriptor_distance(&d_mp, &d);
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx2 = i2;
                        }
                    }

                    if best_dist <= TH_HIGH {
                        *current_frame.map_points.get_mut(best_idx2).unwrap() = Some(mp.clone());
                        n_matches += 1;

                        if self.check_orientation {
                            let kp_lf = if let Some(n_left) = last_frame.n_left {
                                if i < n_left {
                                    last_frame.keys.get(i).unwrap()
                                } else {
                                    last_frame
                                        .keys_right
                                        .as_ref()
                                        .expect("missing keys right")
                                        .get(i - n_left)
                                        .unwrap()
                                }
                            } else {
                                last_frame
                                    .keys_un
                                    .as_ref()
                                    .expect("missing keys un")
                                    .get(i)
                                    .unwrap()
                            };
                            let kp_cf = if let Some(n_left) = current_frame.n_left {
                                if best_idx2 < n_left {
                                    current_frame.keys.get(best_idx2).unwrap()
                                } else {
                                    current_frame
                                        .keys_right
                                        .as_ref()
                                        .expect("missing keys right")
                                        .get(best_idx2 - n_left)
                                        .unwrap()
                                }
                            } else {
                                current_frame
                                    .keys_un
                                    .as_ref()
                                    .expect("missing keys un")
                                    .get(best_idx2)
                                    .unwrap()
                            };

                            let mut rot = kp_lf.angle() - kp_cf.angle();
                            if rot < 0. {
                                rot += 360.;
                            }
                            let mut bin = (rot * factor).round() as usize;
                            if bin == HISTO_LENGTH {
                                bin = 0;
                            }
                            debug_assert!(bin < HISTO_LENGTH);
                            rot_hist[bin].push(best_idx2);
                        }
                    }
                    if let Some(n_left) = current_frame.n_left {
                        let x_3dr = current_frame.get_relative_pose_trl() * x_3dc;
                        let x_3dc_point = Point3::from(x_3dr);
                        let uv = current_frame.camera.project_n(&x_3dc_point);

                        let last_octave =
                            if last_frame.n_left.is_none() || i < last_frame.n_left.unwrap() {
                                last_frame.keys[i].octave()
                            } else {
                                last_frame.keys_right.as_ref().expect("missing keys")
                                    [i - last_frame.n_left.expect("missing n left")]
                                .octave()
                            };

                        // Search in a window. Size depend on scale
                        let radius = th * current_frame.scale_factors[last_octave as usize];

                        let indices2 = {
                            let (min_level, max_level) = if forward {
                                (last_octave, -1)
                            } else if backward {
                                (0, last_octave)
                            } else {
                                (last_octave - 1, last_octave + 1)
                            };
                            current_frame.get_features_in_area(
                                uv[0], uv[1], radius, min_level, max_level, true,
                            )
                        };

                        let d_mp = mp.get_descriptor();

                        let mut best_dist = 256;
                        let mut best_idx2 = 0;

                        for i2 in indices2 {
                            if let Some(Some(mp)) = current_frame.map_points.get(i2 + n_left) {
                                if mp.observations() > 0 {
                                    continue;
                                }
                            }
                            let d = current_frame
                                .descriptors
                                .row((i2 + n_left) as i32)
                                .expect("missing descriptor");
                            let dist = descriptor_distance(&d_mp, &d);
                            if dist < best_dist {
                                best_dist = dist;
                                best_idx2 = i2;
                            }
                        }
                        if best_dist <= TH_HIGH {
                            *current_frame
                                .map_points
                                .get_mut(best_idx2 + n_left)
                                .unwrap() = Some(mp.clone());
                            n_matches += 1;

                            if self.check_orientation {
                                let kp_lf = if let Some(n_left) = last_frame.n_left {
                                    if i < n_left {
                                        last_frame.keys.get(i).unwrap()
                                    } else {
                                        last_frame
                                            .keys_right
                                            .as_ref()
                                            .expect("missing keys right")
                                            .get(i - n_left)
                                            .unwrap()
                                    }
                                } else {
                                    last_frame
                                        .keys_un
                                        .as_ref()
                                        .expect("missing keys un")
                                        .get(i)
                                        .unwrap()
                                };
                                let kp_cf = current_frame
                                    .keys_right
                                    .as_ref()
                                    .unwrap()
                                    .get(best_idx2)
                                    .unwrap();

                                let mut rot = kp_lf.angle() - kp_cf.angle();
                                if rot < 0. {
                                    rot += 360.;
                                }
                                let mut bin = (rot * factor).round() as usize;
                                if bin == HISTO_LENGTH {
                                    bin = 0;
                                }
                                debug_assert!(bin < HISTO_LENGTH);
                                rot_hist[bin].push(best_idx2 + n_left);
                            }
                        }
                    }
                }
            }
        }

        // Apply rotation consistency
        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for j in 0..rot_hist[i].len() {
                        current_frame.map_points[rot_hist[i][j]] = None;
                        n_matches -= 1;
                    }
                }
            }
        }

        n_matches
    }

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    pub fn search_by_projection_keyframe(
        &self,
        current_frame: &mut Frame,
        kf: &KeyFrame,
        already_found: HashSet<Arc<MapPoint>>,
        th: f32, // default 3.0
        orb_dist: i32,
    ) -> i32 {
        let mut n_matches = 0;

        let t_cw = current_frame.get_pose();
        let t_wc = t_cw.inverse();
        let ow = t_wc.translation.vector;

        // Rotation Histogram (to check rotation consistency)
        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        let mps = kf.get_map_point_matches();

        for (i, mp) in mps.into_iter().enumerate() {
            if let Some(mp) = mp
                && !mp.is_bad()
                && !already_found.contains(&mp)
            {
                // Project
                let x_3dw = mp.get_world_pos();
                let x_3dc = t_cw * x_3dw;

                let uv = current_frame.camera.project_n(&Point3::from(x_3dc));

                let bounds = current_frame.constants.bounds;
                if uv[0] < bounds.min_x || uv[0] > bounds.max_x {
                    continue;
                }
                if uv[1] < bounds.min_y || uv[1] > bounds.max_y {
                    continue;
                }

                // Compute predicted scale level
                let po = x_3dw - ow;
                let dist3d = po.norm();

                let max_distance = mp.get_max_distance_invariance();
                let min_distance = mp.get_min_distance_invariance();

                // Depth must be inside the scale pyramid of the image
                if dist3d < min_distance || dist3d > max_distance {
                    continue;
                }

                let predicted_level = mp.predict_scale(dist3d, current_frame);

                // Search in a window
                let radius = th * current_frame.scale_factors[predicted_level];

                let indices2 = current_frame.get_features_in_area(
                    uv[0],
                    uv[1],
                    radius,
                    predicted_level as i32 - 1,
                    predicted_level as i32 + 1,
                    false,
                );
                if indices2.is_empty() {
                    continue;
                }

                let d_mp = mp.get_descriptor();

                let mut best_dist = 256;
                let mut best_idx2 = 0;

                for i2 in indices2 {
                    // Skip keypoints that already have a MapPoint.
                    if current_frame
                        .map_points
                        .get(i2)
                        .is_some_and(|inner| inner.is_some())
                    {
                        continue;
                    }

                    let d = current_frame.descriptors.row(i2 as i32).unwrap();

                    let dist = descriptor_distance(&d_mp, &d);

                    if dist < best_dist {
                        best_dist = dist;
                        best_idx2 = i2;
                    }
                }

                if best_dist <= orb_dist {
                    if let Some(p) = current_frame.map_points.get_mut(best_idx2) {
                        *p = Some(mp);
                        n_matches += 1;
                    }

                    if self.check_orientation {
                        let mut rot = kf.keys_un.get(i).unwrap().angle()
                            - current_frame
                                .keys_un
                                .as_ref()
                                .unwrap()
                                .get(best_idx2)
                                .unwrap()
                                .angle();
                        if rot < 0. {
                            rot += 360.;
                        }
                        let mut bin = (rot * factor).round() as usize;
                        if bin == HISTO_LENGTH {
                            bin = 0;
                        }
                        debug_assert!(bin < HISTO_LENGTH);
                        rot_hist[bin].push(best_idx2);
                    }
                }
            }
        }

        // Apply rotation consistency
        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for j in 0..rot_hist[i].len() {
                        current_frame.map_points[rot_hist[i][j]] = None;
                        n_matches -= 1;
                    }
                }
            }
        }

        n_matches
    }

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    pub fn search_by_projection_similarity_loop_detection(
        &self,
        kf: &KeyFrame,
        scw: &Similarity3<f32>,
        points: &[Arc<MapPoint>],
        matched: &mut [Option<Arc<MapPoint>>],
        th: f32,
        ratio_hamming: f32, // default 1.0
    ) -> i32 {
        let mut n_matches = 0;

        let t_cw = Isometry3::from_parts(
            Translation3::from(scw.isometry.translation.vector / scw.scaling()),
            scw.isometry.rotation,
        );
        let t_wc = t_cw.inverse();
        let ow = t_wc.translation.vector;

        // Set of MapPoints already found in the KeyFrame
        let already_found: HashSet<_> =
            matched.iter().filter_map(|m| m.as_ref().cloned()).collect();

        // For each Candidate MapPoint Project and Match
        for mp in points {
            // Discard bad MapPoints and already found
            if mp.is_bad() || already_found.contains(mp) {
                continue;
            }

            // Get 3D Coords.
            let p3dw = mp.get_world_pos();

            // Transform into Camera Coords
            let p3dc = t_cw * p3dw;

            // Depth must be positive
            if p3dc[2] < 0. {
                continue;
            }

            // Project into Image
            let uv = kf.camera.project_n(&Point3::from(p3dc));

            // Point must be inside the image
            if !kf.is_in_image(uv[0], uv[1]) {
                continue;
            }

            // Depth must be inside the scale invariance region of the point
            let max_distance = mp.get_max_distance_invariance();
            let min_distance = mp.get_min_distance_invariance();
            let po = p3dw - ow;
            let dist = po.norm();

            if dist < min_distance || dist > max_distance {
                continue;
            }

            // Viewing angle must be less than 60 deg
            let pn = mp.get_normal();
            if po.dot(&pn) < 0.5 * dist {
                continue;
            }

            let predicted_level = mp.predict_scale_keyframe(dist, kf);

            // Search in radius
            let radius = th * kf.scale_factors[predicted_level];

            let indices = kf.get_features_in_area(uv[0], uv[1], radius, false);
            if indices.is_empty() {
                continue;
            }

            // Match to the most similar keypoint in the radius
            let d_mp = mp.get_descriptor();

            let mut best_dist = 256;
            let mut best_idx: i32 = -1;

            for idx in indices {
                if matched[idx].is_some() {
                    continue;
                }
                let kp_level = kf.keys_un[idx].octave() as usize;
                if kp_level < predicted_level - 1 || kp_level > predicted_level {
                    continue;
                }
                let d_kf = kf.descriptors.row(idx as i32).unwrap();
                let dist = descriptor_distance(&d_mp, &d_kf);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx as i32;
                }
            }

            if best_dist as f32 <= (TH_LOW as f32 * ratio_hamming) {
                matched[best_idx as usize] = Some(mp.clone());
                n_matches += 1;
            }
        }

        n_matches
    }

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in Place Recognition (Loop Closing and Merging)
    pub fn search_by_projection_similarity_loop_closing_merging(
        &self,
        kf: &KeyFrame,
        scw: &Similarity3<f32>,
        points: &[Arc<MapPoint>],
        points_kf: &[Arc<KeyFrame>],
        matched: &mut [Option<Arc<MapPoint>>],
        matched_kf: &mut [Option<Arc<KeyFrame>>],
        th: f32,
        ratio_hamming: f32, // default 1.0
    ) -> i32 {
        let mut n_matches = 0;

        // Get calibration parameters for later projection
        let fx = kf.fx;
        let fy = kf.fy;
        let cx = kf.cx;
        let cy = kf.cy;

        let t_cw = Isometry3::from_parts(
            Translation3::from(scw.isometry.translation.vector / scw.scaling()),
            scw.isometry.rotation,
        );
        let t_wc = t_cw.inverse();
        let ow = t_wc.translation.vector;

        // Set of MapPoints already found in the KeyFrame
        let already_found: HashSet<_> =
            matched.iter().filter_map(|m| m.as_ref().cloned()).collect();

        // For each Candidate MapPoint Project and Match
        for (mp, kfi) in points.iter().zip(points_kf.iter()) {
            // Discard bad MapPoints and already found
            if mp.is_bad() || already_found.contains(mp) {
                continue;
            }

            // Get 3D Coords.
            let p3dw = mp.get_world_pos();

            // Transform into Camera Coords
            let p3dc = t_cw * p3dw;

            // Depth must be positive
            if p3dc[2] < 0. {
                continue;
            }

            // Project into Image
            let invz = 1. / p3dc[2];
            let x = p3dc[0] * invz;
            let y = p3dc[1] * invz;
            let u = fx * x + cx;
            let v = fy * y + cy;

            // Point must be inside the image
            if !kf.is_in_image(u, v) {
                continue;
            }

            // Depth must be inside the scale invariance region of the point
            let max_distance = mp.get_max_distance_invariance();
            let min_distance = mp.get_min_distance_invariance();
            let po = p3dw - ow;
            let dist = po.norm();

            if dist < min_distance || dist > max_distance {
                continue;
            }

            // Viewing angle must be less than 60 deg
            let pn = mp.get_normal();
            if po.dot(&pn) < 0.5 * dist {
                continue;
            }

            let predicted_level = mp.predict_scale_keyframe(dist, kf);

            // Search in radius
            let radius = th * kf.scale_factors[predicted_level];

            let indices = kf.get_features_in_area(u, v, radius, false);
            if indices.is_empty() {
                continue;
            }

            // Match to the most similar keypoint in the radius
            let d_mp = mp.get_descriptor();

            let mut best_dist = 256;
            let mut best_idx: i32 = -1;

            for idx in indices {
                if matched[idx].is_some() {
                    continue;
                }
                let kp_level = kf.keys_un[idx].octave() as usize;
                if kp_level < predicted_level - 1 || kp_level > predicted_level {
                    continue;
                }
                let d_kf = kf.descriptors.row(idx as i32).unwrap();
                let dist = descriptor_distance(&d_mp, &d_kf);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx as i32;
                }
            }

            if best_dist as f32 <= (TH_LOW as f32 * ratio_hamming) {
                matched[best_idx as usize] = Some(mp.clone());
                matched_kf[best_idx as usize] = Some(kfi.clone());
                n_matches += 1;
            }
        }

        n_matches
    }

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    fn search_by_bow_frame(
        &self,
        kf: &KeyFrame,
        f: &Frame,
        mappoint_matches: &mut Vec<Option<Arc<MapPoint>>>,
    ) -> i32 {
        let mut n_matches = 0;

        let map_points_kf = kf.get_map_point_matches();

        // Output is index-aligned with the Frame's keypoints.
        mappoint_matches.clear();
        mappoint_matches.resize(f.n, None);

        let feat_vec_kf = &kf.feat_vec;

        // Rotation Histogram (to check rotation consistency)
        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        // Select the KeyFrame keypoint for index `idx`
        let kf_keypoint = |idx: usize| -> &KeyPoint {
            if kf.camera2.is_none() {
                &kf.keys_un[idx]
            } else if let Some(n_left) = kf.n_left
                && idx >= n_left
            {
                &kf.keys_right.as_ref().expect("keys_right")[idx - n_left]
            } else {
                &kf.keys[idx]
            }
        };

        // Frame keypoint for a *left* match: uses the KeyFrame's second camera
        // flag, matching `(!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[..]`.
        let f_keypoint_left = |idx: usize| -> &KeyPoint {
            if kf.camera2.is_some()
                && let Some(n_left) = f.n_left
                && idx >= n_left
            {
                &f.keys_right.as_ref().expect("keys_right")[idx - n_left]
            } else {
                &f.keys[idx]
            }
        };

        // Frame keypoint for a *right* match: uses the Frame's second camera
        // flag, matching `(!F.mpCamera2) ? F.mvKeys[..]`.
        let f_keypoint_right = |idx: usize| -> &KeyPoint {
            if f.camera2.is_some()
                && let Some(n_left) = f.n_left
                && idx >= n_left
            {
                &f.keys_right.as_ref().expect("keys_right")[idx - n_left]
            } else {
                &f.keys[idx]
            }
        };

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        let mut kf_it = feat_vec_kf.0.iter().peekable();
        let mut f_it = f.feat_vec.0.iter().peekable();

        while let (Some(kf_entry), Some(f_entry)) = (kf_it.peek(), f_it.peek()) {
            let (&kf_node, kf_indices) = *kf_entry;
            let (&f_node, f_indices) = *f_entry;
            match kf_node.cmp(&f_node) {
                Ordering::Equal => {
                    for &real_idx_kf in kf_indices {
                        let real_idx_kf = real_idx_kf as usize;

                        let Some(mp) = map_points_kf.get(real_idx_kf).and_then(|m| m.as_ref())
                        else {
                            continue;
                        };
                        if mp.is_bad() {
                            continue;
                        }

                        let d_kf = kf
                            .descriptors
                            .row(real_idx_kf as i32)
                            .expect("kf descriptor");

                        // Best/second-best for the left image, and (stereo
                        // fisheye) for the right image.
                        let mut best_dist1 = 256;
                        let mut best_idx_f: i32 = -1;
                        let mut best_dist2 = 256;
                        let mut best_dist1_r = 256;
                        let mut best_idx_fr: i32 = -1;
                        let mut best_dist2_r = 256;

                        for &real_idx_f in f_indices {
                            let real_idx_f = real_idx_f as usize;

                            if mappoint_matches[real_idx_f].is_some() {
                                continue;
                            }

                            let d_f = f.descriptors.row(real_idx_f as i32).expect("f descriptor");
                            let dist = descriptor_distance(&d_kf, &d_f);

                            match f.n_left {
                                // Monocular frame.
                                None => {
                                    if dist < best_dist1 {
                                        best_dist2 = best_dist1;
                                        best_dist1 = dist;
                                        best_idx_f = real_idx_f as i32;
                                    } else if dist < best_dist2 {
                                        best_dist2 = dist;
                                    }
                                }
                                // Stereo fisheye: left indices track the left
                                // best, right indices track the right best.
                                Some(n_left) => {
                                    if real_idx_f < n_left {
                                        if dist < best_dist1 {
                                            best_dist2 = best_dist1;
                                            best_dist1 = dist;
                                            best_idx_f = real_idx_f as i32;
                                        } else if dist < best_dist2 {
                                            best_dist2 = dist;
                                        }
                                    } else if dist < best_dist1_r {
                                        best_dist2_r = best_dist1_r;
                                        best_dist1_r = dist;
                                        best_idx_fr = real_idx_f as i32;
                                    } else if dist < best_dist2_r {
                                        best_dist2_r = dist;
                                    }
                                }
                            }
                        }

                        if best_dist1 <= TH_LOW {
                            // Left match: standard Lowe ratio test.
                            if (best_dist1 as f32) < self.nn_ratio * best_dist2 as f32 {
                                let best_idx_f = best_idx_f as usize;
                                mappoint_matches[best_idx_f] = Some(mp.clone());

                                if self.check_orientation {
                                    let mut rot = kf_keypoint(real_idx_kf).angle()
                                        - f_keypoint_left(best_idx_f).angle();
                                    if rot < 0.0 {
                                        rot += 360.0;
                                    }
                                    let mut bin = (rot * factor).round() as usize;
                                    if bin == HISTO_LENGTH {
                                        bin = 0;
                                    }
                                    debug_assert!(bin < HISTO_LENGTH);
                                    rot_hist[bin].push(best_idx_f);
                                }
                                n_matches += 1;
                            }

                            // Right match (stereo fisheye)
                            if best_dist1_r <= TH_LOW {
                                let best_idx_fr = best_idx_fr as usize;
                                mappoint_matches[best_idx_fr] = Some(mp.clone());

                                if self.check_orientation {
                                    let mut rot = kf_keypoint(real_idx_kf).angle()
                                        - f_keypoint_right(best_idx_fr).angle();
                                    if rot < 0.0 {
                                        rot += 360.0;
                                    }
                                    let mut bin = (rot * factor).round() as usize;
                                    if bin == HISTO_LENGTH {
                                        bin = 0;
                                    }
                                    debug_assert!(bin < HISTO_LENGTH);
                                    rot_hist[bin].push(best_idx_fr);
                                }
                                n_matches += 1;
                            }
                        }
                    }

                    kf_it.next();
                    f_it.next();
                }
                // KFit->first < Fit->first  =>  advance KF
                Ordering::Less => {
                    kf_it.next();
                }
                // else  =>  advance F
                Ordering::Greater => {
                    f_it.next();
                }
            }
        }

        // Apply rotation consistency: drop matches outside the three dominant
        // orientation bins.
        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for j in 0..rot_hist[i].len() {
                        mappoint_matches[rot_hist[i][j]] = None;
                        n_matches -= 1;
                    }
                }
            }
        }

        n_matches
    }

    // Search matches between MapPoints already associated to two KeyFrames,
    // constrained to ORB that belong to the same vocabulary node.
    // Used in Loop Detection.
    fn search_by_bow_keyframe(
        &self,
        kf1: &KeyFrame,
        kf2: &KeyFrame,
        matches12: &mut Vec<Option<Arc<MapPoint>>>,
    ) -> i32 {
        let keys_un1 = &kf1.keys_un;
        let keys_un2 = &kf2.keys_un;
        let feat_vec1 = &kf1.feat_vec;
        let feat_vec2 = &kf2.feat_vec;
        let map_points1 = kf1.get_map_point_matches();
        let map_points2 = kf2.get_map_point_matches();

        // matches12[i] is the MapPoint in kf2 matched to kf1's keypoint i.
        matches12.clear();
        matches12.resize(map_points1.len(), None);
        let mut matched2 = vec![false; map_points2.len()];

        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        let mut n_matches = 0;

        let mut f1_it = feat_vec1.0.iter().peekable();
        let mut f2_it = feat_vec2.0.iter().peekable();

        while let (Some(entry1), Some(entry2)) = (f1_it.peek(), f2_it.peek()) {
            let (&node1, indices1) = *entry1;
            let (&node2, indices2) = *entry2;
            match node1.cmp(&node2) {
                Ordering::Equal => {
                    for &idx1 in indices1 {
                        let idx1 = idx1 as usize;
                        // Skip right-camera indices (stereo fisheye).
                        if kf1.n_left.is_some() && idx1 >= keys_un1.len() {
                            continue;
                        }

                        let Some(mp1) = map_points1.get(idx1).and_then(|m| m.as_ref()) else {
                            continue;
                        };
                        if mp1.is_bad() {
                            continue;
                        }

                        let d1 = kf1.descriptors.row(idx1 as i32).expect("kf1 descriptor");

                        let mut best_dist1 = 256;
                        let mut best_idx2: i32 = -1;
                        let mut best_dist2 = 256;

                        for &idx2 in indices2 {
                            let idx2 = idx2 as usize;
                            // Skip right-camera indices (stereo fisheye).
                            if kf2.n_left.is_some() && idx2 >= keys_un2.len() {
                                continue;
                            }

                            let Some(mp2) = map_points2.get(idx2).and_then(|m| m.as_ref()) else {
                                continue;
                            };
                            if matched2[idx2] || mp2.is_bad() {
                                continue;
                            }

                            let d2 = kf2.descriptors.row(idx2 as i32).expect("kf2 descriptor");
                            let dist = descriptor_distance(&d1, &d2);

                            if dist < best_dist1 {
                                best_dist2 = best_dist1;
                                best_dist1 = dist;
                                best_idx2 = idx2 as i32;
                            } else if dist < best_dist2 {
                                best_dist2 = dist;
                            }
                        }

                        if best_dist1 < TH_LOW
                            && (best_dist1 as f32) < self.nn_ratio * best_dist2 as f32
                        {
                            let best_idx2 = best_idx2 as usize;
                            matches12[idx1] = map_points2[best_idx2].clone();
                            matched2[best_idx2] = true;

                            if self.check_orientation {
                                let mut rot = keys_un1[idx1].angle() - keys_un2[best_idx2].angle();
                                if rot < 0.0 {
                                    rot += 360.0;
                                }
                                let mut bin = (rot * factor).round() as usize;
                                if bin == HISTO_LENGTH {
                                    bin = 0;
                                }
                                debug_assert!(bin < HISTO_LENGTH);
                                rot_hist[bin].push(idx1);
                            }
                            n_matches += 1;
                        }
                    }

                    f1_it.next();
                    f2_it.next();
                }
                Ordering::Less => {
                    f1_it.next();
                }
                Ordering::Greater => {
                    f2_it.next();
                }
            }
        }

        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for j in 0..rot_hist[i].len() {
                        matches12[rot_hist[i][j]] = None;
                        n_matches -= 1;
                    }
                }
            }
        }

        n_matches
    }

    // Matching for the Map Initialization (only used in the monocular case)
    fn search_for_initialization(
        &self,
        f1: &Frame,
        f2: &Frame,
        prev_matched: &mut [Point2f],
        window_size: i32, // default 10
    ) -> (Vec<i32>, i32) {
        let mut n_matches = 0;

        let mut matches12 = vec![-1; f1.keys_un.as_ref().unwrap().len()];

        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        let f2_keys_un_len = f2.keys_un.as_ref().unwrap().len();
        let mut matched_distance = vec![i32::MAX; f2_keys_un_len];
        let mut matches21: Vec<i32> = vec![-1; f2_keys_un_len];

        for (i1, kp1) in f1.keys_un.as_ref().unwrap().iter().enumerate() {
            let level1 = kp1.octave();
            if level1 > 0 {
                continue;
            }

            let indices2 = f2.get_features_in_area(
                prev_matched[i1].x,
                prev_matched[i1].y,
                window_size as f32,
                level1,
                level1,
                false,
            );
            if indices2.is_empty() {
                continue;
            }

            let d1 = f1.descriptors.row(i1 as i32).unwrap();

            let mut best_dist = i32::MAX;
            let mut best_dist2 = i32::MAX;
            let mut best_idx2 = -1;

            for i2 in indices2 {
                let d2 = f2.descriptors.row(i2 as i32).unwrap();
                let dist = descriptor_distance(&d1, &d2);
                if matched_distance[i2] <= dist {
                    continue;
                }
                if dist < best_dist {
                    best_dist2 = best_dist;
                    best_dist = dist;
                    best_idx2 = i2 as i32;
                } else if dist < best_dist2 {
                    best_dist2 = dist;
                }
            }

            if best_dist <= TH_LOW {
                if (best_dist as f32) < best_dist2 as f32 * self.nn_ratio {
                    if matches21[best_idx2 as usize] >= 0 {
                        matches12[matches21[best_idx2 as usize] as usize] = -1;
                        n_matches -= 1;
                    }
                    matches12[i1] = best_idx2;
                    matches21[best_idx2 as usize] = i1 as i32;
                    matched_distance[best_idx2 as usize] = best_dist;
                    n_matches += 1;

                    if self.check_orientation {
                        let mut rot = f1.keys_un.as_ref().unwrap()[i1].angle()
                            - f2.keys_un.as_ref().unwrap()[best_idx2 as usize].angle();
                        if rot < 0. {
                            rot += 360.;
                        }
                        let mut bin = (rot * factor).round() as usize;
                        if bin == HISTO_LENGTH {
                            bin = 0;
                        }
                        debug_assert!(bin < HISTO_LENGTH);
                        rot_hist[bin].push(i1);
                    }
                }
            }
        }

        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for j in 0..rot_hist[i].len() {
                        let idx1 = rot_hist[i][j];
                        if matches12[idx1] >= 0 {
                            matches12[idx1] = -1;
                            n_matches -= 1;
                        }
                    }
                }
            }
        }

        //Update prev matched
        for (i1, &i2) in matches12.iter().enumerate() {
            if i2 >= 0 {
                prev_matched[i1] = f2.keys_un.as_ref().unwrap()[i2 as usize].pt();
            }
        }

        (matches12, n_matches)
    }

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    fn search_for_triangulation(
        &self,
        kf1: &KeyFrame,
        kf2: &KeyFrame,
        matched_pairs: &mut Vec<(usize, usize)>,
        only_stereo: bool,
        coarse: bool,
    ) -> i32 {
        let feat_vec1 = &kf1.feat_vec;
        let feat_vec2 = &kf2.feat_vec;

        // Compute epipole in the second image
        let t1w = kf1.get_pose();
        let t2w = kf2.get_pose();
        let tw2 = kf2.get_pose_inverse(); // for convenience
        let cw = kf1.get_camera_center();
        let c2 = t2w * Point3::from(cw);
        let ep = kf2.camera.project_n(&c2);

        // Decompose an isometry into (R, t)
        let decompose = |iso: &Isometry3<f32>| -> (Matrix3<f32>, Point3<f32>) {
            (
                iso.rotation.to_rotation_matrix().into_inner(),
                Point3::from(iso.translation.vector),
            )
        };

        // Relative pose(s) between the keyframes. The standard (pinhole /
        // rectified-stereo) case uses a single T12; stereo-fisheye needs the
        // four left/right camera combinations (ll, lr, rl, rr).
        let mut r12 = Matrix3::<f32>::identity();
        let mut t12 = Point3::<f32>::origin();
        let mut rll = Matrix3::<f32>::identity();
        let mut tll = Point3::<f32>::origin();
        let mut rlr = Matrix3::<f32>::identity();
        let mut tlr = Point3::<f32>::origin();
        let mut rrl = Matrix3::<f32>::identity();
        let mut trl = Point3::<f32>::origin();
        let mut rrr = Matrix3::<f32>::identity();
        let mut trr = Point3::<f32>::origin();

        if kf1.camera2.is_none() && kf2.camera2.is_none() {
            (r12, t12) = decompose(&(t1w * tw2));
        } else {
            let tr1w = kf1.get_right_pose();
            let twr2 = kf2.get_right_pose_inverse();
            (rll, tll) = decompose(&(t1w * tw2));
            (rlr, tlr) = decompose(&(t1w * twr2));
            (rrl, trl) = decompose(&(tr1w * tw2));
            (rrr, trr) = decompose(&(tr1w * twr2));
        }

        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node
        let map_points1 = kf1.get_map_point_matches();
        let map_points2 = kf2.get_map_point_matches();

        let mut n_matches = 0;
        let mut matches12 = vec![-1i32; kf1.n as usize];

        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        let mut f1_it = feat_vec1.0.iter().peekable();
        let mut f2_it = feat_vec2.0.iter().peekable();

        while let (Some(entry1), Some(entry2)) = (f1_it.peek(), f2_it.peek()) {
            let (&node1, indices1) = *entry1;
            let (&node2, indices2) = *entry2;
            match node1.cmp(&node2) {
                Ordering::Equal => {
                    for &idx1 in indices1 {
                        let idx1 = idx1 as usize;

                        // Skip if this keypoint already has a MapPoint.
                        if map_points1[idx1].is_some() {
                            continue;
                        }

                        let stereo1 = kf1.camera2.is_none() && kf1.u_right[idx1] >= 0.0;
                        if only_stereo && !stereo1 {
                            continue;
                        }

                        let kp1 = select_keypoint(kf1, idx1);
                        let right1 = is_right(kf1, idx1);
                        let d1 = kf1.descriptors.row(idx1 as i32).expect("kf1 descriptor");

                        let mut best_dist = TH_LOW;
                        let mut best_idx2: i32 = -1;

                        for &idx2 in indices2 {
                            let idx2 = idx2 as usize;

                            // Skip if already matched (via a MapPoint).
                            if map_points2[idx2].is_some() {
                                continue;
                            }

                            let stereo2 = kf2.camera2.is_none() && kf2.u_right[idx2] >= 0.0;
                            if only_stereo && !stereo2 {
                                continue;
                            }

                            let d2 = kf2.descriptors.row(idx2 as i32).expect("kf2 descriptor");
                            let dist = descriptor_distance(&d1, &d2);
                            if dist > TH_LOW || dist > best_dist {
                                continue;
                            }

                            let kp2 = select_keypoint(kf2, idx2);
                            let right2 = is_right(kf2, idx2);

                            // For non-stereo, non-fisheye matches discard
                            // candidates too close to the epipole.
                            if !stereo1 && !stereo2 && kf1.camera2.is_none() {
                                let distex = ep.x - kp2.pt().x;
                                let distey = ep.y - kp2.pt().y;
                                if distex * distex + distey * distey
                                    < 100.0 * kf2.scale_factors[kp2.octave() as usize]
                                {
                                    continue;
                                }
                            }

                            // Pick the relative pose / cameras for this pair.
                            let (r12_sel, t12_sel, cam1, cam2) =
                                if kf1.camera2.is_some() && kf2.camera2.is_some() {
                                    match (right1, right2) {
                                        (true, true) => (
                                            rrr,
                                            trr,
                                            kf1.camera2.as_ref().unwrap(),
                                            kf2.camera2.as_ref().unwrap(),
                                        ),
                                        (true, false) => {
                                            (rrl, trl, kf1.camera2.as_ref().unwrap(), &kf2.camera)
                                        }
                                        (false, true) => {
                                            (rlr, tlr, &kf1.camera, kf2.camera2.as_ref().unwrap())
                                        }
                                        (false, false) => (rll, tll, &kf1.camera, &kf2.camera),
                                    }
                                } else {
                                    (r12, t12, &kf1.camera, &kf2.camera)
                                };

                            if coarse
                                || cam1.epipolar_constrain(
                                    cam2.as_ref(),
                                    kp1,
                                    kp2,
                                    &r12_sel,
                                    &t12_sel,
                                    kf1.level_sigma2[kp1.octave() as usize],
                                    kf2.level_sigma2[kp2.octave() as usize],
                                )
                            {
                                best_idx2 = idx2 as i32;
                                best_dist = dist;
                            }
                        }

                        if best_idx2 >= 0 {
                            let best_idx2 = best_idx2 as usize;
                            let kp2 = select_keypoint(kf2, best_idx2);
                            matches12[idx1] = best_idx2 as i32;
                            n_matches += 1;

                            if self.check_orientation {
                                let mut rot = kp1.angle() - kp2.angle();
                                if rot < 0.0 {
                                    rot += 360.0;
                                }
                                let mut bin = (rot * factor).round() as usize;
                                if bin == HISTO_LENGTH {
                                    bin = 0;
                                }
                                debug_assert!(bin < HISTO_LENGTH);
                                rot_hist[bin].push(idx1);
                            }
                        }
                    }

                    f1_it.next();
                    f2_it.next();
                }
                Ordering::Less => {
                    f1_it.next();
                }
                Ordering::Greater => {
                    f2_it.next();
                }
            }
        }

        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for &idx1 in &rot_hist[i] {
                        matches12[idx1] = -1;
                        n_matches -= 1;
                    }
                }
            }
        }

        matched_pairs.clear();
        matched_pairs.reserve(n_matches as usize);
        for (i, &m) in matches12.iter().enumerate() {
            if m >= 0 {
                matched_pairs.push((i, m as usize));
            }
        }

        n_matches
    }

    // Project MapPoints into a KeyFrame and search for duplicated MapPoints,
    // fusing them. `right` selects the second (right) fisheye camera.
    // Used in Local Mapping.
    pub fn fuse(
        &self,
        kf: &Arc<KeyFrame>,
        map_points: &[Option<Arc<MapPoint>>],
        th: f32,
        right: bool,
    ) -> i32 {
        let (tcw, ow, camera) = if right {
            (
                kf.get_right_pose(),
                kf.get_right_camera_center(),
                kf.camera2.as_ref().expect("camera2"),
            )
        } else {
            (kf.get_pose(), kf.get_camera_center(), &kf.camera)
        };
        let bf = kf.bf;

        let mut n_fused = 0;

        for mp in map_points.iter().flatten() {
            if mp.is_bad() || mp.is_in_keyframe(kf) {
                continue;
            }

            let p3dw = mp.get_world_pos();
            let p3dc = tcw * Point3::from(p3dw);

            // Depth must be positive
            if p3dc[2] < 0.0 {
                continue;
            }

            let invz = 1.0 / p3dc[2];
            let uv = camera.project_n(&p3dc);

            // Point must be inside the image
            if !kf.is_in_image(uv.x, uv.y) {
                continue;
            }

            let ur = uv.x - bf * invz;

            let max_distance = mp.get_max_distance_invariance();
            let min_distance = mp.get_min_distance_invariance();
            let po = p3dw - ow;
            let dist3d = po.norm();

            // Depth must be inside the scale pyramid of the image
            if dist3d < min_distance || dist3d > max_distance {
                continue;
            }

            // Viewing angle must be less than 60 deg
            let pn = mp.get_normal();
            if po.dot(&pn) < 0.5 * dist3d {
                continue;
            }

            let predicted_level = mp.predict_scale_keyframe(dist3d, kf);
            let radius = th * kf.scale_factors[predicted_level];
            let indices = kf.get_features_in_area(uv.x, uv.y, radius, right);
            if indices.is_empty() {
                continue;
            }

            // Match to the most similar keypoint in the radius
            let d_mp = mp.get_descriptor();

            let mut best_dist = 256;
            let mut best_idx: i32 = -1;
            for idx in indices {
                let kp = match kf.n_left {
                    None => &kf.keys_un[idx],
                    Some(_) if !right => &kf.keys[idx],
                    Some(_) => &kf.keys_right.as_ref().expect("keys_right")[idx],
                };
                let kp_level = kp.octave();
                if kp_level < predicted_level as i32 - 1 || kp_level > predicted_level as i32 {
                    continue;
                }

                let ex = uv.x - kp.pt().x;
                let ey = uv.y - kp.pt().y;
                if kf.u_right[idx] >= 0.0 {
                    // Check reprojection error in stereo
                    let er = ur - kf.u_right[idx];
                    let e2 = ex * ex + ey * ey + er * er;
                    if e2 * kf.inv_level_sigma2[kp_level as usize] > 7.8 {
                        continue;
                    }
                } else {
                    let e2 = ex * ex + ey * ey;
                    if e2 * kf.inv_level_sigma2[kp_level as usize] > 5.99 {
                        continue;
                    }
                }

                // Right keypoints are stored after the left ones in the
                // descriptor matrix.
                let idx = if right {
                    idx + kf.n_left.expect("n_left")
                } else {
                    idx
                };
                let d_kf = kf.descriptors.row(idx as i32).expect("kf descriptor");
                let dist = descriptor_distance(&d_mp, &d_kf);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx as i32;
                }
            }

            // If there is already a MapPoint replace, otherwise add a new one.
            if best_dist <= TH_LOW {
                if let Some(mp_in_kf) = kf.get_map_point(best_idx as usize) {
                    if !mp_in_kf.is_bad() {
                        if mp_in_kf.observations() > mp.observations() {
                            mp.replace(&mp_in_kf);
                        } else {
                            mp_in_kf.replace(mp);
                        }
                    }
                } else {
                    mp.add_observation(kf, best_idx);
                    kf.add_map_point(mp.clone(), best_idx as usize);
                }
                n_fused += 1;
            }
        }

        n_fused
    }

    // Project MapPoints into a KeyFrame using a Sim3 transform and search for
    // duplications. Matches found against an existing MapPoint are recorded in
    // `replace_points` (to be replaced by the caller). Used in Loop Closing.
    pub fn fuse_by_sim3(
        &self,
        kf: &Arc<KeyFrame>,
        scw: &Similarity3<f32>,
        points: &[Arc<MapPoint>],
        th: f32,
        replace_points: &mut [Option<Arc<MapPoint>>],
    ) -> i32 {
        // Decompose Scw into an SE3 (scale folded into the translation).
        let t_cw = Isometry3::from_parts(
            Translation3::from(scw.isometry.translation.vector / scw.scaling()),
            scw.isometry.rotation,
        );
        let ow = t_cw.inverse().translation.vector;

        // Set of MapPoints already found in the KeyFrame
        let already_found = kf.get_map_points();

        let mut n_fused = 0;

        for (i_mp, mp) in points.iter().enumerate() {
            // Discard bad MapPoints and already found
            if mp.is_bad() || already_found.contains(mp) {
                continue;
            }

            let p3dw = mp.get_world_pos();
            let p3dc = t_cw * Point3::from(p3dw);

            // Depth must be positive
            if p3dc[2] < 0.0 {
                continue;
            }

            let uv = kf.camera.project_n(&p3dc);
            if !kf.is_in_image(uv.x, uv.y) {
                continue;
            }

            let max_distance = mp.get_max_distance_invariance();
            let min_distance = mp.get_min_distance_invariance();
            let po = p3dw - ow;
            let dist3d = po.norm();
            if dist3d < min_distance || dist3d > max_distance {
                continue;
            }

            // Viewing angle must be less than 60 deg
            let pn = mp.get_normal();
            if po.dot(&pn) < 0.5 * dist3d {
                continue;
            }

            let predicted_level = mp.predict_scale_keyframe(dist3d, kf);
            let radius = th * kf.scale_factors[predicted_level];
            let indices = kf.get_features_in_area(uv.x, uv.y, radius, false);
            if indices.is_empty() {
                continue;
            }

            let d_mp = mp.get_descriptor();

            let mut best_dist = i32::MAX;
            let mut best_idx: i32 = -1;
            for idx in indices {
                let kp_level = kf.keys_un[idx].octave();
                if kp_level < predicted_level as i32 - 1 || kp_level > predicted_level as i32 {
                    continue;
                }
                let d_kf = kf.descriptors.row(idx as i32).expect("kf descriptor");
                let dist = descriptor_distance(&d_mp, &d_kf);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx as i32;
                }
            }

            // If there is already a MapPoint replace, otherwise add a new one.
            if best_dist <= TH_LOW {
                if let Some(mp_in_kf) = kf.get_map_point(best_idx as usize) {
                    if !mp_in_kf.is_bad() {
                        replace_points[i_mp] = Some(mp_in_kf);
                    }
                } else {
                    mp.add_observation(kf, best_idx);
                    kf.add_map_point(mp.clone(), best_idx as usize);
                }
                n_fused += 1;
            }
        }

        n_fused
    }

    // Search matches between MapPoints of two KeyFrames given a Sim3
    // transformation S12 (from KF2 to KF1). `matches12` is both input (already
    // known matches) and output. Used in Loop Closing.
    pub fn search_by_sim3(
        &self,
        kf1: &KeyFrame,
        kf2: &KeyFrame,
        matches12: &mut [Option<Arc<MapPoint>>],
        s12: &Similarity3<f32>,
        th: f32,
    ) -> i32 {
        // The original uses KF1's intrinsics for both projections.
        let fx = kf1.fx;
        let fy = kf1.fy;
        let cx = kf1.cx;
        let cy = kf1.cy;

        // Camera 1 & 2 from world
        let t1w = kf1.get_pose();
        let t2w = kf2.get_pose();

        // Transformation between cameras
        let s21 = s12.inverse();

        let map_points1 = kf1.get_map_point_matches();
        let map_points2 = kf2.get_map_point_matches();
        let n1 = map_points1.len();
        let n2 = map_points2.len();

        let mut already_matched1 = vec![false; n1];
        let mut already_matched2 = vec![false; n2];

        for (i, m) in matches12.iter().enumerate().take(n1) {
            if let Some(mp) = m {
                already_matched1[i] = true;
                let idx2 = mp.get_index_in_keyframe(kf2).0;
                if idx2 >= 0 && (idx2 as usize) < n2 {
                    already_matched2[idx2 as usize] = true;
                }
            }
        }

        let mut match1 = vec![-1i32; n1];
        let mut match2 = vec![-1i32; n2];

        // Transform from KF1 to KF2 and search
        for i1 in 0..n1 {
            let Some(mp) = &map_points1[i1] else {
                continue;
            };
            if already_matched1[i1] || mp.is_bad() {
                continue;
            }

            let p3dw = mp.get_world_pos();
            let p3dc1 = t1w * Point3::from(p3dw);
            let p3dc2 = s21 * p3dc1;

            if p3dc2[2] < 0.0 {
                continue;
            }

            let invz = 1.0 / p3dc2[2];
            let u = fx * p3dc2[0] * invz + cx;
            let v = fy * p3dc2[1] * invz + cy;
            if !kf2.is_in_image(u, v) {
                continue;
            }

            let max_distance = mp.get_max_distance_invariance();
            let min_distance = mp.get_min_distance_invariance();
            let dist3d = p3dc2.coords.norm();
            if dist3d < min_distance || dist3d > max_distance {
                continue;
            }

            let predicted_level = mp.predict_scale_keyframe(dist3d, kf2);
            let radius = th * kf2.scale_factors[predicted_level];
            let indices = kf2.get_features_in_area(u, v, radius, false);
            if indices.is_empty() {
                continue;
            }

            let d_mp = mp.get_descriptor();
            let mut best_dist = i32::MAX;
            let mut best_idx: i32 = -1;
            for idx in indices {
                let octave = kf2.keys_un[idx].octave();
                if octave < predicted_level as i32 - 1 || octave > predicted_level as i32 {
                    continue;
                }
                let d_kf = kf2.descriptors.row(idx as i32).expect("kf2 descriptor");
                let dist = descriptor_distance(&d_mp, &d_kf);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx as i32;
                }
            }
            if best_dist <= TH_HIGH {
                match1[i1] = best_idx;
            }
        }

        // Transform from KF2 to KF1 and search
        for i2 in 0..n2 {
            let Some(mp) = &map_points2[i2] else {
                continue;
            };
            if already_matched2[i2] || mp.is_bad() {
                continue;
            }

            let p3dw = mp.get_world_pos();
            let p3dc2 = t2w * Point3::from(p3dw);
            let p3dc1 = s12 * p3dc2;

            if p3dc1[2] < 0.0 {
                continue;
            }

            let invz = 1.0 / p3dc1[2];
            let u = fx * p3dc1[0] * invz + cx;
            let v = fy * p3dc1[1] * invz + cy;
            if !kf1.is_in_image(u, v) {
                continue;
            }

            let max_distance = mp.get_max_distance_invariance();
            let min_distance = mp.get_min_distance_invariance();
            let dist3d = p3dc1.coords.norm();
            if dist3d < min_distance || dist3d > max_distance {
                continue;
            }

            let predicted_level = mp.predict_scale_keyframe(dist3d, kf1);
            let radius = th * kf1.scale_factors[predicted_level];
            let indices = kf1.get_features_in_area(u, v, radius, false);
            if indices.is_empty() {
                continue;
            }

            let d_mp = mp.get_descriptor();
            let mut best_dist = i32::MAX;
            let mut best_idx: i32 = -1;
            for idx in indices {
                let octave = kf1.keys_un[idx].octave();
                if octave < predicted_level as i32 - 1 || octave > predicted_level as i32 {
                    continue;
                }
                let d_kf = kf1.descriptors.row(idx as i32).expect("kf1 descriptor");
                let dist = descriptor_distance(&d_mp, &d_kf);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx as i32;
                }
            }
            if best_dist <= TH_HIGH {
                match2[i2] = best_idx;
            }
        }

        // Check agreement (mutual best match).
        let mut n_found = 0;
        for i1 in 0..n1 {
            let idx2 = match1[i1];
            if idx2 >= 0 {
                let idx1 = match2[idx2 as usize];
                if idx1 == i1 as i32 {
                    matches12[i1] = map_points2[idx2 as usize].clone();
                    n_found += 1;
                }
            }
        }

        n_found
    }
}

// Selects the keypoint for `idx`, picking between the undistorted / left /
// right keypoint vectors depending on the stereo-fisheye configuration.
fn select_keypoint(kf: &KeyFrame, idx: usize) -> &KeyPoint {
    match kf.n_left {
        None => &kf.keys_un[idx],
        Some(n_left) if idx < n_left => &kf.keys[idx],
        Some(n_left) => &kf.keys_right.as_ref().expect("keys_right")[idx - n_left],
    }
}

// Whether `idx` refers to a right-image keypoint (stereo fisheye only).
fn is_right(kf: &KeyFrame, idx: usize) -> bool {
    matches!(kf.n_left, Some(n_left) if idx >= n_left)
}

// Computes the Hamming distance between two ORB descriptors
// Original code uses bit-twiddling here but count_ones() should be efficient as is
pub(crate) fn descriptor_distance(a: &impl MatTraitConst, b: &impl MatTraitConst) -> i32 {
    let pa: &[u8] = a.data_bytes().expect("descriptor a");
    let pb: &[u8] = b.data_bytes().expect("descriptor b");
    pa.iter()
        .zip(pb)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>() as i32
}

fn radius_by_viewing_cos(view_cos: f32) -> f32 {
    // Compare in f64 to match C++ exactly: there `viewCos` (a `float`) is
    // promoted to `double` for the comparison against the `0.998` double
    // literal, so e.g. `0.998f` (= 0.998000025…) compares as `> 0.998`.
    if view_cos as f64 > 0.998 { 2.5 } else { 4.0 }
}

fn compute_three_maxima<T>(histo: &[Vec<T>]) -> [Option<usize>; 3] {
    let mut top = [(0usize, None); 3];
    for (i, bucket) in histo.iter().enumerate() {
        let item = (bucket.len(), Some(i));
        if item.0 > top[0].0 {
            top = [item, top[0], top[1]];
        } else if item.0 > top[1].0 {
            top = [top[0], item, top[1]];
        } else if item.0 > top[2].0 {
            top[2] = item;
        }
    }
    let threshold = top[0].0 as f32 * 0.1;
    [
        top[0].1,
        top[1].1.filter(|_| top[1].0 as f32 >= threshold),
        top[2].1.filter(|_| top[2].0 as f32 >= threshold),
    ]
}

#[cfg(test)]
mod tests {
    // TODO: this is not complete yet, need integration tests once other parts of mapping are more fleshed out
    #![allow(clippy::excessive_precision)]
    use opencv::core::{CV_8U, MatTrait, Scalar};

    use super::*;
    use crate::test_helpers::*;

    /// A 1x32 CV_8U descriptor with `bytes[i] = val` for the listed indices.
    fn desc(nonzero: &[i32], val: u8) -> opencv::core::Mat {
        let mut d =
            opencv::core::Mat::new_rows_cols_with_default(1, 32, CV_8U, Scalar::all(0.0)).unwrap();
        for &i in nonzero {
            *d.at_2d_mut::<u8>(0, i).unwrap() = val;
        }
        d
    }

    #[test]
    fn descriptor_distance_hamming() {
        let zero = desc(&[], 0);
        let ones = desc(&(0..32).collect::<Vec<_>>(), 0xFF);

        assert_eq!(descriptor_distance(&zero, &zero), 0);
        assert_eq!(descriptor_distance(&zero, &ones), 256);
        assert_eq!(descriptor_distance(&ones, &ones), 0);

        // One byte 0xFF -> 8 bits.
        assert_eq!(descriptor_distance(&zero, &desc(&[0], 0xFF)), 8);
        // 32 bytes of 0x01 -> 32 bits.
        assert_eq!(
            descriptor_distance(&zero, &desc(&(0..32).collect::<Vec<_>>(), 0x01)),
            32
        );
        // 0xB2 (10110010) -> 4 bits.
        assert_eq!(descriptor_distance(&zero, &desc(&[0], 0xB2)), 4);
        // 0xB2 ^ 0x46 = 0xF4 (11110100) -> 5 bits.
        assert_eq!(descriptor_distance(&desc(&[0], 0xB2), &desc(&[0], 0x46)), 5);
    }

    #[test]
    fn radius_by_viewing_cos_boundaries() {
        assert_eq!(radius_by_viewing_cos(1.0), 2.5);
        assert_eq!(radius_by_viewing_cos(0.999), 2.5);
        assert_eq!(radius_by_viewing_cos(0.998), 2.5);
        assert_eq!(radius_by_viewing_cos(0.99), 4.0);
        assert_eq!(radius_by_viewing_cos(0.5), 4.0);
    }

    #[test]
    fn compute_three_maxima_cases() {
        let mut h: Vec<Vec<u8>> = vec![Vec::new(); HISTO_LENGTH];

        // Clear top three: bin0=10, bin1=5, bin2=8.
        h[0] = vec![0; 10];
        h[1] = vec![0; 5];
        h[2] = vec![0; 8];
        assert_eq!(compute_three_maxima(&h), [Some(0), Some(2), Some(1)]);

        // Second below the 10% threshold -> 2nd and 3rd nullified.
        let mut h2: Vec<Vec<u8>> = vec![Vec::new(); HISTO_LENGTH];
        h2[4] = vec![0; 100];
        h2[7] = vec![0; 5];
        h2[9] = vec![0; 3];
        assert_eq!(compute_three_maxima(&h2), [Some(4), None, None]);

        // Third below threshold -> only 3rd nullified.
        let mut h3: Vec<Vec<u8>> = vec![Vec::new(); HISTO_LENGTH];
        h3[4] = vec![0; 100];
        h3[7] = vec![0; 50];
        h3[9] = vec![0; 3];
        assert_eq!(compute_three_maxima(&h3), [Some(4), Some(7), None]);
    }

    #[test]
    fn constants() {
        assert_eq!(TH_LOW, 50);
        assert_eq!(TH_HIGH, 100);
        assert_eq!(HISTO_LENGTH, 30);
    }

    #[test]
    fn search_for_initialization_self_matches() {
        // Two frames from the same (non-repetitive) image -> identical
        // extraction, so every matched octave-0 keypoint matches itself.
        let f1 = build_frame_from(&make_noise_image());
        let f2 = build_frame_from(&make_noise_image());
        assert!(f1.n > 0);

        let mut prev: Vec<Point2f> = f1
            .keys_un
            .as_ref()
            .unwrap()
            .iter()
            .map(|k| k.pt())
            .collect();

        let matcher = OrbMatcher::default();
        let (matches12, n) = matcher.search_for_initialization(&f1, &f2, &mut prev, 10);

        assert!(n > 0, "expected some matches");
        let mut self_matches = 0;
        for (i, &m) in matches12.iter().enumerate() {
            if m >= 0 {
                assert_eq!(m as usize, i, "match must be a self-match");
                self_matches += 1;
            }
        }
        assert_eq!(self_matches, n);
    }
}
