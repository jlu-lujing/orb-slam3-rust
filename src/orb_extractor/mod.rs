mod brief_pattern;

use opencv::core::{
    _InputArrayTraitConst, BORDER_ISOLATED, BORDER_REFLECT_101, CV_8U, CV_8UC1, KeyPoint,
    KeyPointTrait, KeyPointTraitConst, Mat, MatTraitConst, Point2f, Point2i, Rect, Scalar, Size,
    ToInputArray, Vector, copy_make_border,
};
use opencv::features2d::fast;
use opencv::imgproc::{INTER_LINEAR, gaussian_blur, resize};
use opencv::prelude::*;

use crate::{
    compat::{cv_ceil_f32, cv_floor_f32, cv_round_f32},
    orb_extractor::brief_pattern::{BRIEF_PATTERN, BRIEF_PATTERN_LEN},
};

const PATCH_SIZE: usize = 31;
const HALF_PATCH_SIZE: usize = 15;
const EDGE_THRESHOLD: i32 = 19;

#[derive(Default)]
struct ExtractorNode {
    pub keys: Vec<KeyPoint>,
    pub ul: Point2i,
    pub ur: Point2i,
    pub bl: Point2i,
    pub br: Point2i,
    pub no_more: bool,
}

impl ExtractorNode {
    pub fn divide_node(&self) -> [ExtractorNode; 4] {
        let half_x = ((self.ur.x - self.ul.x) as f32 / 2.0).ceil() as i32;
        let half_y = ((self.br.y - self.ul.y) as f32 / 2.0).ceil() as i32;

        let mut n1 = ExtractorNode::default();
        let mut n2 = ExtractorNode::default();
        let mut n3 = ExtractorNode::default();
        let mut n4 = ExtractorNode::default();

        // Top-left
        n1.ul = self.ul;
        n1.ur = Point2i::new(self.ul.x + half_x, self.ul.y);
        n1.bl = Point2i::new(self.ul.x, self.ul.y + half_y);
        n1.br = Point2i::new(self.ul.x + half_x, self.ul.y + half_y);

        // Top-right
        n2.ul = n1.ur;
        n2.ur = self.ur;
        n2.bl = n1.br;
        n2.br = Point2i::new(self.ur.x, self.ul.y + half_y);

        // Bottom-left
        n3.ul = n1.bl;
        n3.ur = n1.br;
        n3.bl = self.bl;
        n3.br = Point2i::new(n1.br.x, self.bl.y);

        // Bottom-right
        n4.ul = n3.ur;
        n4.ur = n2.br;
        n4.bl = n3.br;
        n4.br = self.br;

        n1.keys.reserve(self.keys.len());
        n2.keys.reserve(self.keys.len());
        n3.keys.reserve(self.keys.len());
        n4.keys.reserve(self.keys.len());

        for kp in &self.keys {
            let pt = kp.pt();

            if pt.x < n1.ur.x as f32 {
                if pt.y < n1.br.y as f32 {
                    n1.keys.push(kp.clone());
                } else {
                    n3.keys.push(kp.clone());
                }
            } else if pt.y < n1.br.y as f32 {
                n2.keys.push(kp.clone());
            } else {
                n4.keys.push(kp.clone());
            }
        }

        for node in [&mut n1, &mut n2, &mut n3, &mut n4] {
            if node.keys.len() == 1 {
                node.no_more = true;
            }
        }

        [n1, n2, n3, n4]
    }
}

#[derive(Default)]
pub struct OrbExtractor {
    features: usize,
    scale_factor: f64,
    levels: usize,
    ini_th_fast: i32,
    min_th_fast: i32,
    features_per_level: Vec<usize>,
    umax: Vec<i32>,
    v_scale_factor: Vec<f32>,
    v_inv_scale_factor: Vec<f32>,
    v_level_sigma2: Vec<f32>,
    v_inv_level_sigma2: Vec<f32>,
}

pub enum Score {
    Harris,
    Fast,
}

#[derive(Debug)]
pub enum ExtractionError {
    EmptyImage,
    InvalidInput,
    InvalidInputType,
}

pub struct OrbExtractResult {
    pub descriptors: Option<Mat>,
    pub keypoints: Vec<KeyPoint>,
    pub mono_index: i32,
    pub image_pyramid: Vec<Mat>,
}

impl OrbExtractor {
    pub fn new(
        features: usize,
        scale_factor: f32,
        levels: usize,
        ini_th_fast: i32,
        min_th_fast: i32,
    ) -> Self {
        let mut v_scale_factor = vec![0.; levels];
        let mut v_level_sigma2 = vec![0.; levels];
        v_scale_factor[0] = 1.;
        v_level_sigma2[0] = 1.;
        for i in 1..levels {
            v_scale_factor[i] = v_scale_factor[i - 1] * scale_factor;
            v_level_sigma2[i] = v_scale_factor[i] * v_scale_factor[i];
        }

        let mut v_inv_scale_factor = vec![0.; levels];
        let mut v_inv_level_sigma2 = vec![0.; levels];
        for i in 0..levels {
            v_inv_scale_factor[i] = 1. / v_scale_factor[i];
            v_inv_level_sigma2[i] = 1. / v_level_sigma2[i];
        }

        let mut features_per_level = Vec::with_capacity(levels);
        let inv_factor = 1. / scale_factor;
        let mut desired_features_per_scale =
            features as f32 * (1. - inv_factor) / (1.0 - inv_factor.powf(levels as f32));

        let mut sum_features = 0;
        for _ in 0..(levels - 1) {
            let val = cv_round_f32(desired_features_per_scale);
            sum_features += val;
            features_per_level.push(val as usize);
            desired_features_per_scale *= inv_factor;
        }
        features_per_level.push((features as i32 - sum_features).max(0) as usize);

        // This is for orientation
        // Pre-compute the end of a row in a sircular patch
        let mut umax = vec![0; HALF_PATCH_SIZE + 1];
        let vmax = cv_floor_f32(HALF_PATCH_SIZE as f32 * 2.0_f32.sqrt() / 2. + 1.);
        let vmin = cv_ceil_f32(HALF_PATCH_SIZE as f32 * 2.0_f32.sqrt() / 2.);
        let hp2 = (HALF_PATCH_SIZE * HALF_PATCH_SIZE) as i32;
        for v in 0..=vmax {
            umax[v as usize] = cv_round_f32(((hp2 - v * v) as f32).sqrt())
        }

        // Make sure we are symmetric
        let mut v0: i32 = 0;
        for v in (vmin..=HALF_PATCH_SIZE as i32).rev() {
            while umax[v0 as usize] == umax[(v0 + 1) as usize] {
                v0 += 1;
            }
            umax[v as usize] = v0;
            v0 += 1;
        }

        OrbExtractor {
            features,
            scale_factor: scale_factor as f64,
            levels,
            ini_th_fast,
            min_th_fast,
            v_scale_factor,
            v_level_sigma2,
            v_inv_scale_factor,
            v_inv_level_sigma2,
            features_per_level,
            umax,
        }
    }

    /// Compute ORB features and descriptors on an image.
    ///
    /// Keypoints are dispersed using an octree across a Gaussian pyramid.
    /// Mask is ignored in the current implementation.
    ///
    /// Keypoints whose (level-0-scaled) x-coordinate falls inside
    /// `lapping_area = [xmin, xmax]` are packed at the tail of the output
    /// vector (for stereo-fisheye matching); the rest are packed at the head.
    /// The returned `mono_count` is the number of head (non-overlap) keypoints
    pub fn compute(
        &self,
        image: &impl ToInputArray,
        _mask: &impl ToInputArray,
        lapping_area: [i32; 2],
    ) -> Result<OrbExtractResult, ExtractionError> {
        let image_ia = image
            .input_array()
            .map_err(|_| ExtractionError::InvalidInput)?;
        if image_ia.empty().unwrap_or(true) {
            return Err(ExtractionError::EmptyImage);
        }

        let image_mat = image_ia
            .get_mat_def()
            .map_err(|_| ExtractionError::InvalidInput)?;
        if image_mat.typ() != CV_8UC1 {
            return Err(ExtractionError::InvalidInputType);
        }

        // Pre-compute the scale pyramid
        let image_pyramid = self.compute_pyramid(image_mat);

        let mut all_keypoints = self.compute_keypoints_oct_tree(&image_pyramid);

        let n_keypoints: i32 = all_keypoints.iter().map(|kp| kp.len() as i32).sum();
        if n_keypoints == 0 {
            return Ok(OrbExtractResult {
                descriptors: None,
                keypoints: Vec::new(),
                mono_index: 0,
                image_pyramid,
            });
        }

        let mut descriptors =
            Mat::new_rows_cols_with_default(n_keypoints, 32, CV_8U, Scalar::all(0.))
                .expect("allocate descriptors");
        // Pre-sized so we can scatter rows at mono/stereo indices in any order,
        // mirroring the C++ `_keypoints = vector<KeyPoint>(nkeypoints);` pattern.
        // `KeyPoint::default()` is fallible in the opencv bindings but its
        // default constructor cannot actually fail.
        let default_kp = KeyPoint::default().expect("KeyPoint::default");
        let mut keypoints: Vec<KeyPoint> = vec![default_kp; n_keypoints as usize];

        let lap_min = lapping_area[0] as f32;
        let lap_max = lapping_area[1] as f32;

        // Modified for speeding up stereo fisheye matching: non-overlap (mono)
        // keypoints fill from the front, overlap (stereo) keypoints from the back.
        let mut mono_index: i32 = 0;
        let mut stereo_index: i32 = n_keypoints - 1;

        for level in 0..self.levels {
            let kps = &mut all_keypoints[level];
            if kps.is_empty() {
                continue;
            }

            // Preprocess the resized image
            let mut blurred = Mat::default();
            gaussian_blur(
                &image_pyramid[level],
                &mut blurred,
                Size::new(7, 7),
                2.,
                2.,
                BORDER_REFLECT_101,
                opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )
            .unwrap();

            // Compute the descriptors for this level into a temporary Mat,
            // then scatter its rows into `descriptors` at mono/stereo slots.
            let desc = compute_descriptors(&blurred, kps).unwrap();
            let desc_bytes = desc.data_typed::<u8>().unwrap();
            let dst_bytes = descriptors.data_typed_mut::<u8>().unwrap();

            let scale = self.v_scale_factor[level];
            for (i, keypoint) in kps.iter_mut().enumerate() {
                // Scale keypoint coordinates back to level-0 image space.
                if level != 0 {
                    let mut pt = keypoint.pt();
                    pt *= scale;
                    keypoint.set_pt(pt);
                }

                let pt_x = keypoint.pt().x;
                let target = if pt_x >= lap_min && pt_x <= lap_max {
                    let idx = stereo_index;
                    stereo_index -= 1;
                    idx
                } else {
                    let idx = mono_index;
                    mono_index += 1;
                    idx
                } as usize;

                keypoints[target] = keypoint.clone();
                let src = &desc_bytes[i * 32..(i + 1) * 32];
                dst_bytes[target * 32..(target + 1) * 32].copy_from_slice(src);
            }
        }

        Ok(OrbExtractResult {
            descriptors: Some(descriptors),
            keypoints,
            mono_index,
            image_pyramid,
        })
    }

    fn compute_pyramid(&self, image: Mat) -> Vec<Mat> {
        let mut pyramid: Vec<Mat> = Vec::with_capacity(self.levels);
        for level in 0..self.levels {
            let scale = self.v_inv_scale_factor[level];
            let size = Size::new(
                cv_round_f32(image.cols() as f32 * scale),
                cv_round_f32(image.rows() as f32 * scale),
            );
            let whole_size = Size::new(
                size.width + EDGE_THRESHOLD * 2,
                size.height + EDGE_THRESHOLD * 2,
            );

            // In C++ `mvImagePyramid[level]` aliases a ROI of `temp`, so
            // `copyMakeBorder(temp, ...)` fills both the borders and (via the
            // earlier resize) the inner region in a single buffer. We can't
            // alias in Rust, so build `temp` first (resize into a scratch
            // buffer, then bake the borders into `temp`) and clone the inner
            // ROI for the pyramid. The keypoint/descriptor code only reads
            // pixels inside the inner image, so the border padding is
            // semantically a no-op for downstream consumers.
            let mut temp = Mat::new_size_with_default(whole_size, image.typ(), Scalar::all(0.))
                .expect("allocate mat");
            if level == 0 {
                copy_make_border(
                    &image,
                    &mut temp,
                    EDGE_THRESHOLD,
                    EDGE_THRESHOLD,
                    EDGE_THRESHOLD,
                    EDGE_THRESHOLD,
                    BORDER_REFLECT_101,
                    Scalar::default(),
                )
                .expect("copy make border");
            } else {
                let mut resized = Mat::default();
                resize(
                    &pyramid[level - 1],
                    &mut resized,
                    size,
                    0.,
                    0.,
                    INTER_LINEAR,
                )
                .expect("resize image");
                copy_make_border(
                    &resized,
                    &mut temp,
                    EDGE_THRESHOLD,
                    EDGE_THRESHOLD,
                    EDGE_THRESHOLD,
                    EDGE_THRESHOLD,
                    BORDER_REFLECT_101 + BORDER_ISOLATED,
                    Scalar::default(),
                )
                .expect("copy make border");
            }

            let inner = Mat::roi(
                &temp,
                Rect::new(EDGE_THRESHOLD, EDGE_THRESHOLD, size.width, size.height),
            )
            .expect("create roi");
            pyramid.push(inner.try_clone().expect("clone mat"));
        }
        pyramid
    }

    fn compute_keypoints_oct_tree(&self, pyramid: &[Mat]) -> Vec<Vec<KeyPoint>> {
        let mut all_keypoints: Vec<Vec<KeyPoint>> = Vec::with_capacity(self.levels);

        let w = 35.;

        for level in 0..self.levels {
            let min_border_x = EDGE_THRESHOLD - 3;
            let min_border_y = min_border_x;
            let max_border_x = pyramid[level].cols() - EDGE_THRESHOLD + 3;
            let max_border_y = pyramid[level].rows() - EDGE_THRESHOLD + 3;

            let mut to_distribute_keys: Vec<KeyPoint> = Vec::with_capacity(self.features * 10);

            let width = (max_border_x - min_border_x) as f32;
            let height = (max_border_y - min_border_y) as f32;

            let cols = (width / w) as i32;
            let rows = (height / w) as i32;
            let w_cell = (width / cols as f32).ceil();
            let h_cell = (height / rows as f32).ceil();

            for i in 0..rows {
                let ini_y = min_border_y as f32 + i as f32 * h_cell;
                let mut max_y = ini_y + h_cell + 6.0;
                if ini_y >= (max_border_y - 3) as f32 {
                    continue;
                }
                if max_y > max_border_y as f32 {
                    max_y = max_border_y as f32;
                }

                for j in 0..cols {
                    let ini_x = min_border_x as f32 + j as f32 * w_cell;
                    let mut max_x = ini_x + w_cell + 6.0;
                    if ini_x >= (max_border_x - 6) as f32 {
                        continue;
                    }
                    if max_x > max_border_x as f32 {
                        max_x = max_border_x as f32;
                    }

                    let mut keys_cell: Vector<KeyPoint> = Vector::new();

                    let roi_rect = Rect::new(
                        ini_x as i32,
                        ini_y as i32,
                        (max_x - ini_x) as i32,
                        (max_y - ini_y) as i32,
                    );
                    let cell = Mat::roi(&pyramid[level], roi_rect).expect("roi");

                    fast(&cell, &mut keys_cell, self.ini_th_fast, true).expect("FAST");

                    if keys_cell.is_empty() {
                        fast(&cell, &mut keys_cell, self.min_th_fast, true).expect("FAST");
                    }

                    if !keys_cell.is_empty() {
                        for it in keys_cell.iter() {
                            let mut keypoint = it;
                            let mut pt = keypoint.pt();
                            pt.x += j as f32 * w_cell;
                            pt.y += i as f32 * h_cell;
                            keypoint.set_pt(pt);
                            to_distribute_keys.push(keypoint);
                        }
                    }
                }
            }

            let mut keypoints = distribute_oct_tree(
                &to_distribute_keys,
                min_border_x,
                max_border_x,
                min_border_y,
                max_border_y,
                self.features_per_level[level],
            );

            // Add border to coordinates and scale information
            let scaled_patch_size = PATCH_SIZE as f32 * self.v_scale_factor[level];
            for kp in &mut keypoints {
                let mut pt = kp.pt();
                pt.x += min_border_x as f32;
                pt.y += min_border_y as f32;
                kp.set_pt(pt);
                kp.set_octave(level as i32);
                kp.set_size(scaled_patch_size);
            }

            // Compute orientation
            compute_orientation(&pyramid[level], &mut keypoints, &self.umax);

            all_keypoints.push(keypoints);
        }

        all_keypoints
    }

    pub fn get_levels(&self) -> usize {
        self.levels
    }
    pub fn get_scale_factor(&self) -> f32 {
        self.scale_factor as f32
    }
    pub fn get_scale_factors(&self) -> Vec<f32> {
        self.v_scale_factor.clone()
    }
    pub fn get_inverse_scale_factors(&self) -> Vec<f32> {
        self.v_inv_scale_factor.clone()
    }
    pub fn get_scale_sigma2(&self) -> Vec<f32> {
        self.v_level_sigma2.clone()
    }
    pub fn get_inverse_scale_sigma2(&self) -> Vec<f32> {
        self.v_inv_level_sigma2.clone()
    }
}

fn distribute_oct_tree(
    to_distribute_keys: &[KeyPoint],
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    n: usize,
) -> Vec<KeyPoint> {
    // Compute how many initial nodes
    let ini = ((max_x - min_x) as f32 / (max_y - min_y) as f32).round() as usize;

    let h_x = (max_x - min_x) as f32 / ini as f32;

    let mut nodes: Vec<ExtractorNode> = Vec::with_capacity(ini);

    for i in 0..ini {
        let ul = Point2i::new((h_x * i as f32) as i32, 0);
        let ur = Point2i::new((h_x * (i + 1) as f32) as i32, 0);
        nodes.push(ExtractorNode {
            keys: Vec::with_capacity(to_distribute_keys.len()),
            ul,
            ur,
            bl: Point2i::new(ul.x, max_y - min_y),
            br: Point2i::new(ur.x, max_y - min_y),
            no_more: false,
        });
    }

    // Associate points to childs. C++ indexes `vpIniNodes[kp.pt.x/hX]`
    // unconditionally; clamp here defensively because FAST keypoints can land
    // a few pixels past the cell right edge (cells extend by +6 px).
    for kp in to_distribute_keys {
        let pt = kp.pt();
        let idx = (pt.x / h_x).floor() as usize;
        let idx = idx.min(nodes.len() - 1);
        nodes[idx].keys.push(kp.clone());
    }

    // Remove empty nodes and mark single-point keynodes
    nodes.retain_mut(|node| match node.keys.len() {
        0 => false,
        1 => {
            node.no_more = true;
            true
        }
        _ => true,
    });

    loop {
        let prev_size = nodes.len();

        // Full-subdivision pass: every non-`no_more` node is split into 4.
        let old_nodes = std::mem::take(&mut nodes);
        let mut new_nodes: Vec<ExtractorNode> = Vec::with_capacity(old_nodes.len() * 4);
        // Indices (into `new_nodes`) of children produced in this pass that
        // have more than one keypoint and could still be subdivided. These
        // form the "frontier" used by the overshoot branch below.
        let mut frontier: Vec<usize> = Vec::new();
        let mut n_to_expand: usize = 0;

        for node in old_nodes {
            if node.no_more {
                new_nodes.push(node);
                continue;
            }
            for child in node.divide_node() {
                if child.keys.is_empty() {
                    continue;
                }
                let expandable = child.keys.len() > 1;
                let idx = new_nodes.len();
                new_nodes.push(child);
                if expandable {
                    n_to_expand += 1;
                    frontier.push(idx);
                }
            }
        }

        nodes = new_nodes;

        if nodes.len() >= n || nodes.len() == prev_size {
            break;
        }

        if nodes.len() + n_to_expand * 3 > n {
            // Overshoot branch: instead of another full subdivision (which
            // would push us well past `n` nodes), subdivide the frontier
            // selectively, largest-first, breaking ties by larger `ul.x`
            // first. This matches the `compareNodes` ordering in C++:
            //   sort ascending by (size, ul.x), then iterate from the back.
            loop {
                let prev_size_inner = nodes.len();
                frontier.sort_by_key(|&i| (nodes[i].keys.len(), nodes[i].ul.x));

                // Replace `nodes` with an `Option`-wrapped vec so we can
                // remove arbitrary indices without shifting positions.
                let mut slots: Vec<Option<ExtractorNode>> = nodes.drain(..).map(Some).collect();
                let mut new_children: Vec<ExtractorNode> = Vec::new();
                // Indices (into `new_children`) of grandchildren that are
                // themselves expandable; these become the next frontier.
                let mut next_frontier_local: Vec<usize> = Vec::new();
                let mut current_count = slots.len();

                for &i in frontier.iter().rev() {
                    let node = slots[i]
                        .take()
                        .expect("frontier index must point at a live node");
                    current_count -= 1;
                    for child in node.divide_node() {
                        if child.keys.is_empty() {
                            continue;
                        }
                        let expandable = child.keys.len() > 1;
                        let local = new_children.len();
                        new_children.push(child);
                        current_count += 1;
                        if expandable {
                            next_frontier_local.push(local);
                        }
                    }
                    if current_count >= n {
                        break;
                    }
                }

                // Compact survivors, then append the new children.
                let mut rebuilt: Vec<ExtractorNode> = Vec::with_capacity(current_count);
                for slot in slots {
                    if let Some(node) = slot {
                        rebuilt.push(node);
                    }
                }
                let survivors = rebuilt.len();
                rebuilt.extend(new_children);

                frontier = next_frontier_local
                    .into_iter()
                    .map(|local| survivors + local)
                    .collect();
                nodes = rebuilt;

                if nodes.len() >= n || nodes.len() == prev_size_inner || frontier.is_empty() {
                    break;
                }
            }
            break;
        }
    }

    // Retain the best point in each node
    let mut result_keys = Vec::with_capacity(n);
    for node in nodes {
        if let Some(best) = node.keys.into_iter().max_by(|a, b| {
            a.response()
                .partial_cmp(&b.response())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            result_keys.push(best);
        }
    }
    result_keys
}

fn compute_orientation(image: &Mat, keypoints: &mut [KeyPoint], umax: &[i32]) {
    for kp in keypoints {
        kp.set_angle(ic_angle(image, kp.pt(), umax).expect("compute orientation"));
    }
}

fn ic_angle(image: &Mat, pt: Point2f, u_max: &[i32]) -> opencv::Result<f32> {
    let mut m_01: i32 = 0;
    let mut m_10: i32 = 0;

    let cx = pt.x.round() as i32;
    let cy = pt.y.round() as i32;

    let step = image.step1(0)? as isize;

    // OpenCV Mat data as a flat byte slice.
    let data = image.data_bytes()?;

    let center_idx = cy as isize * step + cx as isize;

    // Treat the center line differently, v = 0
    let half_patch_size = HALF_PATCH_SIZE as i32;
    for u in -half_patch_size..=half_patch_size {
        let idx = center_idx + u as isize;
        let pixel = data[idx as usize] as i32;

        m_10 += u * pixel;
    }

    // Go line by line in the circular patch
    for v in 1..=half_patch_size {
        let mut v_sum: i32 = 0;
        let d = u_max[v as usize];

        for u in -d..=d {
            let idx_plus = center_idx + u as isize + v as isize * step;
            let idx_minus = center_idx + u as isize - v as isize * step;

            let val_plus = data[idx_plus as usize] as i32;
            let val_minus = data[idx_minus as usize] as i32;

            v_sum += val_plus - val_minus;
            m_10 += u * (val_plus + val_minus);
        }

        m_01 += v * v_sum;
    }

    // C++: return fastAtan2((float)m_01, (float)m_10);
    Ok(opencv::core::fast_atan2(m_01 as f32, m_10 as f32)?)
}

/// Compute 32-byte rBRIEF descriptors for every keypoint in `keypoints`.
///
/// `image` must be the Gaussian-smoothed level the keypoints belong to.
/// The sampling pattern is the static [`BRIEF_PATTERN`].
fn compute_descriptors(image: &Mat, keypoints: &[KeyPoint]) -> opencv::Result<Mat> {
    let mut descriptors =
        Mat::new_rows_cols_with_default(keypoints.len() as i32, 32, CV_8UC1, Scalar::all(0.0))?;

    {
        let desc_data = descriptors.data_typed_mut::<u8>()?;
        for (i, keypoint) in keypoints.iter().enumerate() {
            let row = &mut desc_data[i * 32..(i + 1) * 32];
            compute_orb_descriptor(keypoint, image, row)?;
        }
    }

    Ok(descriptors)
}

const FACTOR_PI: f32 = std::f32::consts::PI / 180.0;

/// Write the 32-byte (256-bit) steered rBRIEF descriptor for `kpt` into `desc`.
///
/// Each output byte packs 8 bit-tests from [`BRIEF_PATTERN`]; the test offsets
/// are rotated by the keypoint angle θ so the descriptor is rotation-invariant.
fn compute_orb_descriptor(kpt: &KeyPoint, img: &Mat, desc: &mut [u8]) -> opencv::Result<()> {
    const BITS_PER_BYTE: usize = 8;
    const DESC_BYTES: usize = BRIEF_PATTERN_LEN / BITS_PER_BYTE; // 32
    debug_assert_eq!(desc.len(), DESC_BYTES);

    let angle = kpt.angle() * FACTOR_PI;
    let a = angle.cos();
    let b = angle.sin();

    let pt = kpt.pt();
    let cx = cv_round_f32(pt.x);
    let cy = cv_round_f32(pt.y);

    // For CV_8UC1, step1(0) is the row stride in bytes.
    let step = img.step1(0)? as isize;
    let data = img.data_typed::<u8>()?;
    let center_idx = cy as isize * step + cx as isize;

    // Sample one pattern offset (x, y) rotated by (a, b) = (cos θ, sin θ).
    let sample = |p: (i8, i8)| -> u8 {
        let px = p.0 as f32;
        let py = p.1 as f32;
        let x_off = cv_round_f32(px * a - py * b);
        let y_off = cv_round_f32(px * b + py * a);
        data[(center_idx + y_off as isize * step + x_off as isize) as usize]
    };

    for (i, byte) in desc.iter_mut().enumerate() {
        let pairs = &BRIEF_PATTERN[i * BITS_PER_BYTE..(i + 1) * BITS_PER_BYTE];
        let mut val = 0u8;
        for (bit, pair) in pairs.iter().enumerate() {
            let t0 = sample(pair.p1);
            let t1 = sample(pair.p2);
            val |= ((t0 < t1) as u8) << bit;
        }
        *byte = val;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{KeyPoint, Mat, MatTraitConstManual, Scalar};

    // Default parameters used across many tests
    const N_FEATURES: usize = 1000;
    const SCALE_FACTOR: f32 = 1.2;
    const N_LEVELS: usize = 8;
    const INI_TH_FAST: i32 = 20;
    const MIN_TH_FAST: i32 = 7;

    /// Synthetic 8-bit grayscale checkerboard with many corner-like features.
    fn make_checkerboard(rows: i32, cols: i32, square: i32) -> Mat {
        let mut img =
            Mat::new_rows_cols_with_default(rows, cols, CV_8UC1, Scalar::all(0.0)).unwrap();
        let stride = img.step1(0).unwrap() as usize;
        let data = img.data_bytes_mut().unwrap();
        for y in 0..rows {
            for x in 0..cols {
                let a = (x / square) & 1;
                let b = (y / square) & 1;
                let v: u8 = if (a ^ b) != 0 { 230 } else { 25 };
                data[(y as usize) * stride + (x as usize)] = v;
            }
        }
        img
    }

    fn make_uniform(rows: i32, cols: i32, v: u8) -> Mat {
        Mat::new_rows_cols_with_default(rows, cols, CV_8UC1, Scalar::all(v as f64)).unwrap()
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= f32::max(a.abs(), b.abs()) * 1e-5 + 1e-6
    }

    // -----------------------------------------------------------------------
    // Constructor / accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn constructor_stores_basic_parameters() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);

        assert_eq!(ext.get_levels(), N_LEVELS);
        assert!(approx_eq(ext.get_scale_factor() as f32, SCALE_FACTOR));
        assert_eq!(ext.get_scale_factors().len(), N_LEVELS);
        assert_eq!(ext.get_inverse_scale_factors().len(), N_LEVELS);
        assert_eq!(ext.get_scale_sigma2().len(), N_LEVELS);
        assert_eq!(ext.get_inverse_scale_sigma2().len(), N_LEVELS);
    }

    #[test]
    fn scale_factor_sequence_is_geometric() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);

        let sf = ext.get_scale_factors();
        let isf = ext.get_inverse_scale_factors();
        let s2 = ext.get_scale_sigma2();
        let is2 = ext.get_inverse_scale_sigma2();

        assert!(approx_eq(sf[0], 1.0));
        assert!(approx_eq(isf[0], 1.0));
        assert!(approx_eq(s2[0], 1.0));
        assert!(approx_eq(is2[0], 1.0));

        for i in 1..N_LEVELS {
            assert!(approx_eq(sf[i], sf[i - 1] * SCALE_FACTOR));
            assert!(approx_eq(isf[i], 1.0 / sf[i]));
            assert!(approx_eq(s2[i], sf[i] * sf[i]));
            assert!(approx_eq(is2[i], 1.0 / s2[i]));
            assert!(sf[i] > sf[i - 1]); // strictly increasing
            assert!(isf[i] < isf[i - 1]); // strictly decreasing
        }
    }

    #[test]
    fn scale_factor_sequence_with_unit_scale_is_constant() {
        let ext = OrbExtractor::new(500, 1.0, 4, INI_TH_FAST, MIN_TH_FAST);
        for s in ext.get_scale_factors() {
            assert!(approx_eq(s, 1.0));
        }
        for s in ext.get_inverse_scale_factors() {
            assert!(approx_eq(s, 1.0));
        }
    }

    #[test]
    fn constructor_with_single_level_works() {
        let ext = OrbExtractor::new(500, 1.2, 1, INI_TH_FAST, MIN_TH_FAST);
        assert_eq!(ext.get_levels(), 1);
        assert_eq!(ext.get_scale_factors().len(), 1);
        assert!(approx_eq(ext.get_scale_factors()[0], 1.0));
    }

    // -----------------------------------------------------------------------
    // compute() — edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn compute_returns_err_for_empty_image() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);
        let empty = Mat::default();
        let result = ext.compute(&empty, &Mat::default(), [0, 0]);
        assert!(matches!(result, Err(ExtractionError::EmptyImage)));
    }

    #[test]
    fn compute_uniform_image_yields_no_keypoints() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);
        let img = make_uniform(240, 320, 128);
        let result = ext.compute(&img, &Mat::default(), [0, 0]).unwrap();
        assert_eq!(result.mono_index, 0);
        assert!(result.keypoints.is_empty());
        assert!(result.descriptors.is_none()); // C++: _descriptors.release()
    }

    // -----------------------------------------------------------------------
    // compute() — happy path on a checkerboard
    // -----------------------------------------------------------------------

    #[test]
    fn compute_extracts_keypoints_and_descriptors_on_textured_image() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);
        let img = make_checkerboard(480, 640, 20);

        let result = ext.compute(&img, &Mat::default(), [0, 0]).unwrap();
        assert!(result.mono_index > 0);
        assert_eq!(result.mono_index as usize, result.keypoints.len());

        let desc = result.descriptors.expect("descriptors present");
        assert_eq!(desc.rows(), result.keypoints.len() as i32);
        assert_eq!(desc.cols(), 32);
        assert_eq!(desc.typ(), CV_8U);

        for kp in &result.keypoints {
            let pt = kp.pt();
            assert!(pt.x >= 0.0);
            assert!(pt.y >= 0.0);
            assert!(pt.x < img.cols() as f32);
            assert!(pt.y < img.rows() as f32);
            assert!(kp.octave() >= 0);
            assert!(kp.octave() < N_LEVELS as i32);
            assert!(kp.size() > 0.0);
            assert!(kp.angle() >= 0.0);
            assert!(kp.angle() < 360.0);
        }
    }

    #[test]
    fn compute_is_deterministic_for_same_input() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);
        let img = make_checkerboard(480, 640, 20);

        let result1 = ext.compute(&img, &Mat::default(), [0, 0]).unwrap();
        let result2 = ext.compute(&img, &Mat::default(), [0, 0]).unwrap();

        assert_eq!(result1.mono_index, result2.mono_index);
        assert_eq!(result1.keypoints.len(), result2.keypoints.len());
        let desc1 = result1.descriptors.expect("descriptors 1");
        let desc2 = result2.descriptors.expect("descriptors 2");
        assert_eq!(desc1.rows(), desc2.rows());

        for (a, b) in result1.keypoints.iter().zip(result2.keypoints.iter()) {
            assert!(approx_eq(a.pt().x, b.pt().x));
            assert!(approx_eq(a.pt().y, b.pt().y));
            assert_eq!(a.octave(), b.octave());
        }

        // Descriptors bit-identical
        let d1 = desc1.data_bytes().unwrap();
        let d2 = desc2.data_bytes().unwrap();
        assert_eq!(d1, d2);
    }

    #[test]
    fn lower_fast_thresholds_yield_more_keypoints() {
        let img = make_checkerboard(480, 640, 20);

        let ext_high = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, 40, 20);
        let ext_low = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, 7, 3);

        let result_high = ext_high.compute(&img, &Mat::default(), [0, 0]).unwrap();
        let result_low = ext_low.compute(&img, &Mat::default(), [0, 0]).unwrap();

        assert!(result_low.keypoints.len() >= result_high.keypoints.len());
    }

    // -----------------------------------------------------------------------
    // Image pyramid
    // -----------------------------------------------------------------------

    #[test]
    fn pyramid_levels_have_correct_sizes() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);
        let img = make_checkerboard(480, 640, 20);
        let result = ext.compute(&img, &Mat::default(), [0, 0]).unwrap();

        assert_eq!(result.image_pyramid.len(), N_LEVELS);
        let isf = ext.get_inverse_scale_factors().clone();

        // Level 0 matches input image dimensions
        assert_eq!(result.image_pyramid[0].rows(), img.rows());
        assert_eq!(result.image_pyramid[0].cols(), img.cols());

        for level in 0..N_LEVELS {
            let m = &result.image_pyramid[level];
            assert!(!m.empty());
            assert_eq!(m.typ(), CV_8UC1);

            let exp_rows = cv_round_f32(img.rows() as f32 * isf[level]);
            let exp_cols = cv_round_f32(img.cols() as f32 * isf[level]);
            assert_eq!(m.rows(), exp_rows);
            assert_eq!(m.cols(), exp_cols);

            if level > 0 {
                assert!(m.rows() <= result.image_pyramid[level - 1].rows());
                assert!(m.cols() <= result.image_pyramid[level - 1].cols());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Lapping area: stereo / mono split
    // -----------------------------------------------------------------------

    #[test]
    fn lapping_area_splits_keypoints_into_mono_and_stereo() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);
        let img = make_checkerboard(480, 640, 20);

        // Right half is the "lapping" (stereo) area.
        let lap = [img.cols() / 2, img.cols()];
        let result = ext.compute(&img, &Mat::default(), lap).unwrap();

        assert!(result.mono_index >= 0);
        assert!(result.mono_index <= result.keypoints.len() as i32);
        let total = result.keypoints.len() as i32;
        let stereo = total - result.mono_index;

        // Both categories should be populated for a textured image.
        assert!(result.mono_index > 0);
        assert!(stereo > 0);

        for kp in &result.keypoints[..result.mono_index as usize] {
            let x = kp.pt().x;
            let inside = x >= lap[0] as f32 && x <= lap[1] as f32;
            assert!(!inside, "mono keypoint inside lapping area: x={x}");
        }
        for kp in &result.keypoints[result.mono_index as usize..] {
            let x = kp.pt().x;
            assert!(x >= lap[0] as f32);
            assert!(x <= lap[1] as f32);
        }
    }

    #[test]
    fn lapping_area_covering_full_image_marks_every_keypoint_as_stereo() {
        let ext = OrbExtractor::new(N_FEATURES, SCALE_FACTOR, N_LEVELS, INI_TH_FAST, MIN_TH_FAST);
        let img = make_checkerboard(240, 320, 20);

        let lap = [0, img.cols()];
        let result = ext.compute(&img, &Mat::default(), lap).unwrap();
        assert!(!result.keypoints.is_empty());
        assert_eq!(result.mono_index, 0);
    }

    // -----------------------------------------------------------------------
    // ExtractorNode
    // -----------------------------------------------------------------------

    fn kp(x: f32, y: f32) -> KeyPoint {
        KeyPoint::new_coords_def(x, y, 1.0).unwrap()
    }

    #[test]
    fn divide_node_splits_region_into_four_quadrants() {
        let mut node = ExtractorNode {
            ul: Point2i::new(0, 0),
            ur: Point2i::new(100, 0),
            bl: Point2i::new(0, 80),
            br: Point2i::new(100, 80),
            ..Default::default()
        };
        // Sprinkle one keypoint into each quadrant.
        node.keys.push(kp(10.0, 10.0)); // top-left
        node.keys.push(kp(80.0, 10.0)); // top-right
        node.keys.push(kp(10.0, 70.0)); // bottom-left
        node.keys.push(kp(80.0, 70.0)); // bottom-right

        let [n1, n2, n3, n4] = node.divide_node();

        // Quadrants partition the bounding box.
        assert_eq!(n1.ul, node.ul);
        assert_eq!(n4.br, node.br);
        assert_eq!(n1.ur.x, n2.ul.x);
        assert_eq!(n1.bl.y, n3.ul.y);
        assert_eq!(n2.br.x, node.ur.x);
        assert_eq!(n3.br.y, node.br.y);

        // All four keypoints are redistributed across the children.
        let total = n1.keys.len() + n2.keys.len() + n3.keys.len() + n4.keys.len();
        assert_eq!(total, 4);

        // Each child with a single keypoint should set no_more == true.
        for c in [&n1, &n2, &n3, &n4] {
            if c.keys.len() == 1 {
                assert!(c.no_more);
            } else {
                assert!(!c.no_more);
            }
        }
    }

    #[test]
    fn divide_node_with_no_keypoints_leaves_children_empty() {
        let node = ExtractorNode {
            ul: Point2i::new(0, 0),
            ur: Point2i::new(50, 0),
            bl: Point2i::new(0, 40),
            br: Point2i::new(50, 40),
            ..Default::default()
        };
        let [n1, n2, n3, n4] = node.divide_node();
        for c in [&n1, &n2, &n3, &n4] {
            assert!(c.keys.is_empty());
            assert!(!c.no_more);
        }
    }
}
