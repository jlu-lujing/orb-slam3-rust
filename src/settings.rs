use crate::{
    camera_models::{GeometricCamera, kannala_brandt8::KannalaBrandt8, pinhole::Pinhole},
    system::Sensor,
};
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion};
use opencv::{
    calib3d::{CALIB_ZERO_DISPARITY, init_undistort_rectify_map, stereo_rectify},
    core::{CV_32F, CV_64F, FileStorage, FileStorage_READ, Mat, Rect, Size},
    prelude::*,
};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraType {
    PinHole = 0,
    Rectified = 1,
    KannalaBrandt = 2,
}

#[derive(Debug)]
pub struct CameraInfo {
    pub calibration: Arc<dyn GeometricCamera>,
    pub original_calibration: Arc<dyn GeometricCamera>,
    pub pinhole_distortion: Option<Vec<f32>>,
}

impl CameraInfo {
    /// Mutable access to the calibration during settings construction.
    ///
    /// The camera is shared as `Arc<dyn GeometricCamera>` everywhere downstream,
    /// but while `Settings` is still being built the `Arc` is uniquely owned, so
    /// `Arc::get_mut` is guaranteed to succeed. Calibration is logically
    /// immutable once parsing finishes.
    fn calibration_mut(&mut self) -> &mut dyn GeometricCamera {
        Arc::get_mut(&mut self.calibration)
            .expect("camera Arc is unique during settings construction")
    }
}

#[derive(Debug)]
pub struct ImageInfo {
    pub original_size: Size,
    pub new_size: Size,
    pub need_to_resize: bool,
    pub fps: i32,
    pub rgb: bool,
}

#[derive(Debug)]
pub struct IMUInfo {
    pub noise_gyro: f32,
    pub noise_acc: f32,
    pub gyro_walk: f32,
    pub acc_walk: f32,
    pub imu_frequency: f32,
    pub tbc: Isometry3<f32>,
    pub insert_kfs_when_lost: bool,
}

#[derive(Debug)]
pub struct RGBDInfo {
    pub depth_map_factor: f32,
    pub th_depth: f32,
    pub b: f32,
    pub bf: f32,
}

pub struct StereoInfo {
    pub tlr: Isometry3<f32>,
    pub b: f32,
    pub bf: f32,
    // Raw depth the close/far threshold is `b * th_depth`, computed where it is consumed.
    pub th_depth: f32,
    pub m1l: Mat,
    pub m2l: Mat,
    pub m1r: Mat,
    pub m2r: Mat,
}

// Mat does not implement Debug, so we hand-roll it.
impl std::fmt::Debug for StereoInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StereoInfo")
            .field("tlr", &self.tlr)
            .field("b", &self.b)
            .field("bf", &self.bf)
            .field("th_depth", &self.th_depth)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct ORBInfo {
    pub n_features: i32,
    pub scale_factor: f32,
    pub n_levels: i32,
    pub init_th_fast: i32,
    pub min_th_fast: i32,
}

#[derive(Debug)]
pub struct ViewerInfo {
    pub keyframe_size: f32,
    pub keyframe_linewidth: f32,
    pub graph_linewidth: f32,
    pub point_size: f32,
    pub camera_size: f32,
    pub camera_linewidth: f32,
    pub view_point_x: f32,
    pub view_point_y: f32,
    pub view_point_z: f32,
    pub view_point_f: f32,
    pub image_viewer_scale: f32,
}

#[derive(Debug)]
pub struct LoadAndSaveInfo {
    pub load_from: Option<String>,
    pub save_to: Option<String>,
}

#[derive(Debug)]
pub struct OtherInfo {
    pub th_far_points: Option<f32>,
}

#[derive(Debug)]
pub struct Settings {
    pub sensor: Sensor,
    pub camera_model: CameraType,
    pub camera1: Option<CameraInfo>,
    pub camera2: Option<CameraInfo>,
    pub need_to_rectify: bool,
    pub need_to_undistort: bool,
    pub image: ImageInfo,
    pub imu: Option<IMUInfo>,
    pub rgbd: Option<RGBDInfo>,
    pub stereo: Option<StereoInfo>,
    pub orb: ORBInfo,
    pub viewer: ViewerInfo,
    pub load_and_save: LoadAndSaveInfo,
    pub loop_closing: bool,
    pub other: OtherInfo,
}

#[derive(Debug)]
pub enum SettingsError {
    InvalidFile,
    InvalidVersion,
    InvalidDatatype(String),
    MissingCameraModel,
    InvalidCameraModel(String),
    Rectification,
}

impl Settings {
    pub fn new(config_file: &PathBuf, sensor: Sensor) -> Result<Self, SettingsError> {
        // Open settings file
        let settings = FileStorage::new_def(&config_file.to_string_lossy(), FileStorage_READ)
            .map_err(|_| SettingsError::InvalidFile)?;
        if let Err(_) = settings.is_opened() {
            return Err(SettingsError::InvalidFile);
        }
        info!("Load settings from {}", config_file.to_string_lossy());
        let node = settings
            .get("File.version")
            .map_err(|_| SettingsError::InvalidVersion)?;
        if !node
            .is_string()
            .map_err(|_| SettingsError::InvalidVersion)?
            || node.string().map_err(|_| SettingsError::InvalidVersion)? != "1.0"
        {
            return Err(SettingsError::InvalidVersion);
        }
        info!("Settings file is valid");

        // Get camera model
        let mut need_to_undistort = false;
        let model =
            read_param_string(&settings, "Camera.type").ok_or(SettingsError::MissingCameraModel)?;
        let camera_model = match model.as_str() {
            "PinHole" => CameraType::PinHole,
            "Rectified" => CameraType::Rectified,
            "KannalaBrandt8" => CameraType::KannalaBrandt,
            _ => return Err(SettingsError::InvalidCameraModel(model)),
        };

        // Get first camera
        let camera1 = Self::read_camera(0, &settings, &camera_model, &sensor);
        // Check if we need to correct distortion from the images
        if camera_model == CameraType::PinHole
            && matches!(sensor, Sensor::Monocular | Sensor::IMUMonocular)
            && camera1.pinhole_distortion.is_some()
        {
            need_to_undistort = true;
        }
        let mut camera1 = Some(camera1);

        // Get second camera if stereo (not rectified)
        let mut camera2 = None;
        let mut need_to_rectify = false;
        if matches!(sensor, Sensor::Stereo | Sensor::IMUStereo) {
            camera2 = Some(Self::read_camera(1, &settings, &camera_model, &sensor));
            if camera_model == CameraType::PinHole {
                need_to_rectify = true;
            }
        }

        // Image info
        let image = Self::read_image_info(
            &settings,
            need_to_rectify,
            &mut camera1,
            &mut camera2,
            &camera_model,
        );

        // IMU info
        let mut imu = if matches!(
            sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        ) {
            Some(Self::read_imu_info(&settings))
        } else {
            None
        };

        // RGBD info
        let rgbd = if matches!(sensor, Sensor::RGBD | Sensor::IMURGBD) {
            Some(Self::read_rgbd_info(
                &settings,
                camera1
                    .as_ref()
                    .expect("missing camera")
                    .calibration
                    .get_parameter(0),
            ))
        } else {
            None
        };

        // Other info
        let orb = Self::read_orb(&settings);
        let viewer = Self::read_viewer(&settings);
        let load_and_save = Self::read_load_and_save(&settings);
        let other = Self::read_other(&settings);
        let loop_closing = read_param_bool(&settings, "loopClosing").unwrap_or(false);

        // Precompute rectification maps
        let mut stereo = None;
        // For pre-rectified stereo, baseline / ThDepth are still read from YAML.
        if matches!(sensor, Sensor::Stereo | Sensor::IMUStereo)
            && camera_model == CameraType::Rectified
        {
            let b = read_param_float(&settings, "Stereo.b").unwrap_or(0.0);
            let fx = camera1
                .as_ref()
                .expect("camera1 missing")
                .calibration
                .get_parameter(0);
            stereo = Some(StereoInfo {
                tlr: Isometry3::identity(),
                b,
                bf: b * fx,
                th_depth: read_param_float(&settings, "Stereo.ThDepth").unwrap_or(0.0),
                m1l: Mat::default(),
                m2l: Mat::default(),
                m1r: Mat::default(),
                m2r: Mat::default(),
            });
        }
        if need_to_rectify {
            let k1 = camera1.as_ref().unwrap().calibration.to_k();
            let mut k1_converted = Mat::default();
            k1.convert_to(&mut k1_converted, CV_64F, 1., 0.)
                .map_err(|_| SettingsError::Rectification)?;
            let k2 = camera2.as_ref().unwrap().calibration.to_k();
            let mut k2_converted = Mat::default();
            k2.convert_to(&mut k2_converted, CV_64F, 1., 0.)
                .map_err(|_| SettingsError::Rectification)?;

            let tlr_mat =
                read_param_mat(&settings, "Stereo.T_c1_c2").ok_or(SettingsError::Rectification)?;
            let tlr = mat4x4_to_isometry3(&tlr_mat);

            // Invert Tlr_ to get the transform that maps points from cam1 to cam2
            // (R12, t12 as expected by stereoRectify).
            let tlr_inv = tlr.inverse();
            let r_mat = tlr_inv.rotation.to_rotation_matrix();
            let r = r_mat.matrix();
            let t = tlr_inv.translation.vector;
            let r12 = Mat::from_slice_2d(&[
                &[r[(0, 0)] as f64, r[(0, 1)] as f64, r[(0, 2)] as f64],
                &[r[(1, 0)] as f64, r[(1, 1)] as f64, r[(1, 2)] as f64],
                &[r[(2, 0)] as f64, r[(2, 1)] as f64, r[(2, 2)] as f64],
            ])
            .map_err(|_| SettingsError::Rectification)?;
            let t12 = Mat::from_slice_2d(&[&[t[0] as f64], &[t[1] as f64], &[t[2] as f64]])
                .map_err(|_| SettingsError::Rectification)?;

            let dist1_mat = match &camera1.as_ref().unwrap().pinhole_distortion {
                Some(d) => Mat::from_slice(d.as_slice())
                    .map_err(|_| SettingsError::Rectification)?
                    .try_clone()
                    .map_err(|_| SettingsError::Rectification)?,
                None => Mat::default(),
            };
            let dist2_mat = match &camera2.as_ref().unwrap().pinhole_distortion {
                Some(d) => Mat::from_slice(d.as_slice())
                    .map_err(|_| SettingsError::Rectification)?
                    .try_clone()
                    .map_err(|_| SettingsError::Rectification)?,
                None => Mat::default(),
            };

            let new_im_size = image.new_size;
            let mut r_r1_u1 = Mat::default();
            let mut r_r2_u2 = Mat::default();
            let mut p1 = Mat::default();
            let mut p2 = Mat::default();
            let mut q = Mat::default();
            let mut valid_roi1 = Rect::default();
            let mut valid_roi2 = Rect::default();
            stereo_rectify(
                &k1_converted,
                &dist1_mat,
                &k2_converted,
                &dist2_mat,
                new_im_size,
                &r12,
                &t12,
                &mut r_r1_u1,
                &mut r_r2_u2,
                &mut p1,
                &mut p2,
                &mut q,
                CALIB_ZERO_DISPARITY.into(),
                -1.,
                new_im_size,
                &mut valid_roi1,
                &mut valid_roi2,
            )
            .map_err(|_| SettingsError::Rectification)?;

            // P1 and P2 are 3×4; extract the 3×3 upper-left block as new camera matrices.
            let p1_3x3 =
                Mat::roi(&p1, Rect::new(0, 0, 3, 3)).map_err(|_| SettingsError::Rectification)?;
            let p2_3x3 =
                Mat::roi(&p2, Rect::new(0, 0, 3, 3)).map_err(|_| SettingsError::Rectification)?;

            let mut m1l = Mat::default();
            let mut m2l = Mat::default();
            init_undistort_rectify_map(
                &k1_converted,
                &dist1_mat,
                &r_r1_u1,
                &p1_3x3,
                new_im_size,
                CV_32F,
                &mut m1l,
                &mut m2l,
            )
            .map_err(|_| SettingsError::Rectification)?;

            let mut m1r = Mat::default();
            let mut m2r = Mat::default();
            init_undistort_rectify_map(
                &k2_converted,
                &dist2_mat,
                &r_r2_u2,
                &p2_3x3,
                new_im_size,
                CV_32F,
                &mut m1r,
                &mut m2r,
            )
            .map_err(|_| SettingsError::Rectification)?;

            // Update calibration
            let cam1 = camera1.as_mut().unwrap();
            let fx = *p1
                .at_2d::<f64>(0, 0)
                .map_err(|_| SettingsError::Rectification)? as f32;
            let fy = *p1
                .at_2d::<f64>(1, 1)
                .map_err(|_| SettingsError::Rectification)? as f32;
            let cx = *p1
                .at_2d::<f64>(0, 2)
                .map_err(|_| SettingsError::Rectification)? as f32;
            let cy = *p1
                .at_2d::<f64>(1, 2)
                .map_err(|_| SettingsError::Rectification)? as f32;
            cam1.calibration_mut().set_parameter(fx, 0);
            cam1.calibration_mut().set_parameter(fy, 1);
            cam1.calibration_mut().set_parameter(cx, 2);
            cam1.calibration_mut().set_parameter(cy, 3);

            // Update bf
            let b = read_param_float(&settings, "Stereo.b").unwrap_or(0.0);
            let bf = b
                * (*p1
                    .at_2d::<f64>(0, 0)
                    .map_err(|_| SettingsError::Rectification)? as f32);

            // Rectification rotates the left camera frame; propagate that into T_bc.
            if matches!(sensor, Sensor::IMUStereo) {
                if let Some(ref mut imu_info) = imu {
                    let get = |row: i32, col: i32| -> Result<f32, SettingsError> {
                        r_r1_u1
                            .at_2d::<f64>(row, col)
                            .map(|v| *v as f32)
                            .map_err(|_| SettingsError::Rectification)
                    };
                    let r_val = Matrix3::new(
                        get(0, 0)?,
                        get(0, 1)?,
                        get(0, 2)?,
                        get(1, 0)?,
                        get(1, 1)?,
                        get(1, 2)?,
                        get(2, 0)?,
                        get(2, 1)?,
                        get(2, 2)?,
                    );
                    // from_matrix_unchecked is safe: R_r1_u1 is an orthonormal rotation
                    // output from stereoRectify.
                    let rotation = UnitQuaternion::from_rotation_matrix(
                        &Rotation3::from_matrix_unchecked(r_val),
                    );
                    let t_r1_u1 = Isometry3::from_parts(Translation3::identity(), rotation);
                    imu_info.tbc = imu_info.tbc * t_r1_u1.inverse();
                }
            }

            stereo = Some(StereoInfo {
                tlr,
                b,
                bf,
                th_depth: read_param_float(&settings, "Stereo.ThDepth").unwrap_or(0.0),
                m1l,
                m2l,
                m1r,
                m2r,
            });
        }

        info!("Settings succesfully loaded");
        Ok(Settings {
            sensor,
            camera_model,
            camera1,
            camera2,
            need_to_rectify,
            need_to_undistort,
            image,
            imu,
            rgbd,
            stereo,
            orb,
            viewer,
            load_and_save,
            loop_closing,
            other,
        })
    }

    /// Distortion coefficients of the first camera as an N×1 `CV_32F` `Mat`.
    /// Returns an empty `Mat` when the camera is undistorted (e.g. fisheye or rectified).
    pub fn camera1_distortion_coef(&self) -> Mat {
        Self::distortion_coef(self.camera1.as_ref())
    }
    /// Distortion coefficients of the second camera as an N×1 `CV_32F` `Mat`.
    pub fn camera2_distortion_coef(&self) -> Mat {
        Self::distortion_coef(self.camera2.as_ref())
    }
    fn distortion_coef(camera: Option<&CameraInfo>) -> Mat {
        match camera.and_then(|c| c.pinhole_distortion.as_ref()) {
            Some(d) if !d.is_empty() => Mat::from_slice(d.as_slice())
                .expect("from_slice on owned f32 distortion vector cannot fail")
                .reshape(1, d.len() as i32)
                .expect("reshape to N×1 of matching length cannot fail")
                .try_clone()
                .expect("clone of small distortion Mat cannot fail"),
            _ => Mat::default(),
        }
    }

    fn read_camera(
        index: u8,
        storage: &FileStorage,
        model: &CameraType,
        sensor: &Sensor,
    ) -> CameraInfo {
        let index_one = index + 1;

        //Read intrinsic parameters
        let fx = read_param_float(storage, &format!("Camera{}.fx", index_one)).unwrap_or(0.0);
        let fy = read_param_float(storage, &format!("Camera{}.fy", index_one)).unwrap_or(0.0);
        let cx = read_param_float(storage, &format!("Camera{}.cx", index_one)).unwrap_or(0.0);
        let cy = read_param_float(storage, &format!("Camera{}.cy", index_one)).unwrap_or(0.0);

        match model {
            CameraType::PinHole => {
                let calibration = vec![fx, fy, cx, cy];
                let cam = Pinhole::with_params(calibration);
                let original = cam.clone();

                // Check if is a distorted PinHole
                let mut distortion = None;
                let k1 = read_param_float(storage, &format!("Camera{}.k1", index_one));
                if let Some(k1) = k1 {
                    let mut dist = vec![0.0f32; 4];
                    let k3 = read_param_float(storage, &format!("Camera{}.k3", index_one));
                    if let Some(k3) = k3 {
                        dist.push(k3);
                    }
                    dist[0] = k1;
                    dist[1] =
                        read_param_float(storage, &format!("Camera{}.k2", index_one)).unwrap_or(0.);
                    dist[2] =
                        read_param_float(storage, &format!("Camera{}.p1", index_one)).unwrap_or(0.);
                    dist[3] =
                        read_param_float(storage, &format!("Camera{}.p2", index_one)).unwrap_or(0.);
                    distortion = Some(dist);
                }

                CameraInfo {
                    calibration: Arc::new(cam),
                    original_calibration: Arc::new(original),
                    pinhole_distortion: distortion,
                }
            }
            CameraType::Rectified => {
                let calibration = vec![fx, fy, cx, cy];
                let cam = Pinhole::with_params(calibration);
                let original = cam.clone();

                // Rectified images are assumed to be ideal PinHole images (no distortion)
                CameraInfo {
                    calibration: Arc::new(cam),
                    original_calibration: Arc::new(original),
                    pinhole_distortion: None,
                }
            }
            CameraType::KannalaBrandt => {
                let k0 =
                    read_param_float(storage, &format!("Camera{}.k1", index_one)).unwrap_or(0.0);
                let k1 =
                    read_param_float(storage, &format!("Camera{}.k2", index_one)).unwrap_or(0.0);
                let k2 =
                    read_param_float(storage, &format!("Camera{}.k3", index_one)).unwrap_or(0.0);
                let k3 =
                    read_param_float(storage, &format!("Camera{}.k4", index_one)).unwrap_or(0.0);

                let calibration = vec![fx, fy, cx, cy, k0, k1, k2, k3];
                let mut cam = KannalaBrandt8::with_params(calibration);
                let original = cam.clone();

                // Overlapping
                if matches!(*sensor, Sensor::Stereo | Sensor::IMUStereo) {
                    let col_begin: usize =
                        read_param_int(storage, &format!("Camera{}.overlappingBegin", index_one))
                            .unwrap_or(0)
                            .try_into()
                            .unwrap_or(0);
                    let col_end: usize =
                        read_param_int(storage, &format!("Camera{}.overlappingEnd", index_one))
                            .unwrap_or(0)
                            .try_into()
                            .unwrap_or(0);
                    let overlapping = vec![col_begin, col_end];
                    cam.lapping_area = overlapping;
                }

                CameraInfo {
                    calibration: Arc::new(cam),
                    original_calibration: Arc::new(original),
                    pinhole_distortion: None,
                }
            }
        }
    }

    fn read_image_info(
        storage: &FileStorage,
        need_to_rectify: bool,
        camera1: &mut Option<CameraInfo>,
        camera2: &mut Option<CameraInfo>,
        camera_type: &CameraType,
    ) -> ImageInfo {
        // Read original and desired image dimensions
        let original_rows = read_param_int(storage, "Camera.height").unwrap_or(0);
        let original_cols = read_param_int(storage, "Camera.width").unwrap_or(0);
        let original_size = Size::new(
            original_cols.try_into().expect("size overflow"),
            original_rows.try_into().expect("size overflow"),
        );
        let mut new_size = original_size.clone();
        let mut need_to_resize = false;

        let new_height = read_param_int(storage, "Camera.newHeight");
        if let Some(new_height) = new_height {
            need_to_resize = true;
            new_size.height = new_height.try_into().expect("size oveflow");
            if !need_to_rectify {
                // Update calibration
                let scale_factor = new_size.height as f32 / original_size.height as f32;
                if let Some(camera) = camera1 {
                    let p1 = camera.calibration.get_parameter(1) * scale_factor;
                    let p3 = camera.calibration.get_parameter(3) * scale_factor;
                    camera.calibration_mut().set_parameter(p1, 1);
                    camera.calibration_mut().set_parameter(p3, 3);
                }
                if let Some(camera) = camera2 {
                    let p1 = camera.calibration.get_parameter(1) * scale_factor;
                    let p3 = camera.calibration.get_parameter(3) * scale_factor;
                    camera.calibration_mut().set_parameter(p1, 1);
                    camera.calibration_mut().set_parameter(p3, 3);
                }
            }
        }

        let new_width = read_param_int(storage, "Camera.newWidth");
        if let Some(new_width) = new_width {
            need_to_resize = true;
            new_size.width = new_width.try_into().expect("size overflow");
            if !need_to_rectify {
                // update calibration
                let scale_factor = new_size.width as f32 / original_size.width as f32;
                if let Some(camera) = camera1 {
                    let p0 = camera.calibration.get_parameter(0) * scale_factor;
                    let p2 = camera.calibration.get_parameter(2) * scale_factor;
                    camera.calibration_mut().set_parameter(p0, 0);
                    camera.calibration_mut().set_parameter(p2, 2);
                    if *camera_type == CameraType::KannalaBrandt {
                        if let Some(kanalla) = camera
                            .calibration_mut()
                            .as_any_mut()
                            .downcast_mut::<KannalaBrandt8>()
                        {
                            kanalla.lapping_area[0] =
                                (kanalla.lapping_area[0] as f32 * scale_factor) as usize;
                            kanalla.lapping_area[1] =
                                (kanalla.lapping_area[1] as f32 * scale_factor) as usize;
                        }
                    }
                }
                if let Some(camera) = camera2 {
                    let p0 = camera.calibration.get_parameter(0) * scale_factor;
                    let p2 = camera.calibration.get_parameter(2) * scale_factor;
                    camera.calibration_mut().set_parameter(p0, 0);
                    camera.calibration_mut().set_parameter(p2, 2);
                    if *camera_type == CameraType::KannalaBrandt {
                        if let Some(kanalla) = camera
                            .calibration_mut()
                            .as_any_mut()
                            .downcast_mut::<KannalaBrandt8>()
                        {
                            kanalla.lapping_area[0] =
                                (kanalla.lapping_area[0] as f32 * scale_factor) as usize;
                            kanalla.lapping_area[1] =
                                (kanalla.lapping_area[1] as f32 * scale_factor) as usize;
                        }
                    }
                }
            }
        }

        let fps = read_param_int(storage, "Camera.fps")
            .unwrap_or(30)
            .try_into()
            .unwrap_or(30);

        let rgb = read_param_bool(storage, "Camera.RGB").unwrap_or(false);

        ImageInfo {
            original_size,
            new_size,
            need_to_resize,
            fps,
            rgb,
        }
    }

    fn read_imu_info(storage: &FileStorage) -> IMUInfo {
        let cv_tbc = read_param_mat(storage, "IMU.T_b_c1");
        let mut tbc = Isometry3::identity();
        if let Some(cv_tbc) = cv_tbc {
            tbc = mat4x4_to_isometry3(&cv_tbc);
        }
        IMUInfo {
            noise_gyro: read_param_float(storage, "IMU.NoiseGyro").unwrap_or(1.7e-04),
            noise_acc: read_param_float(storage, "IMU.NoiseAcc").unwrap_or(2.0e-03),
            gyro_walk: read_param_float(storage, "IMU.GyroWalk").unwrap_or(1.9393e-05),
            acc_walk: read_param_float(storage, "IMU.AccWalk").unwrap_or(3.0e-03),
            imu_frequency: read_param_float(storage, "IMU.Frequency").unwrap_or(200.0),
            tbc: tbc,
            insert_kfs_when_lost: read_param_bool(storage, "IMU.InsertKFsWhenLost").unwrap_or(true),
        }
    }

    fn read_rgbd_info(storage: &FileStorage, calib1_param: f32) -> RGBDInfo {
        let b = read_param_float(storage, "Stereo.b").unwrap_or(0.);
        RGBDInfo {
            depth_map_factor: read_param_float(storage, "RGBD.DepthMapFactor").unwrap_or(0.),
            th_depth: read_param_float(storage, "Stereo.ThDepth").unwrap_or(0.),
            b,
            bf: b * calib1_param,
        }
    }

    fn read_orb(storage: &FileStorage) -> ORBInfo {
        ORBInfo {
            n_features: read_param_int(storage, "ORBextractor.nFeatures")
                .unwrap_or(1200)
                .try_into()
                .unwrap_or(1200),
            scale_factor: read_param_float(storage, "ORBextractor.scaleFactor").unwrap_or(0.),
            n_levels: read_param_int(storage, "ORBextractor.nLevels")
                .unwrap_or(8)
                .try_into()
                .unwrap_or(8),
            init_th_fast: read_param_int(storage, "ORBextractor.iniThFAST")
                .unwrap_or(20)
                .try_into()
                .unwrap_or(20),
            min_th_fast: read_param_int(storage, "ORBextractor.minThFAST")
                .unwrap_or(7)
                .try_into()
                .unwrap_or(7),
        }
    }

    fn read_viewer(storage: &FileStorage) -> ViewerInfo {
        ViewerInfo {
            keyframe_size: read_param_float(storage, "Viewer.KeyFrameSize").unwrap_or(0.05),
            keyframe_linewidth: read_param_float(storage, "Viewer.KeyFrameLineWidth")
                .unwrap_or(1.05),
            graph_linewidth: read_param_float(storage, "Viewer.GraphLineWidth").unwrap_or(0.9),
            point_size: read_param_float(storage, "Viewer.PointSize").unwrap_or(2.0),
            camera_size: read_param_float(storage, "Viewer.CameraSize").unwrap_or(0.08),
            camera_linewidth: read_param_float(storage, "Viewer.CameraLineWidth").unwrap_or(3.0),
            view_point_x: read_param_float(storage, "Viewer.ViewpointX").unwrap_or(0.0),
            view_point_y: read_param_float(storage, "Viewer.ViewpointY").unwrap_or(-0.7),
            view_point_z: read_param_float(storage, "Viewer.ViewpointZ").unwrap_or(-1.8),
            view_point_f: read_param_float(storage, "Viewer.ViewpointF").unwrap_or(500.0),
            image_viewer_scale: read_param_float(storage, "Viewer.imageViewScale").unwrap_or(1.0),
        }
    }

    fn read_load_and_save(storage: &FileStorage) -> LoadAndSaveInfo {
        LoadAndSaveInfo {
            load_from: read_param_string(storage, "System.LoadAtlasFromFile"),
            save_to: read_param_string(storage, "System.SaveAtlasToFile"),
        }
    }

    fn read_other(storage: &FileStorage) -> OtherInfo {
        OtherInfo {
            th_far_points: read_param_float(storage, "System.thFarPoints"),
        }
    }
}

impl std::fmt::Display for Settings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SLAM settings:")?;
        let model = match self.camera_model {
            CameraType::PinHole => "Pinhole",
            CameraType::Rectified => "Rectified (Pinhole)",
            CameraType::KannalaBrandt => "KannalaBrandt8 (Fisheye)",
        };
        writeln!(f, "\t-Camera model: {}", model)?;
        if let Some(c) = &self.camera1 {
            let cam = c.calibration.as_ref();
            write!(f, "\t-Camera1 parameters: [")?;
            for i in 0..cam.size() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", cam.get_parameter(i))?;
            }
            writeln!(f, "]")?;
            if let Some(d) = &c.pinhole_distortion {
                writeln!(f, "\t-Camera1 distortion parameters: {:?}", d)?;
            }
        }
        writeln!(
            f,
            "\t-Original image size: [{}, {}]",
            self.image.original_size.width, self.image.original_size.height
        )?;
        writeln!(
            f,
            "\t-Current image size: [{}, {}]",
            self.image.new_size.width, self.image.new_size.height
        )?;
        writeln!(f, "\t-Sequence FPS: {}", self.image.fps)?;
        writeln!(f, "\t-Features per image: {}", self.orb.n_features)?;
        writeln!(f, "\t-ORB scale factor: {}", self.orb.scale_factor)?;
        writeln!(f, "\t-ORB number of scales: {}", self.orb.n_levels)?;
        writeln!(f, "\t-Initial FAST threshold: {}", self.orb.init_th_fast)?;
        writeln!(f, "\t-Minimum FAST threshold: {}", self.orb.min_th_fast)?;
        Ok(())
    }
}

fn read_param_string(storage: &FileStorage, node: &str) -> Option<String> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_string().expect("expected string") {
            return Some(node.to_string().expect("expected string"));
        }
    }
    None
}

fn read_param_float(storage: &FileStorage, node: &str) -> Option<f32> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_real().expect("expected float") {
            return Some(node.to_f32().expect("expected float"));
        }
    }
    None
}

fn read_param_int(storage: &FileStorage, node: &str) -> Option<i64> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_int().expect("expected float") {
            return Some(node.to_i64().expect("expected int"));
        }
    }
    None
}

fn read_param_bool(storage: &FileStorage, node: &str) -> Option<bool> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_int().expect("expected bool") {
            return Some(node.to_i32().expect("expected bool") == 1);
        }
    }
    None
}

fn read_param_mat(storage: &FileStorage, node: &str) -> Option<Mat> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if let Ok(mat) = node.mat() {
            return Some(mat);
        }
    }
    None
}

// Converts a 4×4 Mat (CV_64F row-major) into an Isometry3
// The matrix layout is:
// [ R  t ]
// [ 0  1 ]
// where R is 3×3 rotation and t is 3×1 translation.
fn mat4x4_to_isometry3(mat: &Mat) -> Isometry3<f32> {
    // YAML matrices are written as f64 (CV_64F) by OpenCV.
    let r = Matrix3::new(
        *mat.at_2d::<f64>(0, 0).expect("T_b_c1 row 0 col 0") as f32,
        *mat.at_2d::<f64>(0, 1).expect("T_b_c1 row 0 col 1") as f32,
        *mat.at_2d::<f64>(0, 2).expect("T_b_c1 row 0 col 2") as f32,
        *mat.at_2d::<f64>(1, 0).expect("T_b_c1 row 1 col 0") as f32,
        *mat.at_2d::<f64>(1, 1).expect("T_b_c1 row 1 col 1") as f32,
        *mat.at_2d::<f64>(1, 2).expect("T_b_c1 row 1 col 2") as f32,
        *mat.at_2d::<f64>(2, 0).expect("T_b_c1 row 2 col 0") as f32,
        *mat.at_2d::<f64>(2, 1).expect("T_b_c1 row 2 col 1") as f32,
        *mat.at_2d::<f64>(2, 2).expect("T_b_c1 row 2 col 2") as f32,
    );
    let t = Translation3::new(
        *mat.at_2d::<f64>(0, 3).expect("T_b_c1 row 0 col 3") as f32,
        *mat.at_2d::<f64>(1, 3).expect("T_b_c1 row 1 col 3") as f32,
        *mat.at_2d::<f64>(2, 3).expect("T_b_c1 row 2 col 3") as f32,
    );
    // from_matrix_unchecked is safe here: the source matrix is a rotation read
    // from a calibration file and is assumed to be orthonormal.
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    Isometry3::from_parts(t, rotation)
}

// ---------------------------------------------------------------------------
// Tests — mirror tests/tests_settings.cpp from the C++ ORB-SLAM3 reference.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Sensor;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// RAII wrapper around a temporary YAML file
    struct TempYaml {
        path: PathBuf,
    }
    impl TempYaml {
        fn new(body: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "orbslam3_settings_{}_{}.yaml",
                std::process::id(),
                n
            ));
            let mut f = File::create(&path).expect("create temp yaml");
            // OpenCV's YAML parser requires the version directive on the first
            // line; our Settings parser also requires `File.version: "1.0"`.
            writeln!(f, "%YAML:1.0").unwrap();
            writeln!(f, "---").unwrap();
            writeln!(f, "File.version: \"1.0\"").unwrap();
            f.write_all(body.as_bytes()).unwrap();
            TempYaml { path }
        }
    }
    impl Drop for TempYaml {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    const COMMON_IMAGE: &str = "Camera.width: 640\n\
                                Camera.height: 480\n\
                                Camera.fps: 30\n\
                                Camera.RGB: 1\n";

    const COMMON_ORB: &str = "ORBextractor.nFeatures: 1000\n\
                              ORBextractor.scaleFactor: 1.2\n\
                              ORBextractor.nLevels: 8\n\
                              ORBextractor.iniThFAST: 20\n\
                              ORBextractor.minThFAST: 7\n";

    const COMMON_VIEWER: &str = "Viewer.KeyFrameSize: 0.05\n\
                                 Viewer.KeyFrameLineWidth: 1.0\n\
                                 Viewer.GraphLineWidth: 0.9\n\
                                 Viewer.PointSize: 2.0\n\
                                 Viewer.CameraSize: 0.08\n\
                                 Viewer.CameraLineWidth: 3.0\n\
                                 Viewer.ViewpointX: 0.0\n\
                                 Viewer.ViewpointY: -0.7\n\
                                 Viewer.ViewpointZ: -1.8\n\
                                 Viewer.ViewpointF: 500.0\n";

    fn pinhole_mono_yaml(with_distortion: bool, with_k3: bool, with_resize: bool) -> String {
        let mut s = String::new();
        s += "Camera.type: \"PinHole\"\n";
        s += "Camera1.fx: 500.0\n";
        s += "Camera1.fy: 510.0\n";
        s += "Camera1.cx: 320.0\n";
        s += "Camera1.cy: 240.0\n";
        if with_distortion {
            s += "Camera1.k1: 0.1\n";
            s += "Camera1.k2: -0.05\n";
            s += "Camera1.p1: 0.001\n";
            s += "Camera1.p2: 0.002\n";
            if with_k3 {
                s += "Camera1.k3: 0.01\n";
            }
        }
        s += COMMON_IMAGE;
        if with_resize {
            s += "Camera.newWidth: 320\n";
            s += "Camera.newHeight: 240\n";
        }
        s += COMMON_ORB;
        s += COMMON_VIEWER;
        s
    }

    fn stereo_rectified_yaml() -> String {
        let mut s = String::new();
        s += "Camera.type: \"Rectified\"\n";
        s += "Camera1.fx: 500.0\nCamera1.fy: 510.0\nCamera1.cx: 320.0\nCamera1.cy: 240.0\n";
        s += "Camera2.fx: 500.0\nCamera2.fy: 510.0\nCamera2.cx: 320.0\nCamera2.cy: 240.0\n";
        s += "Stereo.b: 0.12\n";
        s += "Stereo.ThDepth: 35.0\n";
        s += COMMON_IMAGE;
        s += COMMON_ORB;
        s += COMMON_VIEWER;
        s
    }

    fn stereo_pinhole_yaml() -> String {
        let mut s = String::new();
        s += "Camera.type: \"PinHole\"\n";
        s += "Camera1.fx: 500.0\nCamera1.fy: 510.0\nCamera1.cx: 320.0\nCamera1.cy: 240.0\n";
        s += "Camera1.k1: 0.0\nCamera1.k2: 0.0\nCamera1.p1: 0.0\nCamera1.p2: 0.0\n";
        s += "Camera2.fx: 500.0\nCamera2.fy: 510.0\nCamera2.cx: 320.0\nCamera2.cy: 240.0\n";
        s += "Camera2.k1: 0.0\nCamera2.k2: 0.0\nCamera2.p1: 0.0\nCamera2.p2: 0.0\n";
        // Use dt: d (CV_64F) — settings.rs reads matrices as f64.
        s += "Stereo.T_c1_c2: !!opencv-matrix\n";
        s += "  rows: 4\n  cols: 4\n  dt: d\n";
        s += "  data: [1.0, 0.0, 0.0, 0.12,\n";
        s += "         0.0, 1.0, 0.0, 0.0,\n";
        s += "         0.0, 0.0, 1.0, 0.0,\n";
        s += "         0.0, 0.0, 0.0, 1.0]\n";
        s += "Stereo.b: 0.12\n";
        s += "Stereo.ThDepth: 40.0\n";
        s += COMMON_IMAGE;
        s += COMMON_ORB;
        s += COMMON_VIEWER;
        s
    }

    fn kannala_brandt_mono_yaml() -> String {
        let mut s = String::new();
        s += "Camera.type: \"KannalaBrandt8\"\n";
        s += "Camera1.fx: 380.0\nCamera1.fy: 380.0\nCamera1.cx: 320.0\nCamera1.cy: 240.0\n";
        s += "Camera1.k1: 0.01\n";
        s += "Camera1.k2: 0.001\n";
        s += "Camera1.k3: -0.0005\n";
        s += "Camera1.k4: 0.00001\n";
        s += COMMON_IMAGE;
        s += COMMON_ORB;
        s += COMMON_VIEWER;
        s
    }

    fn rgbd_yaml() -> String {
        let mut s = String::new();
        s += "Camera.type: \"PinHole\"\n";
        s += "Camera1.fx: 500.0\nCamera1.fy: 510.0\nCamera1.cx: 320.0\nCamera1.cy: 240.0\n";
        s += "RGBD.DepthMapFactor: 5000.0\n";
        s += "Stereo.b: 0.05\n";
        s += "Stereo.ThDepth: 40.0\n";
        s += COMMON_IMAGE;
        s += COMMON_ORB;
        s += COMMON_VIEWER;
        s
    }

    fn imu_mono_yaml(with_insert_kfs_when_lost: bool, insert_kfs_value: i32) -> String {
        let mut s = String::new();
        s += "Camera.type: \"PinHole\"\n";
        s += "Camera1.fx: 500.0\nCamera1.fy: 510.0\nCamera1.cx: 320.0\nCamera1.cy: 240.0\n";
        s += "IMU.NoiseGyro: 0.0017\n";
        s += "IMU.NoiseAcc: 0.002\n";
        s += "IMU.GyroWalk: 1.9e-05\n";
        s += "IMU.AccWalk: 3.0e-03\n";
        s += "IMU.Frequency: 200.0\n";
        s += "IMU.T_b_c1: !!opencv-matrix\n";
        s += "  rows: 4\n  cols: 4\n  dt: d\n";
        s += "  data: [1.0, 0.0, 0.0, 0.05,\n";
        s += "         0.0, 1.0, 0.0, 0.04,\n";
        s += "         0.0, 0.0, 1.0, 0.03,\n";
        s += "         0.0, 0.0, 0.0, 1.0]\n";
        if with_insert_kfs_when_lost {
            s += &format!("IMU.InsertKFsWhenLost: {}\n", insert_kfs_value);
        }
        s += COMMON_IMAGE;
        s += COMMON_ORB;
        s += COMMON_VIEWER;
        s
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4_f32.max(b.abs() * 1e-4)
    }

    // Pinhole monocular
    #[test]
    fn pinhole_mono_without_distortion() {
        let y = TempYaml::new(&pinhole_mono_yaml(false, false, false));
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        assert_eq!(s.camera_model, CameraType::PinHole);
        let cam = &s.camera1.as_ref().expect("camera1").calibration;
        assert!(approx(cam.get_parameter(0), 500.0));
        assert!(approx(cam.get_parameter(1), 510.0));
        assert!(approx(cam.get_parameter(2), 320.0));
        assert!(approx(cam.get_parameter(3), 240.0));

        assert!(!s.need_to_undistort);
        assert!(!s.image.need_to_resize);
        assert!(!s.need_to_rectify);

        assert_eq!(s.image.new_size.width, 640);
        assert_eq!(s.image.new_size.height, 480);
        assert_eq!(s.image.fps, 30);
        assert!(s.image.rgb);
    }

    #[test]
    fn pinhole_mono_with_4coef_distortion_triggers_undistort() {
        let y = TempYaml::new(&pinhole_mono_yaml(true, false, false));
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        assert_eq!(s.camera_model, CameraType::PinHole);
        assert!(s.need_to_undistort);

        let d = s
            .camera1
            .as_ref()
            .unwrap()
            .pinhole_distortion
            .as_ref()
            .expect("distortion");
        assert_eq!(d.len(), 4);
        assert!(approx(d[0], 0.1));
        assert!(approx(d[1], -0.05));
        assert!(approx(d[2], 0.001));
        assert!(approx(d[3], 0.002));
    }

    #[test]
    fn pinhole_mono_with_5coef_distortion() {
        let y = TempYaml::new(&pinhole_mono_yaml(true, true, false));
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        let d = s
            .camera1
            .as_ref()
            .unwrap()
            .pinhole_distortion
            .as_ref()
            .expect("distortion");
        assert_eq!(d.len(), 5);
        assert!(approx(d[4], 0.01));
    }

    #[test]
    fn image_resize_updates_calibration_and_sets_need_to_resize() {
        let y = TempYaml::new(&pinhole_mono_yaml(false, false, true));
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        assert!(s.image.need_to_resize);
        assert_eq!(s.image.new_size.width, 320);
        assert_eq!(s.image.new_size.height, 240);

        let cam = &s.camera1.as_ref().unwrap().calibration;
        // half scale → fx, fy, cx, cy halved
        assert!(approx(cam.get_parameter(0), 250.0));
        assert!(approx(cam.get_parameter(1), 255.0));
        assert!(approx(cam.get_parameter(2), 160.0));
        assert!(approx(cam.get_parameter(3), 120.0));
    }

    // KannalaBrandt monocular
    #[test]
    fn kannala_brandt8_monocular() {
        let y = TempYaml::new(&kannala_brandt_mono_yaml());
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        assert_eq!(s.camera_model, CameraType::KannalaBrandt);
        let cam_info = s.camera1.as_ref().expect("camera1");
        let cam = &cam_info.calibration;
        // 4 intrinsics + 4 distortion params packed into the camera object.
        assert_eq!(cam.size(), 8);
        assert!(approx(cam.get_parameter(0), 380.0));
        assert!(approx(cam.get_parameter(4), 0.01));
        assert!(approx(cam.get_parameter(7), 0.00001));
        // KB monocular does not populate the pinhole distortion vector.
        assert!(!s.need_to_undistort);
        assert!(cam_info.pinhole_distortion.is_none());
    }

    // Stereo
    #[test]
    fn stereo_rectified_uses_stereo_b_directly() {
        let y = TempYaml::new(&stereo_rectified_yaml());
        let s = Settings::new(&y.path, Sensor::Stereo).expect("settings");

        assert_eq!(s.camera_model, CameraType::Rectified);
        assert!(!s.need_to_rectify);
        let stereo = s.stereo.as_ref().expect("stereo");
        assert!(approx(stereo.b, 0.12));
        assert!(approx(stereo.bf, 0.12 * 500.0));
    }

    #[test]
    fn stereo_pinhole_with_t_c1_c2_triggers_rectification_maps() {
        let y = TempYaml::new(&stereo_pinhole_yaml());
        let s = Settings::new(&y.path, Sensor::Stereo).expect("settings");

        assert_eq!(s.camera_model, CameraType::PinHole);
        assert!(s.need_to_rectify);

        let stereo = s.stereo.as_ref().expect("stereo");
        // Baseline = ‖t‖ from T_c1_c2.
        assert!(approx(stereo.b, 0.12));

        // Rectification maps must have been precomputed.
        assert!(!stereo.m1l.empty());
        assert!(!stereo.m2l.empty());
        assert!(!stereo.m1r.empty());
        assert!(!stereo.m2r.empty());
        assert_eq!(stereo.m1l.size().unwrap(), s.image.new_size);
        assert_eq!(stereo.m1r.size().unwrap(), s.image.new_size);

        // bf must have been recomputed using P1(0,0).
        let fx = s.camera1.as_ref().unwrap().calibration.get_parameter(0);
        assert!(approx(stereo.bf, stereo.b * fx));

        // Tlr translation magnitude == baseline.
        assert!(approx(stereo.tlr.translation.vector.norm(), 0.12));
    }

    // RGB-D
    #[test]
    fn rgbd_parses_depth_map_factor_and_stereo_baseline() {
        let y = TempYaml::new(&rgbd_yaml());
        let s = Settings::new(&y.path, Sensor::RGBD).expect("settings");

        let rgbd = s.rgbd.as_ref().expect("rgbd");
        assert!(approx(rgbd.depth_map_factor, 5000.0));
        assert!(approx(rgbd.b, 0.05));
        assert!(approx(rgbd.bf, 0.05 * 500.0));
        assert!(approx(rgbd.th_depth, 40.0));
    }

    // IMU
    #[test]
    fn imu_monocular_parses_noise_walk_tbc_and_default_insert_kfs_when_lost() {
        let y = TempYaml::new(&imu_mono_yaml(false, 0));
        let s = Settings::new(&y.path, Sensor::IMUMonocular).expect("settings");

        let imu = s.imu.as_ref().expect("imu");
        assert!(approx(imu.noise_gyro, 0.0017));
        assert!(approx(imu.noise_acc, 0.002));
        assert!(approx(imu.gyro_walk, 1.9e-5));
        // NOTE: stored as `gyro_acc` in IMUInfo (historical typo for AccWalk).
        assert!(approx(imu.acc_walk, 3.0e-3));
        assert!(approx(imu.imu_frequency, 200.0));

        assert!(approx(imu.tbc.translation.vector.x, 0.05));
        assert!(approx(imu.tbc.translation.vector.y, 0.04));
        assert!(approx(imu.tbc.translation.vector.z, 0.03));

        // Default when key absent.
        assert!(imu.insert_kfs_when_lost);
    }

    #[test]
    fn imu_insert_kfs_when_lost_honored_false() {
        let y = TempYaml::new(&imu_mono_yaml(true, 0));
        let s = Settings::new(&y.path, Sensor::IMUMonocular).expect("settings");
        assert!(!s.imu.as_ref().unwrap().insert_kfs_when_lost);
    }

    #[test]
    fn imu_insert_kfs_when_lost_honored_true() {
        let y = TempYaml::new(&imu_mono_yaml(true, 1));
        let s = Settings::new(&y.path, Sensor::IMUMonocular).expect("settings");
        assert!(s.imu.as_ref().unwrap().insert_kfs_when_lost);
    }

    // ORB / Viewer / optional parameters
    #[test]
    fn orb_and_viewer_parameters_round_trip() {
        let y = TempYaml::new(&pinhole_mono_yaml(false, false, false));
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        assert_eq!(s.orb.n_features, 1000);
        assert!(approx(s.orb.scale_factor, 1.2));
        assert_eq!(s.orb.n_levels, 8);
        assert_eq!(s.orb.init_th_fast, 20);
        assert_eq!(s.orb.min_th_fast, 7);

        assert!(approx(s.viewer.keyframe_size, 0.05));
        assert!(approx(s.viewer.keyframe_linewidth, 1.0));
        assert!(approx(s.viewer.graph_linewidth, 0.9));
        assert!(approx(s.viewer.point_size, 2.0));
        assert!(approx(s.viewer.camera_size, 0.08));
        assert!(approx(s.viewer.camera_linewidth, 3.0));
        assert!(approx(s.viewer.view_point_x, 0.0));
        assert!(approx(s.viewer.view_point_y, -0.7));
        assert!(approx(s.viewer.view_point_z, -1.8));
        assert!(approx(s.viewer.view_point_f, 500.0));

        // imageViewScale defaults to 1.0 when absent.
        assert!(approx(s.viewer.image_viewer_scale, 1.0));

        // Atlas load/save are optional → None when absent.
        assert!(s.load_and_save.load_from.is_none());
        assert!(s.load_and_save.save_to.is_none());
    }

    #[test]
    fn optional_atlas_load_save_and_image_view_scale_parsed_when_present() {
        let mut body = pinhole_mono_yaml(false, false, false);
        body += "Viewer.imageViewScale: 0.5\n";
        body += "System.LoadAtlasFromFile: \"my_atlas\"\n";
        body += "System.SaveAtlasToFile: \"out_atlas\"\n";
        body += "System.thFarPoints: 50.0\n";
        let y = TempYaml::new(&body);
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        assert!(approx(s.viewer.image_viewer_scale, 0.5));
        assert_eq!(s.load_and_save.load_from.as_deref(), Some("my_atlas"));
        assert_eq!(s.load_and_save.save_to.as_deref(), Some("out_atlas"));
        assert!(approx(s.other.th_far_points.expect("thFarPoints"), 50.0));
    }

    // Display impl
    #[test]
    fn display_emits_human_readable_summary() {
        let y = TempYaml::new(&pinhole_mono_yaml(true, false, false));
        let s = Settings::new(&y.path, Sensor::Monocular).expect("settings");

        let text = format!("{}", s);
        assert!(text.contains("SLAM settings"));
        assert!(text.contains("Pinhole"));
        assert!(text.contains("distortion parameters"));
        assert!(text.contains("Sequence FPS"));
        assert!(text.contains("Features per image"));
    }
}
