use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::{Isometry3, Matrix2x1, Matrix2x3, Matrix3, Point2, Point3, Vector3};
use opencv::core::{KeyPoint, Mat, Point2f, Point3f};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::two_view_reconstruction::ReconstructResult;

pub mod kannala_brandt8;
pub mod pinhole;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Type {
    Pinhole,
    Fisheye,
}

static NEXT_GEOMETRIC_CAMERA_ID: AtomicU64 = AtomicU64::new(0);
fn next_geometric_camera_id() -> u64 {
    NEXT_GEOMETRIC_CAMERA_ID.fetch_add(1, Ordering::Relaxed)
}

/// Current value of the global camera-id counter (for serialization).
pub(crate) fn peek_next_camera_id() -> u64 {
    NEXT_GEOMETRIC_CAMERA_ID.load(Ordering::Relaxed)
}
/// Restore the global camera-id counter (after deserialization).
pub(crate) fn set_next_camera_id(v: u64) {
    NEXT_GEOMETRIC_CAMERA_ID.store(v, Ordering::Relaxed);
}

// `Send + Sync` supertrait: cameras are shared across the Tracking /
// LocalMapping / LoopClosing threads via `Arc<dyn GeometricCamera>`. The
// implementors (`Pinhole`, `KannalaBrandt8`) embed a `TwoViewReconstruction`,
// which holds `Vec<KeyPoint>`; that type's audited `unsafe impl Send + Sync`
// (see `unsafe_impl_ffi_send_sync` in `lib.rs`) lets them auto-derive both.
pub trait GeometricCamera: Send + Sync {
    fn new() -> Self
    where
        Self: Sized;
    fn with_params(params: Vec<f32>) -> Self
    where
        Self: Sized;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn project(&self, p3d: &Point3f) -> Point2f;
    fn project_n(&self, p3d: &Point3<f32>) -> Point2<f32>;
    fn project_n_d(&self, p3d: &Point3<f64>) -> Point2<f64>;
    fn project_mat(&self, p3d: &Point3f) -> Point2<f32>;

    fn uncertainty(&self, p2d: &Matrix2x1<f64>) -> f32;

    fn unproject_n(&self, p2d: &Point2f) -> Vector3<f32>;
    fn unproject(&self, p2d: &Point2f) -> Point3f;

    fn project_jac(&self, p3d: &Point3<f64>) -> Matrix2x3<f64>;

    fn reconstruct_with_two_views(
        &mut self,
        keys1: &Vec<KeyPoint>,
        keys2: &Vec<KeyPoint>,
        matches: &Vec<Option<usize>>,
    ) -> Option<ReconstructResult>;

    fn to_k(&self) -> Mat;
    fn to_k_n(&self) -> Matrix3<f32>;

    fn epipolar_constrain(
        &self,
        other_camera: &dyn GeometricCamera,
        kp1: &KeyPoint,
        kp2: &KeyPoint,
        r12: &Matrix3<f32>,
        t12: &Point3<f32>,
        sigma_level: f32,
        unc: f32,
    ) -> bool;

    fn get_parameter(&self, i: usize) -> f32;
    fn set_parameter(&mut self, p: f32, i: usize);

    fn size(&self) -> usize;

    fn match_and_triangulate(
        &self,
        kp1: &KeyPoint,
        kp2: &KeyPoint,
        other: &dyn GeometricCamera,
        tcw1: &Isometry3<f32>,
        tcw2: &Isometry3<f32>,
        sigma_level1: f32,
        sigma_level2: f32,
    ) -> Option<Vector3<f32>>;

    fn get_id(&self) -> u64;
    fn get_type(&self) -> Type;

    fn is_equal(&self, other: &Arc<dyn GeometricCamera>) -> bool;
}

impl std::fmt::Debug for dyn GeometricCamera {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Geometric Camera {} {:?}",
            self.get_id(),
            self.get_type(),
        )
    }
}

// Externally tagged (serde default): encodes as a `u32` variant index plus the
// payload, which is what non-self-describing binary formats like postcard need.
// Avoid `#[serde(tag = ...)]` here — adjacently/internally tagged enums can route
// through `deserialize_any`, which postcard does not support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometricCameraSnapshot {
    Pinhole(pinhole::Pinhole),
    KannalaBrandt8(kannala_brandt8::KannalaBrandt8),
}
impl GeometricCameraSnapshot {
    pub fn into_boxed(self) -> Box<dyn GeometricCamera> {
        match self {
            GeometricCameraSnapshot::Pinhole(cam) => Box::new(cam),
            GeometricCameraSnapshot::KannalaBrandt8(cam) => Box::new(cam),
        }
    }
}

impl TryFrom<&dyn GeometricCamera> for GeometricCameraSnapshot {
    type Error = &'static str;
    fn try_from(camera: &dyn GeometricCamera) -> Result<Self, Self::Error> {
        if let Some(cam) = camera.as_any().downcast_ref::<pinhole::Pinhole>() {
            return Ok(GeometricCameraSnapshot::Pinhole(cam.clone()));
        }
        if let Some(cam) = camera
            .as_any()
            .downcast_ref::<kannala_brandt8::KannalaBrandt8>()
        {
            return Ok(GeometricCameraSnapshot::KannalaBrandt8(cam.clone()));
        }
        Err("unknown GeometricCamera implementation")
    }
}
