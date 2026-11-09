pub mod atlas;
pub mod camera_models;
pub mod compat;
pub mod converter;
pub mod frame;
pub mod g2o_core;
pub mod g2o_types;
pub mod geometric_tools;
pub mod imu_types;
pub mod key_frame;
pub mod key_frame_database;
pub mod local_mapping;
pub mod loop_closing;
pub mod map;
pub mod map_point;
pub mod optimizable_types;
pub mod optimizer;
pub mod orb_extractor;
pub mod orb_matcher;
pub mod orb_vocabulary;
pub mod serialization_utils;
pub mod settings;
pub mod system;
#[cfg(test)]
mod test_helpers;
pub mod tracking;
pub mod two_view_reconstruction;
pub mod viewer;

/// Asserts that an aggregate is safe to send and share across threads even
/// though it stores OpenCV FFI handles (`cv::Mat`, `cv::KeyPoint`). Those types
/// are `!Send + !Sync` only because they carry raw `*mut c_void` pointers into
/// C++-owned storage; the aggregates below uphold a discipline that makes
/// concurrent use sound.
///
/// # Safety
///
/// This mirrors the threading contract upstream ORB-SLAM3 relies on: Tracking,
/// LocalMapping and LoopClosing run on separate `std::thread`s and share
/// `MapPoint*` / `KeyFrame*` / `Map*`. It is sound here for the same reasons:
///
/// * Every `Mat` retained in one of these shared structs owns a **deep-copied,
///   non-aliased** buffer — `try_clone` on store and on read, mirroring
///   `MapPoint::GetDescriptor()` returning `mDescriptor.clone()` and the
///   `KeyFrame(Frame&)` constructor cloning `mDescriptors`. No `Mat` buffer is
///   shared between two of these structs, so no cross-thread buffer aliasing
///   can occur. (OpenCV's `Mat` refcount is additionally updated atomically via
///   `CV_XADD`, so the clones themselves are race-free.)
/// * Stored `KeyPoint`s are construction-time-immutable plain data (`pt`,
///   `size`, `angle`, …) behind the FFI handle and are only ever read
///   concurrently.
/// * All interior mutation goes through the struct's own `RwLock` fields,
///   matching upstream's per-object `mMutex*` locking.
///
/// Maintainer note: this is an aggregate-level promise. If you add a field whose
/// thread-safety is *not* covered above (e.g. an `Rc`, `Cell`, or a raw pointer
/// with shared-mutation semantics), you must re-audit the corresponding impl.
macro_rules! unsafe_impl_ffi_send_sync {
    ($($t:ty),+ $(,)?) => {
        $(
            // SAFETY: see the `unsafe_impl_ffi_send_sync` macro documentation.
            unsafe impl Send for $t {}
            // SAFETY: see the `unsafe_impl_ffi_send_sync` macro documentation.
            unsafe impl Sync for $t {}
        )+
    };
}

unsafe_impl_ffi_send_sync! {
    crate::frame::Frame,
    crate::frame::FrameConstants,
    crate::key_frame::KeyFrame,
    crate::map_point::MapPoint,
    crate::two_view_reconstruction::TwoViewReconstruction,
}
