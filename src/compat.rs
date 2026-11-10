#[inline]
pub fn cv_round_f32(x: f32) -> i32 {
    x.round_ties_even() as i32
}

#[inline]
pub fn cv_round_f64(x: f64) -> i32 {
    x.round_ties_even() as i32
}

#[inline]
pub fn cv_floor_f32(x: f32) -> i32 {
    x.floor() as i32
}
#[inline]
pub fn cv_floor_f64(x: f64) -> i32 {
    x.floor() as i32
}

#[inline]
pub fn cv_ceil_f32(x: f32) -> i32 {
    x.ceil() as i32
}
#[inline]
pub fn cv_ceil_f64(x: f64) -> i32 {
    x.ceil() as i32
}

/// `cvt_color` with the trailing `AlgorithmHint` that OpenCV 4.9+ bindings
/// require and 4.8 bindings reject.
#[cfg(opencv_ge_49)]
pub fn cvt_color_compat(
    src: &opencv::core::Mat,
    dst: &mut opencv::core::Mat,
    code: i32,
    dst_cn: i32,
) -> opencv::Result<()> {
    opencv::imgproc::cvt_color(
        src,
        dst,
        code,
        dst_cn,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
}

/// See [`cvt_color_compat`].
#[cfg(not(opencv_ge_49))]
pub fn cvt_color_compat(
    src: &opencv::core::Mat,
    dst: &mut opencv::core::Mat,
    code: i32,
    dst_cn: i32,
) -> opencv::Result<()> {
    opencv::imgproc::cvt_color(src, dst, code, dst_cn)
}

/// `gaussian_blur` with the trailing `AlgorithmHint` that OpenCV 4.9+ bindings
/// require and 4.8 bindings reject.
#[cfg(opencv_ge_49)]
pub fn gaussian_blur_compat(
    src: &opencv::core::Mat,
    dst: &mut opencv::core::Mat,
    ksize: opencv::core::Size,
    sigma_x: f64,
    sigma_y: f64,
    border: i32,
) -> opencv::Result<()> {
    opencv::imgproc::gaussian_blur(
        src,
        dst,
        ksize,
        sigma_x,
        sigma_y,
        border,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
}

/// See [`gaussian_blur_compat`].
#[cfg(not(opencv_ge_49))]
pub fn gaussian_blur_compat(
    src: &opencv::core::Mat,
    dst: &mut opencv::core::Mat,
    ksize: opencv::core::Size,
    sigma_x: f64,
    sigma_y: f64,
    border: i32,
) -> opencv::Result<()> {
    opencv::imgproc::gaussian_blur(src, dst, ksize, sigma_x, sigma_y, border)
}
