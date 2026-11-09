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
