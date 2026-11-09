use opencv::core::{
    KeyPoint, KeyPointTraitConst, Mat, MatTraitConst, MatTraitConstManual, MatTraitManual,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Plain, serde-friendly mirror of an OpenCV [`KeyPoint`].
///
/// Used directly inside the snapshot structs (see `KeyFrameSnapshot`) rather than
/// as a `#[serde(with = ...)]` adapter, so the backup layout is fully owned by
/// serde-derivable types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableKeyPoint {
    x: f32,
    y: f32,
    size: f32,
    angle: f32,
    response: f32,
    octave: i32,
    class_id: i32,
}
impl SerializableKeyPoint {
    pub fn from_kp(kp: &KeyPoint) -> Self {
        let pt = kp.pt();
        Self {
            x: pt.x,
            y: pt.y,
            size: kp.size(),
            angle: kp.angle(),
            response: kp.response(),
            octave: kp.octave(),
            class_id: kp.class_id(),
        }
    }
    pub fn to_kp(&self) -> opencv::Result<KeyPoint> {
        KeyPoint::new_coords(
            self.x,
            self.y,
            self.size,
            self.angle,
            self.response,
            self.octave,
            self.class_id,
        )
    }

    /// Convert a slice of OpenCV keypoints into serde-friendly mirrors.
    pub fn from_slice(kps: &[KeyPoint]) -> Vec<Self> {
        kps.iter().map(Self::from_kp).collect()
    }

    /// Rebuild a vector of OpenCV keypoints from their mirrors.
    pub fn to_vec(items: &[Self]) -> opencv::Result<Vec<KeyPoint>> {
        items.iter().map(Self::to_kp).collect()
    }
}

/// Drop-in adapter for `#[serde(with = "mat_serde")]` on `Mat` fields.
pub mod mat_serde {
    use super::*;

    pub fn serialize<S: Serializer>(mat: &Mat, s: S) -> Result<S::Ok, S::Error> {
        SerializableMat::from_mat(mat)
            .map_err(|e| serde::ser::Error::custom(format!("Mat serialize: {e}")))?
            .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Mat, D::Error> {
        SerializableMat::deserialize(d)?
            .to_mat()
            .map_err(|e| serde::de::Error::custom(format!("Mat deserialize: {e}")))
    }
}

#[derive(Serialize, Deserialize)]
pub struct SerializableMat {
    rows: i32,
    cols: i32,
    typ: i32,
    data: Vec<u8>,
}
impl SerializableMat {
    pub fn from_mat(mat: &Mat) -> opencv::Result<Self> {
        // Make sure the Mat is continuous, otherwise data_bytes()
        // may include row padding or may not represent the logical matrix cleanly.
        let mat = if mat.is_continuous() {
            mat.try_clone()?
        } else {
            let mut cloned = Mat::default();
            mat.copy_to(&mut cloned)?;
            cloned
        };
        Ok(Self {
            rows: mat.rows(),
            cols: mat.cols(),
            typ: mat.typ(),
            data: mat.data_bytes()?.to_vec(),
        })
    }

    pub fn to_mat(&self) -> opencv::Result<Mat> {
        let mut mat = Mat::new_rows_cols_with_default(
            self.rows,
            self.cols,
            self.typ,
            opencv::core::Scalar::default(),
        )?;
        mat.data_bytes_mut()?.copy_from_slice(&self.data);
        Ok(mat)
    }
}
