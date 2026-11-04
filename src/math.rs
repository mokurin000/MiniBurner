//! A tiny column-major matrix library for the donut camera.
//!
//! Only the operations needed to place the camera and orient the model are
//! implemented, so the app does not pull in a full linear algebra crate.

use std::ops::Mul;

/// A column-major 4x4 matrix, matching the memory layout of WGSL's
/// `mat4x4<f32>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// The columns of the matrix.
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Build a matrix from its columns.
    pub const fn from_cols(cols: [[f32; 4]; 4]) -> Self {
        Self { cols }
    }

    /// A rotation around the X axis by `angle` radians.
    pub fn from_rotation_x(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::from_cols([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, sin, 0.0],
            [0.0, -sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// A rotation around the Z axis by `angle` radians.
    pub fn from_rotation_z(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::from_cols([
            [cos, sin, 0.0, 0.0],
            [-sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// A translation by `offset`.
    pub fn from_translation(offset: [f32; 3]) -> Self {
        Self::from_cols([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [offset[0], offset[1], offset[2], 1.0],
        ])
    }

    /// A right-handed view matrix looking from `eye` towards `target`.
    pub fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Self {
        let forward = normalize(sub(target, eye));
        let right = normalize(cross(forward, up));
        let up = cross(right, forward);
        Self::from_cols([
            [right[0], up[0], -forward[0], 0.0],
            [right[1], up[1], -forward[1], 0.0],
            [right[2], up[2], -forward[2], 0.0],
            [-dot(right, eye), -dot(up, eye), dot(forward, eye), 1.0],
        ])
    }

    /// A right-handed perspective projection mapping the view frustum to the
    /// `0..1` clip-space depth range expected by wgpu.
    pub fn perspective_rh(fov_y: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let inv_range = 1.0 / (z_near - z_far);
        Self::from_cols([
            [f / aspect_ratio, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, z_far * inv_range, -1.0],
            [0.0, 0.0, z_near * z_far * inv_range, 0.0],
        ])
    }

    /// Transform a point (w = 1) and return the result including w.
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 4] {
        self.transform([point[0], point[1], point[2], 1.0])
    }

    /// Transform a homogeneous vector.
    pub fn transform(&self, vector: [f32; 4]) -> [f32; 4] {
        let mut result = [0.0; 4];
        for (row, value) in result.iter_mut().enumerate() {
            *value = (0..4).map(|col| self.cols[col][row] * vector[col]).sum();
        }
        result
    }
}

impl Mul for Mat4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}

impl Mul<&Mat4> for &Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: &Mat4) -> Mat4 {
        let mut cols = [[0.0; 4]; 4];
        for (col, out) in cols.iter_mut().enumerate() {
            *out = self.transform(rhs.cols[col]);
        }
        Mat4::from_cols(cols)
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-5, "expected {right}, got {left}");
    }

    #[test]
    fn identity_leaves_points_untouched() {
        let point = Mat4::IDENTITY.transform_point([1.0, -2.0, 3.0]);
        assert_eq!(point, [1.0, -2.0, 3.0, 1.0]);
    }

    #[test]
    fn rotation_x_maps_y_to_negative_z() {
        let rotated =
            Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2).transform([0.0, 1.0, 0.0, 0.0]);
        assert_close(rotated[1], 0.0);
        assert_close(rotated[2], -1.0);
    }

    #[test]
    fn rotation_z_spins_in_the_screen_plane() {
        let rotated =
            Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2).transform([1.0, 0.0, 0.0, 0.0]);
        assert_close(rotated[0], 0.0);
        assert_close(rotated[1], 1.0);
        assert_close(rotated[2], 0.0);
    }

    #[test]
    fn look_at_places_eye_at_origin() {
        let eye = [1.0, 2.0, 3.0];
        let view = Mat4::look_at_rh(eye, [0.0; 3], [0.0, 1.0, 0.0]);
        let origin = view.transform_point(eye);
        assert_close(origin[0], 0.0);
        assert_close(origin[1], 0.0);
        assert_close(origin[2], 0.0);
    }

    #[test]
    fn perspective_maps_near_to_zero_and_far_to_one() {
        let (near, far) = (0.1, 100.0);
        let projection = Mat4::perspective_rh(45f32.to_radians(), 4.0 / 3.0, near, far);
        let near_point = projection.transform_point([0.0, 0.0, -near]);
        let far_point = projection.transform_point([0.0, 0.0, -far]);
        assert_close(near_point[2] / near_point[3], 0.0);
        assert_close(far_point[2] / far_point[3], 1.0);
    }
}
