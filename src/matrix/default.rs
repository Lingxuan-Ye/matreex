use super::Matrix;

impl<T> Default for Matrix<T> {
    fn default() -> Self {
        Self::new()
    }
}
