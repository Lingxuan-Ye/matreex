use super::Matrix;

impl<T> Default for Matrix<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
