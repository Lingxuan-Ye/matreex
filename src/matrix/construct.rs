use super::order::Order;
use super::shape::{AxisShape, Shape};
use super::Matrix;
use crate::error::Result;

impl<T> Matrix<T> {
    /// Creates a new, empty [`Matrix<T>`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Matrix;
    ///
    /// let matrix = Matrix::<i32>::new();
    /// assert_eq!(matrix.nrows(), 0);
    /// assert_eq!(matrix.ncols(), 0);
    /// assert!(matrix.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            order: Order::default(),
            shape: AxisShape::default(),
            data: Vec::new(),
        }
    }

    /// Creates a new, empty [`Matrix<T>`] with at least the specified
    /// capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Matrix;
    ///
    /// let matrix = Matrix::<i32>::with_capacity(10);
    /// assert_eq!(matrix.nrows(), 0);
    /// assert_eq!(matrix.ncols(), 0);
    /// assert!(matrix.is_empty());
    /// assert!(matrix.capacity() >= 10);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            order: Order::default(),
            shape: AxisShape::default(),
            data: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new [`Matrix<T>`] with the specified shape, filled with
    /// the given value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if total bytes stored exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error, Matrix};
    ///
    /// let result = Matrix::with_value((2, 3), 0);
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    ///
    /// let result = Matrix::<u8>::with_value((usize::MAX, 2), 0);
    /// assert_eq!(result, Err(Error::SizeOverflow));
    ///
    /// let result = Matrix::<u8>::with_value((isize::MAX as usize + 1, 1), 0);
    /// assert_eq!(result, Err(Error::CapacityOverflow));
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn with_value<S>(shape: S, value: T) -> Result<Self>
    where
        T: Clone,
        S: Into<Shape>,
    {
        let order = Order::default();
        let shape = shape.into().try_to_axis_shape(order)?;
        let size = Self::check_size(shape.size())?;
        let data = vec![value; size];
        Ok(Self { order, shape, data })
    }

    /// Creates a new [`Matrix<T>`] with the specified shape, filled with
    /// elements generated by the provided initializer.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if total bytes stored exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error, Matrix};
    ///
    /// let result = Matrix::with_initializer((2, 3), Default::default);
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    ///
    /// let result = Matrix::<u8>::with_initializer((usize::MAX, 2), Default::default);
    /// assert_eq!(result, Err(Error::SizeOverflow));
    ///
    /// let result = Matrix::<u8>::with_initializer((isize::MAX as usize + 1, 1), Default::default);
    /// assert_eq!(result, Err(Error::CapacityOverflow));
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn with_initializer<S, F>(shape: S, initializer: F) -> Result<Self>
    where
        S: Into<Shape>,
        F: FnMut() -> T,
    {
        let order = Order::default();
        let shape = shape.into().try_to_axis_shape(order)?;
        let size = Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(size);
        data.resize_with(size, initializer);
        Ok(Self { order, shape, data })
    }

    /// Creates a new [`Matrix<T>`] with the specified shape, filled with
    /// the default value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if total bytes stored exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error, Matrix};
    ///
    /// let result = Matrix::with_default((2, 3));
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    ///
    /// let result = Matrix::<u8>::with_default((usize::MAX, 2));
    /// assert_eq!(result, Err(Error::SizeOverflow));
    ///
    /// let result = Matrix::<u8>::with_default((isize::MAX as usize + 1, 1));
    /// assert_eq!(result, Err(Error::CapacityOverflow));
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn with_default<S>(shape: S) -> Result<Self>
    where
        T: Default,
        S: Into<Shape>,
    {
        Self::with_initializer(shape, T::default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::matrix;

    #[test]
    fn test_new() {
        let matrix = Matrix::<i32>::new();
        assert_eq!(matrix.order, Order::default());
        assert_eq!(matrix.nrows(), 0);
        assert_eq!(matrix.ncols(), 0);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let matrix = Matrix::<i32>::with_capacity(10);
        assert_eq!(matrix.order, Order::default());
        assert_eq!(matrix.nrows(), 0);
        assert_eq!(matrix.ncols(), 0);
        assert!(matrix.is_empty());
        assert!(matrix.capacity() >= 10);
    }

    #[test]
    fn test_with_value() {
        assert_eq!(
            Matrix::with_value((2, 3), 0).unwrap(),
            matrix![[0, 0, 0], [0, 0, 0]]
        );

        assert_eq!(
            Matrix::<i32>::with_value((usize::MAX, 2), 0).unwrap_err(),
            Error::SizeOverflow
        );

        assert_eq!(
            Matrix::<u8>::with_value((isize::MAX as usize + 1, 1), 0).unwrap_err(),
            Error::CapacityOverflow
        );
        assert_eq!(
            Matrix::<i32>::with_value((isize::MAX as usize / 4 + 1, 1), 0).unwrap_err(),
            Error::CapacityOverflow
        );

        // zero-sized types
        assert!(Matrix::<()>::with_value((isize::MAX as usize + 1, 1), ()).is_ok());
    }

    #[test]
    fn test_with_initializer() {
        assert_eq!(
            Matrix::with_initializer((2, 3), Default::default).unwrap(),
            matrix![[0, 0, 0], [0, 0, 0]]
        );

        assert_eq!(
            Matrix::<i32>::with_initializer((usize::MAX, 2), Default::default).unwrap_err(),
            Error::SizeOverflow
        );

        assert_eq!(
            Matrix::<u8>::with_initializer((isize::MAX as usize + 1, 1), Default::default)
                .unwrap_err(),
            Error::CapacityOverflow
        );
        assert_eq!(
            Matrix::<i32>::with_initializer((isize::MAX as usize / 4 + 1, 1), Default::default)
                .unwrap_err(),
            Error::CapacityOverflow
        );

        // The following test case for zero-sized types is impractical to
        // run in debug mode, and since `#[cfg(not(debug_assertions))]` does
        // not strictly match release mode, these tests are commented out.

        // assert!(
        //     Matrix::<()>::with_initializer((isize::MAX as usize + 1, 1), Default::default).is_ok()
        // );
    }

    #[test]
    fn test_with_default() {
        assert_eq!(
            Matrix::with_default((2, 3)).unwrap(),
            matrix![[0, 0, 0], [0, 0, 0]]
        );

        assert_eq!(
            Matrix::<i32>::with_default((usize::MAX, 2)).unwrap_err(),
            Error::SizeOverflow
        );

        assert_eq!(
            Matrix::<u8>::with_default((isize::MAX as usize + 1, 1)).unwrap_err(),
            Error::CapacityOverflow
        );
        assert_eq!(
            Matrix::<i32>::with_default((isize::MAX as usize / 4 + 1, 1)).unwrap_err(),
            Error::CapacityOverflow
        );

        // assert!(Matrix::<()>::with_default((isize::MAX as usize + 1, 1)).is_ok());
    }
}
