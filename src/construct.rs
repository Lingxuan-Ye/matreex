use crate::Matrix;
use crate::error::Result;
use crate::index::Index;
use crate::order::Order;
use crate::shape::{AsShape, AxisShape};
use alloc::vec;
use alloc::vec::Vec;

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
    #[inline]
    pub fn new() -> Self {
        let order = Order::default();
        let shape = AxisShape::default();
        let data = Vec::new();
        Self { order, shape, data }
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
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let order = Order::default();
        let shape = AxisShape::default();
        let data = Vec::with_capacity(capacity);
        Self { order, shape, data }
    }

    /// Creates a new [`Matrix<T>`] with the specified shape, filled with
    /// the default value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let result = Matrix::with_default((2, 3));
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn with_default<S>(shape: S) -> Result<Self>
    where
        T: Default,
        S: AsShape,
    {
        let order = Order::default();
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.resize_with(size, T::default);
        Ok(Self { order, shape, data })
    }

    /// Creates a new [`Matrix<T>`] with the specified shape, filled with
    /// the given value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let result = Matrix::with_value((2, 3), 0);
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn with_value<S>(shape: S, value: T) -> Result<Self>
    where
        T: Clone,
        S: AsShape,
    {
        let order = Order::default();
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let data = vec![value; size];
        Ok(Self { order, shape, data })
    }

    /// Creates a new [`Matrix<T>`] with the specified shape, where each
    /// element is initialized using its index.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let result = Matrix::with_initializer((2, 3), |index| index.row + index.col);
    /// assert_eq!(result, Ok(matrix![[0, 1, 2], [1, 2, 3]]));
    /// ```
    ///
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn with_initializer<S, F>(shape: S, mut initializer: F) -> Result<Self>
    where
        S: AsShape,
        F: FnMut(Index) -> T,
    {
        let order = Order::default();
        let shape = AxisShape::from_shape(shape, order);
        let stride = shape.stride();
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        for index in 0..size {
            let index = Index::from_flattened(index, order, stride);
            let element = initializer(index);
            data.push(element);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T> Default for Matrix<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::matrix;
    use crate::shape::Shape;
    use crate::testkit;

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
    fn test_with_default() {
        let shape = Shape::new(2, 3);
        let matrix = Matrix::<i32>::with_default(shape).unwrap();
        assert_eq!(matrix.order, Order::default());
        let expected = matrix![[0, 0, 0], [0, 0, 0]];
        testkit::assert_loose_eq(&matrix, &expected);

        let shape = Shape::new(usize::MAX, 2);
        let error = Matrix::<i32>::with_default(shape).unwrap_err();
        assert_eq!(error, Error::SizeOverflow);

        let shape = Shape::new(isize::MAX as usize / 4 + 1, 1);
        let error = Matrix::<i32>::with_default(shape).unwrap_err();
        assert_eq!(error, Error::CapacityOverflow);

        let shape = Shape::new(isize::MAX as usize + 1, 1);
        let error = Matrix::<u8>::with_default(shape).unwrap_err();
        assert_eq!(error, Error::CapacityOverflow);

        // unable to cover
        // let shape = Shape::new(isize::MAX as usize + 1, 1);
        // assert!(Matrix::<()>::with_default(shape).is_ok());
    }

    #[test]
    fn test_with_value() {
        let shape = Shape::new(2, 3);
        let matrix = Matrix::with_value(shape, 0).unwrap();
        assert_eq!(matrix.order, Order::default());
        let expected = matrix![[0, 0, 0], [0, 0, 0]];
        testkit::assert_loose_eq(&matrix, &expected);

        let shape = Shape::new(usize::MAX, 2);
        let error = Matrix::with_value(shape, 0).unwrap_err();
        assert_eq!(error, Error::SizeOverflow);

        let shape = Shape::new(isize::MAX as usize / 4 + 1, 1);
        let error = Matrix::<i32>::with_value(shape, 0).unwrap_err();
        assert_eq!(error, Error::CapacityOverflow);

        let shape = Shape::new(isize::MAX as usize + 1, 1);
        let error = Matrix::<u8>::with_value(shape, 0).unwrap_err();
        assert_eq!(error, Error::CapacityOverflow);

        let shape = Shape::new(isize::MAX as usize + 1, 1);
        assert!(Matrix::<()>::with_value(shape, ()).is_ok());
    }

    #[test]
    fn test_with_initializer() {
        let shape = Shape::new(2, 3);
        let matrix = Matrix::with_initializer(shape, |index| index.row + index.col).unwrap();
        assert_eq!(matrix.order, Order::default());
        let expected = matrix![[0, 1, 2], [1, 2, 3]];
        testkit::assert_loose_eq(&matrix, &expected);

        // assert no panic from unflattening indices occurs
        let shape = Shape::new(2, 0);
        let matrix = Matrix::with_initializer(shape, |index| index.row + index.col).unwrap();
        assert_eq!(matrix.order, Order::default());
        let expected = matrix![[0; 0]; 2];
        testkit::assert_loose_eq(&matrix, &expected);

        // assert no panic from unflattening indices occurs
        let shape = Shape::new(0, 3);
        let matrix = Matrix::with_initializer(shape, |index| index.row + index.col).unwrap();
        assert_eq!(matrix.order, Order::default());
        let expected = matrix![[0; 3]; 0];
        testkit::assert_loose_eq(&matrix, &expected);

        let shape = Shape::new(usize::MAX, 2);
        let error = Matrix::with_initializer(shape, |_| 0).unwrap_err();
        assert_eq!(error, Error::SizeOverflow);

        let shape = Shape::new(isize::MAX as usize / 4 + 1, 1);
        let error = Matrix::<i32>::with_initializer(shape, |_| 0).unwrap_err();
        assert_eq!(error, Error::CapacityOverflow);

        let shape = Shape::new(isize::MAX as usize + 1, 1);
        let error = Matrix::<u8>::with_initializer(shape, |_| 0).unwrap_err();
        assert_eq!(error, Error::CapacityOverflow);

        // unable to cover
        // let shape = Shape::new(isize::MAX as usize + 1, 1);
        // assert!(Matrix::<()>::with_initializer(shape, |_| ()).is_ok());
    }
}
