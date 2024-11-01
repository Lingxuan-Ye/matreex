use super::order::Order;
use super::shape::{AxisShape, Shape};
use super::Matrix;
use crate::error::{Error, Result};

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

    pub fn with_value<S>(shape: S, value: T) -> Result<Self>
    where
        T: Clone,
        S: Shape,
    {
        let order = Order::default();
        let shape = AxisShape::try_from_shape(shape, order)?;
        let size = Self::check_size(shape.size())?;
        let data = vec![value; size];
        Ok(Self { order, shape, data })
    }

    pub fn with_initializer<S, F>(shape: S, initializer: F) -> Result<Self>
    where
        S: Shape,
        F: FnMut() -> T,
    {
        let order = Order::default();
        let shape = AxisShape::try_from_shape(shape, order)?;
        let size = Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(size);
        data.resize_with(size, initializer);
        Ok(Self { order, shape, data })
    }

    /// Creates a new [`Matrix<T>`] with the specified shape, filled with
    /// default values.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityExceeded`] if total bytes stored exceeds [`isize::MAX`].
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
    /// assert_eq!(result, Err(Error::CapacityExceeded));
    /// ```
    pub fn with_default<S>(shape: S) -> Result<Self>
    where
        T: Default,
        S: Shape,
    {
        Self::with_initializer(shape, T::default)
    }
}

/// Methods in this impl block are primarily for internal use.
/// Changes to these methods will not be considered breaking changes.
#[doc(hidden)]
impl<T> Matrix<T> {
    /// Creates a new [`Matrix<T>`] from its component parts, without
    /// checking if the size matches.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the size of `shape` matches the length
    /// of `data`. If the length is greater, extra elements will not be
    /// accessible. If the size is greater, accessing the matrix may result
    /// in out-of-bounds memory access, leading to *[undefined behavior]*.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Matrix, Order};
    ///
    /// let order = Order::default();
    /// let shape = (2, 3);
    /// let data = vec![0, 1, 2, 3, 4, 5];
    /// let result = unsafe { Matrix::from_parts_unchecked(order, shape, data) };
    /// assert_eq!(result, matrix![[0, 1, 2], [3, 4, 5]]);
    /// ```
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn from_parts_unchecked<S>(order: Order, shape: S, data: Vec<T>) -> Self
    where
        S: Shape,
    {
        let shape = AxisShape::from_shape_unchecked(shape, order);
        Self { order, shape, data }
    }

    /// Creates a new [`Matrix<T>`] from its component parts.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeMismatch`] if the size of `shape` does not match
    ///   the length of `data`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error, Matrix, Order};
    ///
    /// let order = Order::default();
    /// let data = vec![0, 1, 2, 3, 4, 5];
    ///
    /// let shape = (2, 3);
    /// let result = Matrix::from_parts(order, shape, data.clone());
    /// assert_eq!(result, Ok(matrix![[0, 1, 2], [3, 4, 5]]));
    ///
    /// let shape = (2, 2);
    /// let result = Matrix::from_parts(order, shape, data.clone());
    /// assert_eq!(result, Err(Error::SizeMismatch));
    /// ```
    pub fn from_parts<S>(order: Order, shape: S, data: Vec<T>) -> Result<Self>
    where
        S: Shape,
    {
        let Ok(size) = shape.size() else {
            return Err(Error::SizeMismatch);
        };
        if data.len() != size {
            return Err(Error::SizeMismatch);
        }
        unsafe { Ok(Self::from_parts_unchecked(order, shape, data)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_with_default() {
        let expected = matrix![[0, 0, 0], [0, 0, 0]];

        assert_eq!(Matrix::with_default((2, 3)).unwrap(), expected);
        assert_ne!(Matrix::with_default((3, 2)).unwrap(), expected);

        assert_eq!(
            Matrix::<u8>::with_default((usize::MAX, 2)).unwrap_err(),
            Error::SizeOverflow
        );
        assert_eq!(
            Matrix::<u8>::with_default((isize::MAX as usize + 1, 1)).unwrap_err(),
            Error::CapacityExceeded
        );

        assert_eq!(
            Matrix::<i32>::with_default((usize::MAX, 2)).unwrap_err(),
            Error::SizeOverflow
        );
        assert_eq!(
            Matrix::<i32>::with_default((isize::MAX as usize / 4 + 1, 1)).unwrap_err(),
            Error::CapacityExceeded
        );

        // The following test cases for zero-sized types are impractical to
        // run in debug mode, and since `#[cfg(not(debug_assertions))]` does
        // not strictly match release mode, these tests are commented out.

        // assert_eq!(
        //     Matrix::<()>::with_default((usize::MAX, 2)).unwrap_err(),
        //     Error::SizeOverflow
        // );
        // assert!(Matrix::<()>::with_default((isize::MAX as usize + 1, 1)).is_ok());

        // #[derive(Debug, Default)]
        // struct Foo;
        // assert_eq!(
        //     Matrix::<Foo>::with_default((usize::MAX, 2)).unwrap_err(),
        //     Error::SizeOverflow
        // );
        // assert!(Matrix::<Foo>::with_default((isize::MAX as usize + 1, 1)).is_ok());
    }

    #[test]
    fn test_from_parts_unchecked() {
        let order = Order::default();
        let shape = (2, 3);
        let data = vec![0, 1, 2, 3, 4, 5];
        let result = unsafe { Matrix::from_parts_unchecked(order, shape, data) };
        assert_eq!(result, matrix![[0, 1, 2], [3, 4, 5]]);
    }

    #[test]
    fn test_from_parts() {
        let order = Order::default();
        let data = vec![0, 1, 2, 3, 4, 5];

        let shape = (2, 3);
        let matrix = Matrix::from_parts(order, shape, data.clone()).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        let shape = (2, 2);
        let error = Matrix::from_parts(order, shape, data.clone()).unwrap_err();
        assert_eq!(error, Error::SizeMismatch);

        let shape = (usize::MAX, 2);
        let error = Matrix::from_parts(order, shape, data.clone()).unwrap_err();
        assert_eq!(error, Error::SizeMismatch);

        let shape = (isize::MAX as usize + 1, 1);
        let error = Matrix::from_parts(order, shape, data.clone()).unwrap_err();
        assert_eq!(error, Error::SizeMismatch);
    }
}