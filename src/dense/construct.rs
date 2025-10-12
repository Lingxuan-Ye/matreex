use super::Matrix;
use super::layout::{Layout, Order};
use crate::error::Result;
use crate::index::Index;
use crate::shape::AsShape;
use alloc::vec;
use alloc::vec::Vec;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Creates a new, empty [`Matrix<T, O>`].
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
        let layout = Layout::default();
        let data = Vec::new();
        Self { layout, data }
    }

    /// Creates a new, empty [`Matrix<T, O>`] with at least the specified capacity.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let matrix = Matrix::<i32>::with_capacity(10)?;
    /// assert_eq!(matrix.nrows(), 0);
    /// assert_eq!(matrix.ncols(), 0);
    /// assert!(matrix.is_empty());
    /// assert!(matrix.capacity() >= 10);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Layout::<T, O>::ensure_can_hold(capacity)?;
        let layout = Layout::default();
        let data = Vec::with_capacity(capacity);
        Ok(Self { layout, data })
    }

    /// Creates a new [`Matrix<T, O>`] with the specified shape, filling with the
    /// default value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size of the shape exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let result = Matrix::with_default((2, 3));
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    /// ```
    pub fn with_default<S>(shape: S) -> Result<Self>
    where
        S: AsShape,
        T: Default,
    {
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.resize_with(size, T::default);
        Ok(Self { layout, data })
    }

    /// Creates a new [`Matrix<T, O>`] with the specified shape, filling with the
    /// given value.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size of the shape exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let result = Matrix::with_value((2, 3), 0);
    /// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
    /// ```
    pub fn with_value<S>(shape: S, value: T) -> Result<Self>
    where
        S: AsShape,
        T: Clone,
    {
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let data = vec![value; size];
        Ok(Self { layout, data })
    }

    /// Creates a new [`Matrix<T, O>`] with the specified shape, filling with values
    /// initialized using their indices.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size of the shape exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Index, Matrix, matrix};
    ///
    /// let result = Matrix::with_initializer((2, 3), |index| index);
    /// assert_eq!(
    ///     result,
    ///     Ok(matrix![
    ///         [Index::new(0, 0), Index::new(0, 1), Index::new(0, 2)],
    ///         [Index::new(1, 0), Index::new(1, 1), Index::new(1, 2)],
    ///     ])
    /// );
    /// ```
    pub fn with_initializer<S, F>(shape: S, mut initializer: F) -> Result<Self>
    where
        S: AsShape,
        F: FnMut(Index) -> T,
    {
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let stride = layout.stride();
        let mut data = Vec::with_capacity(size);
        for index in 0..size {
            let index = Index::from_flattened::<O>(index, stride);
            let element = initializer(index);
            data.push(element);
        }
        Ok(Self { layout, data })
    }
}

impl<T, O> Default for Matrix<T, O>
where
    O: Order,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::index::Index;
    use crate::shape::Shape;
    use crate::{dispatch_unary, matrix};

    #[test]
    fn test_new() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::new();
            assert_eq!(matrix.nrows(), 0);
            assert_eq!(matrix.ncols(), 0);
            assert!(matrix.is_empty());
        }}
    }

    #[test]
    fn test_with_capacity() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::with_capacity(10).unwrap();
            assert_eq!(matrix.nrows(), 0);
            assert_eq!(matrix.ncols(), 0);
            assert!(matrix.is_empty());
            assert!(matrix.capacity() >= 10);
        }}
    }

    #[test]
    fn test_with_default() {
        dispatch_unary! {{
            let shape = Shape::new(2, 3);
            let matrix = Matrix::<i32, O>::with_default(shape).unwrap();
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            assert_eq!(&matrix, &expected);

            let shape = Shape::new(2, usize::MAX);
            let error = Matrix::<i32, O>::with_default(shape).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let shape = Shape::new(1, usize::MAX);
            let error = Matrix::<i32, O>::with_default(shape).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);

            // Unable to cover.
            // let shape = Shape::new(1, usize::MAX);
            // ssert!(Matrix::<(), O>::with_default(shape).is_ok());
        }}
    }

    #[test]
    fn test_with_value() {
        dispatch_unary! {{
            let shape = Shape::new(2, 3);
            let matrix = Matrix::<i32, O>::with_value(shape, 0).unwrap();
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            assert_eq!(&matrix, &expected);

            let shape = Shape::new(2, usize::MAX);
            let error = Matrix::<i32, O>::with_value(shape, 0).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let shape = Shape::new(1, usize::MAX);
            let error = Matrix::<i32, O>::with_value(shape, 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);

            let shape = Shape::new(1, usize::MAX);
            assert!(Matrix::<(), O>::with_value(shape, ()).is_ok());
        }}
    }

    #[test]
    fn test_with_initializer() {
        dispatch_unary! {{
            let shape = Shape::new(2, 3);
            let matrix = Matrix::<Index, O>::with_initializer(shape, |index| index).unwrap();
            let expected = matrix![
                [Index::new(0, 0), Index::new(0, 1), Index::new(0, 2)],
                [Index::new(1, 0), Index::new(1, 1), Index::new(1, 2)],
            ];
            assert_eq!(&matrix, &expected);

            // Assert no panic from unflattening indices occurs.
            let shape = Shape::new(2, 0);
            let matrix = Matrix::<Index, O>::with_initializer(shape, |index| index).unwrap();
            let expected = matrix![[Index::default(); 0]; 2];
            assert_eq!(&matrix, &expected);

            // Assert no panic from unflattening indices occurs.
            let shape = Shape::new(0, 3);
            let matrix = Matrix::<Index, O>::with_initializer(shape, |index| index).unwrap();
            let expected = matrix![[Index::default(); 3]; 0];
            assert_eq!(&matrix, &expected);

            let shape = Shape::new(2, usize::MAX);
            let error = Matrix::<i32, O>::with_initializer(shape, |_| 0).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let shape = Shape::new(1, usize::MAX);
            let error = Matrix::<i32, O>::with_initializer(shape, |_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);

            // Unable to cover.
            // let shape = Shape::new(1, usize::MAX);
            // assert!(Matrix::<(), O>::with_initializer(shape, |_| ()).is_ok());
        }}
    }

    #[test]
    fn test_default() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::default();
            assert_eq!(matrix.nrows(), 0);
            assert_eq!(matrix.ncols(), 0);
            assert!(matrix.is_empty());
        }}
    }
}
