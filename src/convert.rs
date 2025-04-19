use crate::Matrix;
use crate::error::{Error, Result};
use crate::order::Order;
use crate::shape::Shape;
use alloc::vec::Vec;

impl<T> Matrix<T> {
    /// Creates a new single-row [`Matrix<T>`] from a vector.
    ///
    /// # Examples
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let row_vec = Matrix::from_row([1, 2, 3]);
    /// assert_eq!(row_vec, matrix![[1, 2, 3]]);
    /// ```
    pub fn from_row<R>(row: R) -> Self
    where
        R: Into<Vec<T>>,
    {
        let data = row.into();
        let order = Order::default();
        let shape = Shape::new(1, data.len()).to_axis_shape_unchecked(order);
        Self { order, shape, data }
    }

    /// Creates a new single-column [`Matrix<T>`] from a vector.
    ///
    /// # Examples
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let col_vec = Matrix::from_col([1, 2, 3]);
    /// assert_eq!(col_vec, matrix![[1], [2], [3]]);
    /// ```
    pub fn from_col<C>(col: C) -> Self
    where
        C: Into<Vec<T>>,
    {
        let data = col.into();
        let order = Order::default();
        let shape = Shape::new(data.len(), 1).to_axis_shape_unchecked(order);
        Self { order, shape, data }
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    /// Converts to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::RowMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows = [[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from(rows);
    /// // this is actually a circular validation
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from(value: [[T; C]; R]) -> Self {
        let order = Order::RowMajor;
        let shape = Shape::new(R, C).to_axis_shape_unchecked(order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> From<Vec<[T; C]>> for Matrix<T> {
    /// Converts to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::RowMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows = vec![[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from(value: Vec<[T; C]>) -> Self {
        let order = Order::RowMajor;
        let nrows = value.len();
        let shape = Shape::new(nrows, C).to_axis_shape_unchecked(order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T: Clone, const C: usize> From<&[[T; C]]> for Matrix<T> {
    /// Converts to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::RowMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows = &[[1, 2, 3], [4, 5, 6]][..];
    /// let matrix = Matrix::from(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from(value: &[[T; C]]) -> Self {
        let order = Order::RowMajor;
        let nrows = value.len();
        let shape = Shape::new(nrows, C).to_axis_shape_unchecked(order);
        let data = value.iter().flatten().cloned().collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> TryFrom<[Vec<T>; C]> for Matrix<T> {
    type Error = Error;

    /// Converts to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::RowMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows = [vec![1, 2, 3], vec![4, 5, 6]];
    /// let matrix = Matrix::try_from(rows);
    /// assert_eq!(matrix, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    fn try_from(value: [Vec<T>; C]) -> Result<Self> {
        let order = Order::RowMajor;
        let nrows = C;
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
        let size = Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(size);
        for row in value {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = Error;

    /// Converts to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::RowMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows = vec![vec![1, 2, 3], vec![4, 5, 6]];
    /// let matrix = Matrix::try_from(rows);
    /// assert_eq!(matrix, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    fn try_from(value: Vec<Vec<T>>) -> Result<Self> {
        let order = Order::RowMajor;
        let nrows = value.len();
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
        let size = Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(size);
        for row in value {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T: Clone> TryFrom<&[Vec<T>]> for Matrix<T> {
    type Error = Error;

    /// Converts to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::RowMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows = &[vec![1, 2, 3], vec![4, 5, 6]][..];
    /// let matrix = Matrix::try_from(rows);
    /// assert_eq!(matrix, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    fn try_from(value: &[Vec<T>]) -> Result<Self> {
        let order = Order::RowMajor;
        let nrows = value.len();
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
        let size = Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(size);
        for row in value {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend_from_slice(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T, V> FromIterator<V> for Matrix<T>
where
    V: IntoIterator<Item = T>,
{
    /// Converts to [`Matrix<T>`] from an iterator over rows.
    ///
    /// # Panics
    ///
    /// Panics if rows have inconsistent lengths or if memory allocation fails.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::RowMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Matrix, matrix};
    ///
    /// let iter = [[1, 2, 3], [4, 5, 6]].into_iter();
    /// let matrix = Matrix::from_iter(iter);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        let mut iter = iter.into_iter();
        let Some(row) = iter.next() else {
            return Self::new();
        };
        let mut data: Vec<T> = row.into_iter().collect(); // could panic for running out of memory
        let mut nrows = 1;
        let ncols = data.len();
        let mut size = ncols;
        for row in iter {
            data.extend(row); // could panic for running out of memory
            if data.len() - size != ncols {
                panic!("{}", Error::LengthInconsistent);
            }
            nrows += 1;
            size = data.len();
        }
        data.shrink_to_fit();
        let order = Order::RowMajor;
        let shape = Shape::new(nrows, ncols).to_axis_shape_unchecked(order);
        Self { order, shape, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // tests in this module should avoid direct comparison of `Matrix<T>`

    #[test]
    fn test_from_row() {
        let expected_order = Order::default();
        let expected_shape = Shape::new(1, 3).to_axis_shape_unchecked(expected_order);
        let expected_data = vec![1, 2, 3];

        let row_vec = Matrix::from_row([1, 2, 3]);
        assert_eq!(row_vec.order, expected_order);
        assert_eq!(row_vec.shape, expected_shape);
        assert_eq!(row_vec.data, expected_data);
    }

    #[test]
    fn test_from_col() {
        let expected_order = Order::default();
        let expected_shape = Shape::new(3, 1).to_axis_shape_unchecked(expected_order);
        let expected_data = vec![1, 2, 3];

        let col_vec = Matrix::from_col([1, 2, 3]);
        assert_eq!(col_vec.order, expected_order);
        assert_eq!(col_vec.shape, expected_shape);
        assert_eq!(col_vec.data, expected_data);
    }

    #[test]
    fn test_from() {
        let expected_order = Order::RowMajor;
        let expected_shape = Shape::new(2, 3).to_axis_shape_unchecked(expected_order);
        let expected_data = vec![1, 2, 3, 4, 5, 6];

        let rows = [[1, 2, 3], [4, 5, 6]];

        let matrix = Matrix::from(rows);
        assert_eq!(matrix.order, expected_order);
        assert_eq!(matrix.shape, expected_shape);
        assert_eq!(matrix.data, expected_data);

        let matrix = Matrix::from(rows.to_vec());
        assert_eq!(matrix.order, expected_order);
        assert_eq!(matrix.shape, expected_shape);
        assert_eq!(matrix.data, expected_data);

        let matrix = Matrix::from(&rows[..]);
        assert_eq!(matrix.order, expected_order);
        assert_eq!(matrix.shape, expected_shape);
        assert_eq!(matrix.data, expected_data);
    }

    #[test]
    fn test_try_from() {
        const MAX: usize = isize::MAX as usize;

        {
            let expected_order = Order::RowMajor;
            let expected_shape = Shape::new(2, 3).to_axis_shape_unchecked(expected_order);
            let expected_data = vec![1, 2, 3, 4, 5, 6];

            let rows = [vec![1, 2, 3], vec![4, 5, 6]];

            let matrix = Matrix::try_from(rows.clone()).unwrap();
            assert_eq!(matrix.order, expected_order);
            assert_eq!(matrix.shape, expected_shape);
            assert_eq!(matrix.data, expected_data);

            let matrix = Matrix::try_from(rows.to_vec()).unwrap();
            assert_eq!(matrix.order, expected_order);
            assert_eq!(matrix.shape, expected_shape);
            assert_eq!(matrix.data, expected_data);

            let matrix = Matrix::try_from(&rows[..]).unwrap();
            assert_eq!(matrix.order, expected_order);
            assert_eq!(matrix.shape, expected_shape);
            assert_eq!(matrix.data, expected_data);
        }

        {
            let rows = [vec![(); MAX], vec![(); MAX]];

            assert!(Matrix::try_from(rows.clone()).is_ok());
            assert!(Matrix::try_from(rows.to_vec()).is_ok());
            assert!(Matrix::try_from(&rows[..]).is_ok());
        }

        {
            let rows = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];

            let error = Matrix::try_from(rows.clone()).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let error = Matrix::try_from(rows.to_vec()).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let error = Matrix::try_from(&rows[..]).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
        }

        // unable to cover (run out of memory)
        // {
        //     let rows = [vec![0u8; MAX], vec![0u8; MAX]];
        //
        //     let error = Matrix::try_from(rows.clone()).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);
        //
        //     let error = Matrix::try_from(rows.to_vec()).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);
        //
        //     let error = Matrix::try_from(&rows[..]).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);
        // }

        {
            let rows = [vec![1, 2, 3], vec![4, 5]];

            let error = Matrix::try_from(rows.clone()).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let error = Matrix::try_from(rows.to_vec()).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let error = Matrix::try_from(&rows[..]).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);
        }
    }

    #[test]
    fn test_from_iter() {
        let expected_order = Order::RowMajor;
        let expected_shape = Shape::new(2, 3).to_axis_shape_unchecked(expected_order);
        let expected_data = vec![1, 2, 3, 4, 5, 6];

        let iterable = [[1, 2, 3], [4, 5, 6]];

        let matrix = Matrix::from_iter(iterable);
        assert_eq!(matrix.order, expected_order);
        assert_eq!(matrix.shape, expected_shape);
        assert_eq!(matrix.data, expected_data);
    }

    #[test]
    #[should_panic]
    fn test_from_iter_fails() {
        Matrix::from_iter([vec![1, 2, 3], vec![4, 5]]);
    }
}
