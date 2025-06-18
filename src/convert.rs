use crate::Matrix;
use crate::error::{Error, Result};
use crate::order::Order;
use crate::shape::{AxisShape, Shape};
use alloc::vec::Vec;

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
        let shape = Shape::new(R, C);
        let shape = AxisShape::from_shape(shape, order);
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
        let shape = Shape::new(nrows, C);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> From<&[[T; C]]> for Matrix<T>
where
    T: Clone,
{
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
        let shape = Shape::new(nrows, C);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.iter().flatten().cloned().collect();
        Self { order, shape, data }
    }
}

impl<T, const R: usize> TryFrom<[Vec<T>; R]> for Matrix<T> {
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
    fn try_from(value: [Vec<T>; R]) -> Result<Self> {
        let order = Order::RowMajor;
        let ncols = match value.first() {
            Some(row) => row.len(),
            None => 0,
        };
        let shape = Shape::new(R, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
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
        let ncols = match value.first() {
            Some(row) => row.len(),
            None => 0,
        };
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
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

impl<T> TryFrom<&[Vec<T>]> for Matrix<T>
where
    T: Clone,
{
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
        let ncols = match value.first() {
            Some(row) => row.len(),
            None => 0,
        };
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
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
    /// Panics if rows have inconsistent lengths or capacity overflows.
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
    /// let matrix = Matrix::from_iter(rows);
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
        // could panic if capacity overflows
        let mut data: Vec<T> = row.into_iter().collect();
        let mut nrows = 1;
        let ncols = data.len();
        let mut size = ncols;
        for row in iter {
            // could panic if capacity overflows
            data.extend(row);
            if data.len() - size != ncols {
                panic!("{}", Error::LengthInconsistent);
            }
            nrows += 1;
            size = data.len();
        }
        data.shrink_to_fit();
        let order = Order::RowMajor;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        Self { order, shape, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit;
    use alloc::vec;

    #[test]
    fn test_from() {
        let rows = [[1, 2, 3], [4, 5, 6]];
        let expected = {
            let order = Order::RowMajor;
            let shape = Shape::new(2, 3);
            let shape = AxisShape::from_shape(shape, order);
            let data = vec![1, 2, 3, 4, 5, 6];
            Matrix { order, shape, data }
        };

        let matrix = Matrix::from(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let matrix = Matrix::from(rows.to_vec());
        testkit::assert_strict_eq(&matrix, &expected);

        let matrix = Matrix::from(&rows[..]);
        testkit::assert_strict_eq(&matrix, &expected);
    }

    #[test]
    fn test_try_from() {
        const MAX: usize = isize::MAX as usize;

        {
            let rows = [vec![1, 2, 3], vec![4, 5, 6]];
            let expected = {
                let order = Order::RowMajor;
                let shape = Shape::new(2, 3);
                let shape = AxisShape::from_shape(shape, order);
                let data = vec![1, 2, 3, 4, 5, 6];
                Matrix { order, shape, data }
            };

            let matrix = Matrix::try_from(rows.clone()).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let matrix = Matrix::try_from(rows.to_vec()).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let matrix = Matrix::try_from(&rows[..]).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);
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
        let rows = [[1, 2, 3], [4, 5, 6]];
        let expected = {
            let order = Order::RowMajor;
            let shape = Shape::new(2, 3);
            let shape = AxisShape::from_shape(shape, order);
            let data = vec![1, 2, 3, 4, 5, 6];
            Matrix { order, shape, data }
        };

        let matrix = Matrix::from_iter(rows);
        testkit::assert_strict_eq(&matrix, &expected);
    }

    #[test]
    #[should_panic]
    fn test_from_iter_fails() {
        Matrix::from_iter([vec![1, 2, 3], vec![4, 5]]);
    }
}
