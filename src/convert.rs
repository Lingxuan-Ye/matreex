pub use self::from_cols::{FromColIterator, FromCols, TryFromCols};
pub use self::from_rows::{FromRowIterator, FromRows, TryFromRows};

use crate::Matrix;
use crate::error::{Error, Result};
use alloc::boxed::Box;
use alloc::vec::Vec;

mod from_cols;
mod from_rows;

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
    #[inline]
    fn from(value: [[T; C]; R]) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const R: usize, const C: usize> From<[Box<[T; C]>; R]> for Matrix<T> {
    #[inline]
    fn from(value: [Box<[T; C]>; R]) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const R: usize, const C: usize> From<Box<[[T; C]; R]>> for Matrix<T> {
    #[inline]
    fn from(value: Box<[[T; C]; R]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const R: usize, const C: usize> From<Box<[Box<[T; C]>; R]>> for Matrix<T> {
    #[inline]
    fn from(value: Box<[Box<[T; C]>; R]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const C: usize> From<Box<[[T; C]]>> for Matrix<T> {
    #[inline]
    fn from(value: Box<[[T; C]]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const C: usize> From<Box<[Box<[T; C]>]>> for Matrix<T> {
    #[inline]
    fn from(value: Box<[Box<[T; C]>]>) -> Self {
        Self::from_rows(value)
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
    #[inline]
    fn from(value: Vec<[T; C]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const C: usize> From<Vec<Box<[T; C]>>> for Matrix<T> {
    #[inline]
    fn from(value: Vec<Box<[T; C]>>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const R: usize> TryFrom<[Box<[T]>; R]> for Matrix<T> {
    type Error = Error;

    #[inline]
    fn try_from(value: [Box<[T]>; R]) -> Result<Self> {
        Self::try_from_rows(value)
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
    #[inline]
    fn try_from(value: [Vec<T>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const R: usize> TryFrom<Box<[Box<[T]>; R]>> for Matrix<T> {
    type Error = Error;

    #[inline]
    fn try_from(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const R: usize> TryFrom<Box<[Vec<T>; R]>> for Matrix<T> {
    type Error = Error;

    #[inline]
    fn try_from(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T> TryFrom<Box<[Box<[T]>]>> for Matrix<T> {
    type Error = Error;

    #[inline]
    fn try_from(value: Box<[Box<[T]>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T> TryFrom<Box<[Vec<T>]>> for Matrix<T> {
    type Error = Error;

    #[inline]
    fn try_from(value: Box<[Vec<T>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T> TryFrom<Vec<Box<[T]>>> for Matrix<T> {
    type Error = Error;

    #[inline]
    fn try_from(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_rows(value)
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
    #[inline]
    fn try_from(value: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_rows(value)
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
    #[inline]
    fn from_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        Self::from_row_iter(iter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order::Order;
    use crate::shape::{AxisShape, Shape};
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
        }

        {
            let rows = [vec![(); MAX], vec![(); MAX]];

            assert!(Matrix::try_from(rows.clone()).is_ok());
            assert!(Matrix::try_from(rows.to_vec()).is_ok());
        }

        {
            let rows = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];

            let error = Matrix::try_from(rows.clone()).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let error = Matrix::try_from(rows.to_vec()).unwrap_err();
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
        // }

        {
            let rows = [vec![1, 2, 3], vec![4, 5]];

            let error = Matrix::try_from(rows.clone()).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let error = Matrix::try_from(rows.to_vec()).unwrap_err();
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
