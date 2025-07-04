//! Provides and implements traits for matrix conversions.

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
    /// let rows: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from(rows);
    /// // This is actually a circular validation.
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn from(value: [[T; C]; R]) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const R: usize, const C: usize> From<Box<[[T; C]; R]>> for Matrix<T> {
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
    /// let rows: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
    /// let matrix = Matrix::from(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn from(value: Box<[[T; C]; R]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const C: usize> From<Box<[[T; C]]>> for Matrix<T> {
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
    /// let rows: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
    /// let matrix = Matrix::from(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn from(value: Box<[[T; C]]>) -> Self {
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
    /// let rows: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn from(value: Vec<[T; C]>) -> Self {
        Self::from_rows(value)
    }
}

impl<T, const R: usize, const C: usize> TryFrom<[Box<[T; C]>; R]> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: [Box<[T; C]>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const R: usize, const C: usize> TryFrom<Box<[Box<[T; C]>; R]>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Box<[Box<[i32; 3]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Box<[Box<[T; C]>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const C: usize> TryFrom<Box<[Box<[T; C]>]>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Box<[Box<[T; C]>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const C: usize> TryFrom<Vec<Box<[T; C]>>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Vec<Box<[T; C]>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const R: usize> TryFrom<[Box<[T]>; R]> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: [Box<[T]>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const R: usize> TryFrom<Box<[Box<[T]>; R]>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T> TryFrom<Box<[Box<[T]>]>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Box<[Box<[T]>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T> TryFrom<Vec<Box<[T]>>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const R: usize> TryFrom<[Vec<T>; R]> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: [Vec<T>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, const R: usize> TryFrom<Box<[Vec<T>; R]>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T> TryFrom<Box<[Vec<T>]>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn try_from(value: Box<[Vec<T>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = Error;

    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
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
    /// let rows: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
    /// let result = Matrix::try_from(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
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
    ///
    /// [`Order::RowMajor`]: crate::order::Order::RowMajor
    #[inline]
    fn from_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        Self::from_row_iter(iter)
    }
}
