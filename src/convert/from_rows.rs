use crate::Matrix;
use crate::error::{Error, Result};
use crate::order::Order;
use crate::shape::{AxisShape, Shape};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A trait for conversion from a sequence of rows.
pub trait FromRows<T>: Sized {
    /// Converts from a sequence of rows.
    fn from_rows(value: T) -> Self;
}

/// A trait for fallible conversion from a sequence of rows.
pub trait TryFromRows<T>: Sized {
    /// Attempts to convert from a sequence of rows.
    fn try_from_rows(value: T) -> Result<Self>;
}

/// A trait for conversion from an iterator over rows.
pub trait FromRowIterator<T, V>: Sized
where
    V: IntoIterator<Item = T>,
{
    /// Converts from an iterator over rows.
    fn from_row_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>;
}

impl<T, const R: usize, const C: usize> FromRows<[[T; C]; R]> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from_rows(rows);
    /// // this is actually a circular validation
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from_rows(value: [[T; C]; R]) -> Self {
        let order = Order::RowMajor;
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const R: usize, const C: usize> FromRows<[Box<[T; C]>; R]> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let matrix = Matrix::from_rows(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from_rows(value: [Box<[T; C]>; R]) -> Self {
        let order = Order::RowMajor;
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flat_map(|row| *row).collect();
        Self { order, shape, data }
    }
}

impl<T, const R: usize, const C: usize> FromRows<Box<[[T; C]; R]>> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
    /// let matrix = Matrix::from_rows(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    #[inline]
    fn from_rows(value: Box<[[T; C]; R]>) -> Self {
        Self::from_rows(*value)
    }
}

impl<T, const R: usize, const C: usize> FromRows<Box<[Box<[T; C]>; R]>> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[Box<[i32; 3]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let matrix = Matrix::from_rows(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    #[inline]
    fn from_rows(value: Box<[Box<[T; C]>; R]>) -> Self {
        Self::from_rows(*value)
    }
}

impl<T, const C: usize> FromRows<Box<[[T; C]]>> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
    /// let matrix = Matrix::from_rows(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from_rows(value: Box<[[T; C]]>) -> Self {
        let order = Order::RowMajor;
        let nrows = value.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> FromRows<Box<[Box<[T; C]>]>> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let matrix = Matrix::from_rows(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from_rows(value: Box<[Box<[T; C]>]>) -> Self {
        let order = Order::RowMajor;
        let nrows = value.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flat_map(|row| *row).collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> FromRows<Vec<[T; C]>> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from_rows(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    #[inline]
    fn from_rows(value: Vec<[T; C]>) -> Self {
        Self::from_rows(value.into_boxed_slice())
    }
}

impl<T, const C: usize> FromRows<Vec<Box<[T; C]>>> for Matrix<T> {
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
    /// use matreex::convert::FromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let matrix = Matrix::from_rows(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    #[inline]
    fn from_rows(value: Vec<Box<[T; C]>>) -> Self {
        Self::from_rows(value.into_boxed_slice())
    }
}

impl<T, U> TryFromRows<T> for Matrix<U>
where
    Matrix<U>: FromRows<T>,
{
    #[inline]
    fn try_from_rows(value: T) -> Result<Self> {
        Ok(Self::from_rows(value))
    }
}

impl<T, const R: usize> TryFromRows<[Box<[T]>; R]> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    fn try_from_rows(value: [Box<[T]>; R]) -> Result<Self> {
        let order = Order::RowMajor;
        if R == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let nrows = R;
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T, const R: usize> TryFromRows<[Vec<T>; R]> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    fn try_from_rows(value: [Vec<T>; R]) -> Result<Self> {
        let order = Order::RowMajor;
        if R == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let nrows = R;
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T, const R: usize> TryFromRows<Box<[Box<[T]>; R]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    #[inline]
    fn try_from_rows(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(*value)
    }
}

impl<T, const R: usize> TryFromRows<Box<[Vec<T>; R]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    #[inline]
    fn try_from_rows(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(*value)
    }
}

impl<T> TryFromRows<Box<[Box<[T]>]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    fn try_from_rows(value: Box<[Box<[T]>]>) -> Result<Self> {
        let order = Order::RowMajor;
        let nrows = value.len();
        if nrows == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T> TryFromRows<Box<[Vec<T>]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    fn try_from_rows(value: Box<[Vec<T>]>) -> Result<Self> {
        let order = Order::RowMajor;
        let nrows = value.len();
        if nrows == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T> TryFromRows<Vec<Box<[T]>>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    #[inline]
    fn try_from_rows(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_rows(value.into_boxed_slice())
    }
}

impl<T> TryFromRows<Vec<Vec<T>>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of rows.
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
    /// use matreex::convert::TryFromRows;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
    /// let result = Matrix::try_from_rows(rows);
    /// assert_eq!(result, Ok(matrix![[1, 2, 3], [4, 5, 6]]));
    /// ```
    #[inline]
    fn try_from_rows(value: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_rows(value.into_boxed_slice())
    }
}

impl<T, V> FromRowIterator<T, V> for Matrix<T>
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
    /// use matreex::convert::FromRowIterator;
    /// use matreex::{Matrix, matrix};
    ///
    /// let rows = [[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from_row_iter(rows);
    /// assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
    /// ```
    fn from_row_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        let order = Order::RowMajor;
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Self { order, shape, data };
        };
        // could panic if capacity overflows
        let mut data: Vec<T> = first.into_iter().collect();
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
    fn test_from_rows() {
        let expected = {
            let order = Order::RowMajor;
            let shape = Shape::new(2, 3);
            let shape = AxisShape::from_shape(shape, order);
            let data = vec![1, 2, 3, 4, 5, 6];
            Matrix { order, shape, data }
        };

        let rows: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let rows: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let rows: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let rows: Box<[Box<[i32; 3]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let rows: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let rows: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let rows: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);

        let rows: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
        let matrix = Matrix::from_rows(rows);
        testkit::assert_strict_eq(&matrix, &expected);
    }

    #[test]
    fn test_try_from_rows() {
        const MAX: usize = isize::MAX as usize;

        {
            let expected = {
                let order = Order::RowMajor;
                let shape = Shape::new(2, 3);
                let shape = AxisShape::from_shape(shape, order);
                let data = vec![1, 2, 3, 4, 5, 6];
                Matrix { order, shape, data }
            };

            let rows: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let rows: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let rows: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let rows: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let rows: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let rows: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let rows: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let rows: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let matrix = Matrix::try_from_rows(rows).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);
        }

        {
            let rows: [Box<[()]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::try_from_rows(rows).is_ok());

            let rows: [Vec<()>; 2] = [vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::try_from_rows(rows).is_ok());

            let rows: Box<[Box<[()]>; 2]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::try_from_rows(rows).is_ok());

            let rows: Box<[Vec<()>; 2]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::try_from_rows(rows).is_ok());

            let rows: Box<[Box<[()]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::try_from_rows(rows).is_ok());

            let rows: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::try_from_rows(rows).is_ok());

            let rows: Vec<Box<[()]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::try_from_rows(rows).is_ok());

            let rows: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::try_from_rows(rows).is_ok());
        }

        {
            let rows: [Box<[()]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let rows: [Vec<()>; 3] = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let rows: Box<[Box<[()]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let rows: Box<[Vec<()>; 3]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let rows: Box<[Box<[()]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let rows: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let rows: Vec<Box<[()]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let rows: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
        }

        // unable to cover (run out of memory)
        // {
        //     let rows: [Box<[u8]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let rows: [Vec<u8>; 2] = [vec![0; MAX], vec![0; MAX]];
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let rows: Box<[Box<[u8]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let rows: Box<[Vec<u8>; 2]> = Box::new([vec![0; MAX], vec![0; MAX]]);
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let rows: Box<[Box<[u8]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let rows: Box<[Vec<u8>]> = Box::new([vec![0; MAX], vec![0; MAX]]);
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let rows: Vec<Box<[u8]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let rows: Vec<Vec<u8>> = vec![vec![0; MAX], vec![0; MAX]];
        //     let error = Matrix::try_from_rows(rows).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);
        // }

        {
            let rows: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let rows: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let rows: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let rows: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let rows: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let rows: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let rows: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let rows: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::try_from_rows(rows).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);
        }
    }

    #[test]
    fn test_from_row_iter() {
        let expected = {
            let order = Order::RowMajor;
            let shape = Shape::new(2, 3);
            let shape = AxisShape::from_shape(shape, order);
            let data = vec![1, 2, 3, 4, 5, 6];
            Matrix { order, shape, data }
        };

        let rows = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_row_iter(rows);
        testkit::assert_strict_eq(&matrix, &expected);
    }

    #[test]
    #[should_panic]
    fn test_from_row_iter_fails() {
        let rows = [vec![1, 2, 3], vec![4, 5]];
        Matrix::from_row_iter(rows);
    }
}
