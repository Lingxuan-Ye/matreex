use crate::Matrix;
use crate::error::{Error, Result};
use crate::order::Order;
use crate::shape::{AxisShape, Shape};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A trait for conversion from a sequence of columns.
pub trait FromCols<T>: Sized {
    /// Converts from a sequence of columns.
    fn from_cols(value: T) -> Self;
}

/// A trait for fallible conversion from a sequence of columns.
pub trait TryFromCols<T>: Sized {
    /// Attempts to convert from a sequence of columns.
    fn try_from_cols(value: T) -> Result<Self>;
}

/// A trait for conversion from an iterator over columns.
pub trait FromColIterator<T, V>: Sized
where
    V: IntoIterator<Item = T>,
{
    /// Converts from an iterator over columns.
    fn from_col_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>;
}

impl<T, const R: usize, const C: usize> FromCols<[[T; R]; C]> for Matrix<T> {
    /// Converts to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::FromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from_cols(cols);
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    /// ```
    fn from_cols(value: [[T; R]; C]) -> Self {
        let order = Order::ColMajor;
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const R: usize, const C: usize> FromCols<Box<[[T; R]; C]>> for Matrix<T> {
    /// Converts to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::FromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
    /// let matrix = Matrix::from_cols(cols);
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    /// ```
    #[inline]
    fn from_cols(value: Box<[[T; R]; C]>) -> Self {
        Self::from_cols(value as Box<[[T; R]]>)
    }
}

impl<T, const R: usize> FromCols<Box<[[T; R]]>> for Matrix<T> {
    /// Converts to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::FromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
    /// let matrix = Matrix::from_cols(cols);
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    /// ```
    fn from_cols(value: Box<[[T; R]]>) -> Self {
        let order = Order::ColMajor;
        let nrows = R;
        let ncols = value.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const R: usize> FromCols<Vec<[T; R]>> for Matrix<T> {
    /// Converts to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::FromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from_cols(cols);
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    /// ```
    #[inline]
    fn from_cols(value: Vec<[T; R]>) -> Self {
        Self::from_cols(value.into_boxed_slice())
    }
}

impl<T, U> TryFromCols<T> for Matrix<U>
where
    Matrix<U>: FromCols<T>,
{
    #[inline]
    fn try_from_cols(value: T) -> Result<Self> {
        Ok(Self::from_cols(value))
    }
}

impl<T, const R: usize, const C: usize> TryFromCols<[Box<[T; R]>; C]> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    fn try_from_cols(value: [Box<[T; R]>; C]) -> Result<Self> {
        let order = Order::ColMajor;
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        shape.size::<T>()?;
        let data = value.into_iter().flat_map(|col| col as Box<[T]>).collect();
        Ok(Self { order, shape, data })
    }
}

impl<T, const R: usize, const C: usize> TryFromCols<Box<[Box<[T; R]>; C]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[Box<[i32; 3]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    #[inline]
    fn try_from_cols(value: Box<[Box<[T; R]>; C]>) -> Result<Self> {
        Self::try_from_cols(value as Box<[Box<[T; R]>]>)
    }
}

impl<T, const R: usize> TryFromCols<Box<[Box<[T; R]>]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    fn try_from_cols(value: Box<[Box<[T; R]>]>) -> Result<Self> {
        let order = Order::ColMajor;
        let nrows = R;
        let ncols = value.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        shape.size::<T>()?;
        let data = value.into_iter().flat_map(|col| col as Box<[T]>).collect();
        Ok(Self { order, shape, data })
    }
}

impl<T, const R: usize> TryFromCols<Vec<Box<[T; R]>>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    #[inline]
    fn try_from_cols(value: Vec<Box<[T; R]>>) -> Result<Self> {
        Self::try_from_cols(value.into_boxed_slice())
    }
}

impl<T, const C: usize> TryFromCols<[Box<[T]>; C]> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    fn try_from_cols(value: [Box<[T]>; C]) -> Result<Self> {
        let order = Order::ColMajor;
        if C == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let nrows = first.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T, const C: usize> TryFromCols<Box<[Box<[T]>; C]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    #[inline]
    fn try_from_cols(value: Box<[Box<[T]>; C]>) -> Result<Self> {
        Self::try_from_cols(value as Box<[Box<[T]>]>)
    }
}

impl<T> TryFromCols<Box<[Box<[T]>]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    fn try_from_cols(value: Box<[Box<[T]>]>) -> Result<Self> {
        let order = Order::ColMajor;
        let ncols = value.len();
        if ncols == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let nrows = first.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T> TryFromCols<Vec<Box<[T]>>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    #[inline]
    fn try_from_cols(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_cols(value.into_boxed_slice())
    }
}

impl<T, const C: usize> TryFromCols<[Vec<T>; C]> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    fn try_from_cols(value: [Vec<T>; C]) -> Result<Self> {
        let order = Order::ColMajor;
        if C == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let nrows = first.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T, const C: usize> TryFromCols<Box<[Vec<T>; C]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    #[inline]
    fn try_from_cols(value: Box<[Vec<T>; C]>) -> Result<Self> {
        Self::try_from_cols(value as Box<[Vec<T>]>)
    }
}

impl<T> TryFromCols<Box<[Vec<T>]>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    fn try_from_cols(value: Box<[Vec<T>]>) -> Result<Self> {
        let order = Order::ColMajor;
        let ncols = value.len();
        if ncols == 0 {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Ok(Self { order, shape, data });
        }
        let mut iter = value.into_iter();
        let first = unsafe { iter.next().unwrap_unchecked() };
        let nrows = first.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<T>()?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T> TryFromCols<Vec<Vec<T>>> for Matrix<T> {
    /// Attempts to convert to [`Matrix<T>`] from a sequence of columns.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if columns have inconsistent lengths.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::TryFromCols;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
    /// let result = Matrix::try_from_cols(cols);
    /// assert_eq!(result, Ok(matrix![[1, 4], [2, 5], [3, 6]]));
    /// ```
    #[inline]
    fn try_from_cols(value: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_cols(value.into_boxed_slice())
    }
}

impl<T, V> FromColIterator<T, V> for Matrix<T>
where
    V: IntoIterator<Item = T>,
{
    /// Converts to [`Matrix<T>`] from an iterator over columns.
    ///
    /// # Panics
    ///
    /// Panics if columns have inconsistent lengths or capacity overflows.
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be [`Order::ColMajor`],
    /// regardless of the default.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::convert::FromColIterator;
    /// use matreex::{Matrix, matrix};
    ///
    /// let cols = [[1, 2, 3], [4, 5, 6]];
    /// let matrix = Matrix::from_col_iter(cols);
    /// assert_eq!(matrix, matrix![[1, 4], [2, 5], [3, 6]]);
    /// ```
    fn from_col_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        let order = Order::ColMajor;
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else {
            let shape = AxisShape::default();
            let data = Vec::new();
            return Self { order, shape, data };
        };
        // could panic if capacity overflows
        let mut data: Vec<T> = first.into_iter().collect();
        let mut ncols = 1;
        let nrows = data.len();
        let mut size = nrows;
        for col in iter {
            // could panic if capacity overflows
            data.extend(col);
            if data.len() - size != nrows {
                panic!("{}", Error::LengthInconsistent);
            }
            ncols += 1;
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
    fn test_from_cols() {
        let expected = {
            let order = Order::ColMajor;
            let shape = Shape::new(3, 2);
            let shape = AxisShape::from_shape(shape, order);
            let data = vec![1, 2, 3, 4, 5, 6];
            Matrix { order, shape, data }
        };

        let cols: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_cols(cols);
        testkit::assert_strict_eq(&matrix, &expected);

        let cols: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
        let matrix = Matrix::from_cols(cols);
        testkit::assert_strict_eq(&matrix, &expected);

        let cols: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
        let matrix = Matrix::from_cols(cols);
        testkit::assert_strict_eq(&matrix, &expected);

        let cols: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_cols(cols);
        testkit::assert_strict_eq(&matrix, &expected);
    }

    #[test]
    fn test_try_from_cols() {
        const MAX: usize = isize::MAX as usize;

        {
            let expected = {
                let order = Order::ColMajor;
                let shape = Shape::new(3, 2);
                let shape = AxisShape::from_shape(shape, order);
                let data = vec![1, 2, 3, 4, 5, 6];
                Matrix { order, shape, data }
            };

            let cols: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Box<[Box<[i32; 3]>; 2]> =
                Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);

            let cols: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let matrix = Matrix::try_from_cols(cols).unwrap();
            testkit::assert_strict_eq(&matrix, &expected);
        }

        {
            // unable to cover
            // let cols: [Box<[(); MAX]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            // assert!(Matrix::try_from_cols(cols).is_ok());

            // let cols: Box<[Box<[(); MAX]>; 2]> =
            //     Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            // assert!(Matrix::try_from_cols(cols).is_ok());

            // let cols: Box<[Box<[(); MAX]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            // assert!(Matrix::try_from_cols(cols).is_ok());

            // let cols: Vec<Box<[(); MAX]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            // assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: [Box<[()]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: Box<[Box<[()]>; 2]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: Box<[Box<[()]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: Vec<Box<[()]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: [Vec<()>; 2] = [vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: Box<[Vec<()>; 2]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::try_from_cols(cols).is_ok());

            let cols: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::try_from_cols(cols).is_ok());
        }

        {
            let cols: [Box<[(); MAX]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Box<[Box<[(); MAX]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Box<[Box<[(); MAX]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Vec<Box<[(); MAX]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: [Box<[()]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Box<[Box<[()]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Box<[Box<[()]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Vec<Box<[()]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: [Vec<()>; 3] = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Box<[Vec<()>; 3]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let cols: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
        }

        // unable to cover (run out of memory)
        // {
        //     let cols: [Box<[u8; MAX]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Box<[Box<[u8; MAX]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Box<[Box<[u8; MAX]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Vec<Box<[u8; MAX]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: [Box<[u8]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Box<[Box<[u8]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Box<[Box<[u8]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Vec<Box<[u8]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: [Vec<u8>; 2] = [vec![0; MAX], vec![0; MAX]];
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Box<[Vec<u8>; 2]> = Box::new([vec![0; MAX], vec![0; MAX]]);
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Box<[Vec<u8>]> = Box::new([vec![0; MAX], vec![0; MAX]]);
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);

        //     let cols: Vec<Vec<u8>> = vec![vec![0; MAX], vec![0; MAX]];
        //     let error = Matrix::try_from_cols(cols).unwrap_err();
        //     assert_eq!(error, Error::CapacityOverflow);
        // }

        {
            let cols: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let cols: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let cols: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let cols: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let cols: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let cols: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let cols: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let cols: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::try_from_cols(cols).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);
        }
    }

    #[test]
    fn test_from_col_iter() {
        let expected = {
            let order = Order::ColMajor;
            let shape = Shape::new(3, 2);
            let shape = AxisShape::from_shape(shape, order);
            let data = vec![1, 2, 3, 4, 5, 6];
            Matrix { order, shape, data }
        };

        let cols = [[1, 2, 3], [4, 5, 6]];
        let matrix = Matrix::from_col_iter(cols);
        testkit::assert_strict_eq(&matrix, &expected);
    }

    #[test]
    #[should_panic]
    fn test_from_col_iter_fails() {
        let cols = [vec![1, 2, 3], vec![4, 5]];
        Matrix::from_col_iter(cols);
    }
}
