use super::super::Matrix;
use super::super::layout::{Layout, Order, RowMajor};
use crate::convert::{FromRowIterator, TryFromRows};
use crate::error::{Error, Result};
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem;

impl<T, O, const R: usize, const C: usize> TryFromRows<[[T; C]; R]> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from_rows(value: [[T; C]; R]) -> Result<Self> {
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        if size_of::<T>() == 0 {
            // Prevent dropping `T`s, since they are logically moved.
            mem::forget(value);
            unsafe {
                data.set_len(size);
            }
        } else {
            for row in value {
                data.extend(row);
            }
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize, const C: usize> TryFromRows<Box<[[T; C]; R]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from_rows(value: Box<[[T; C]; R]>) -> Result<Self> {
        Self::try_from_rows(value as Box<[[T; C]]>)
    }
}

impl<T, O, const C: usize> TryFromRows<Box<[[T; C]]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from_rows(value: Box<[[T; C]]>) -> Result<Self> {
        Self::try_from_rows(value.into_vec())
    }
}

impl<T, O, const C: usize> TryFromRows<Vec<[T; C]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from_rows(value: Vec<[T; C]>) -> Result<Self> {
        let nrows = value.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        if size_of::<T>() == 0 {
            // Prevent dropping `T`s, since they are logically moved. Note that this will
            // not leak, since a zero-sized type vector does not actually allocate.
            mem::forget(value);
            unsafe {
                data.set_len(size);
            }
        } else {
            for row in value {
                data.extend(row);
            }
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize, const C: usize> TryFromRows<[Box<[T; C]>; R]> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from_rows(value: [Box<[T; C]>; R]) -> Result<Self> {
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        for row in value {
            data.extend(row as Box<[T]>);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize, const C: usize> TryFromRows<Box<[Box<[T; C]>; R]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from_rows(value: Box<[Box<[T; C]>; R]>) -> Result<Self> {
        Self::try_from_rows(value as Box<[Box<[T; C]>]>)
    }
}

impl<T, O, const C: usize> TryFromRows<Box<[Box<[T; C]>]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from_rows(value: Box<[Box<[T; C]>]>) -> Result<Self> {
        Self::try_from_rows(value.into_vec())
    }
}

impl<T, O, const C: usize> TryFromRows<Vec<Box<[T; C]>>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from_rows(value: Vec<Box<[T; C]>>) -> Result<Self> {
        let nrows = value.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        for row in value {
            data.extend(row as Box<[T]>);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize> TryFromRows<[Box<[T]>; R]> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: [Box<[T]>; R]) -> Result<Self> {
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = R;
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize> TryFromRows<Box<[Box<[T]>; R]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(value as Box<[Box<[T]>]>)
    }
}

impl<T, O> TryFromRows<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: Box<[Box<[T]>]>) -> Result<Self> {
        Self::try_from_rows(value.into_vec())
    }
}

impl<T, O> TryFromRows<Vec<Box<[T]>>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: Vec<Box<[T]>>) -> Result<Self> {
        let nrows = value.len();
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize> TryFromRows<[Vec<T>; R]> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: [Vec<T>; R]) -> Result<Self> {
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = R;
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize> TryFromRows<Box<[Vec<T>; R]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(value as Box<[Vec<T>]>)
    }
}

impl<T, O> TryFromRows<Box<[Vec<T>]>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: Box<[Vec<T>]>) -> Result<Self> {
        Self::try_from_rows(value.into_vec())
    }
}

impl<T, O> TryFromRows<Vec<Vec<T>>> for Matrix<T, O>
where
    O: Order,
{
    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from_rows(value: Vec<Vec<T>>) -> Result<Self> {
        let nrows = value.len();
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, RowMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for row in iter {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, R> FromRowIterator<R, T> for Matrix<T, O>
where
    O: Order,
    R: IntoIterator<Item = T>,
{
    /// Converts from an iterator over rows.
    ///
    /// # Panics
    ///
    /// Panics if rows have inconsistent lengths or capacity overflows.
    fn from_row_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = R>,
    {
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else {
            return Self::new();
        };
        // Could panic if capacity overflows.
        let mut data: Vec<T> = first.into_iter().collect();
        let mut nrows = 1;
        let ncols = data.len();
        let mut size = ncols;
        for row in iter {
            // Could panic if capacity overflows.
            data.extend(row);
            if data.len() - size != ncols {
                panic!("{}", Error::LengthInconsistent);
            }
            nrows += 1;
            size = data.len();
        }
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<T, RowMajor>::from_shape_unchecked(shape);
        Matrix { layout, data }.with_order()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::layout::ColMajor;
    use super::*;
    use crate::dispatch_unary;
    use alloc::vec;

    #[test]
    fn test_try_from_rows() {
        const MAX: usize = isize::MAX as usize;

        let expected: Matrix<i32, RowMajor> = {
            let shape = Shape::new(2, 3);
            let layout = Layout::from_shape_unchecked(shape);
            let data = vec![1, 2, 3, 4, 5, 6];
            Matrix { layout, data }
        };

        dispatch_unary! {{
            let seq: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32; 3]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let output = Matrix::<i32, O>::try_from_rows(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [[(); MAX]; 2] = [[(); MAX], [(); MAX]];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[[(); MAX]; 2]> = Box::new([[(); MAX], [(); MAX]]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[[(); MAX]]> = Box::new([[(); MAX], [(); MAX]]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Vec<[(); MAX]> = vec![[(); MAX], [(); MAX]];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: [Box<[(); MAX]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[Box<[(); MAX]>; 2]> =
                Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[Box<[(); MAX]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Vec<Box<[(); MAX]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: [Box<[()]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[Box<[()]>; 2]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[Box<[()]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Vec<Box<[()]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: [Vec<()>; 2] = [vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[Vec<()>; 2]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::<(), O>::try_from_rows(seq).is_ok());

            let seq: [[(); MAX]; 3] = [[(); MAX], [(); MAX], [(); MAX]];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[[(); MAX]; 3]> = Box::new([[(); MAX], [(); MAX], [(); MAX]]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[[(); MAX]]> = Box::new([[(); MAX], [(); MAX], [(); MAX]]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<[(); MAX]> = vec![[(); MAX], [(); MAX], [(); MAX]];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Box<[(); MAX]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[(); MAX]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[(); MAX]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Box<[(); MAX]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Box<[()]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[()]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[()]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Box<[()]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Vec<()>; 3] = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Vec<()>; 3]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::<(), O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            // Unable to cover.
            // let seq: [Box<[u8; MAX]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8; MAX]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8; MAX]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Box<[u8; MAX]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: [Box<[u8]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Box<[u8]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: [Vec<u8>; 2] = [vec![0; MAX], vec![0; MAX]];
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Vec<u8>; 2]> = Box::new([vec![0; MAX], vec![0; MAX]]);
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Vec<u8>]> = Box::new([vec![0; MAX], vec![0; MAX]]);
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Vec<u8>> = vec![vec![0; MAX], vec![0; MAX]];
            // let error = Matrix::<u8, O>::try_from_rows(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            let seq: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::<i32, O>::try_from_rows(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);
        }}
    }

    #[test]
    fn test_from_row_iter() {
        let expected = Matrix::<i32, RowMajor>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();

        dispatch_unary! {{
            let iter = [[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::from_row_iter(iter);
            assert_eq!(output, expected);
        }}
    }

    #[test]
    #[should_panic]
    fn test_from_row_iter_fails_row_major() {
        let iter = [vec![1, 2, 3], vec![4, 5]];
        Matrix::<i32, RowMajor>::from_row_iter(iter);
    }

    #[test]
    #[should_panic]
    fn test_from_row_iter_fails_col_major() {
        let iter = [vec![1, 2, 3], vec![4, 5]];
        Matrix::<i32, ColMajor>::from_row_iter(iter);
    }
}
