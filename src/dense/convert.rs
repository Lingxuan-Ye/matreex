use super::Matrix;
use super::layout::Order;
use crate::convert::{FromRowIterator, IntoRows, TryFromRows};
use crate::error::{Error, Result};
use alloc::boxed::Box;
use alloc::vec::Vec;

mod from_cols;
mod from_rows;
mod into_cols;
mod into_rows;

impl<T, O, const R: usize, const C: usize> TryFrom<[[T; C]; R]> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from(value: [[T; C]; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize, const C: usize> TryFrom<Box<[[T; C]; R]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from(value: Box<[[T; C]; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const C: usize> TryFrom<Box<[[T; C]]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from(value: Box<[[T; C]]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const C: usize> TryFrom<Vec<[T; C]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    fn try_from(value: Vec<[T; C]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize, const C: usize> TryFrom<[Box<[T; C]>; R]> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from(value: [Box<[T; C]>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize, const C: usize> TryFrom<Box<[Box<[T; C]>; R]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from(value: Box<[Box<[T; C]>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const C: usize> TryFrom<Box<[Box<[T; C]>]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from(value: Box<[Box<[T; C]>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const C: usize> TryFrom<Vec<Box<[T; C]>>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    fn try_from(value: Vec<Box<[T; C]>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<[Box<[T]>; R]> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: [Box<[T]>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<Box<[Box<[T]>; R]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: Box<[Box<[T]>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Vec<Box<[T]>>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<[Vec<T>; R]> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: [Vec<T>; R]) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, const R: usize> TryFrom<Box<[Vec<T>; R]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Box<[Vec<T>]>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: Box<[Vec<T>]>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O> TryFrom<Vec<Vec<T>>> for Matrix<T, O>
where
    O: Order,
{
    type Error = Error;

    /// Attempts to convert from a sequence of rows.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the total number of elements exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    /// - [`Error::LengthInconsistent`] if rows have inconsistent lengths.
    fn try_from(value: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_rows(value)
    }
}

impl<T, O, V> FromIterator<V> for Matrix<T, O>
where
    O: Order,
    V: IntoIterator<Item = T>,
{
    /// Converts from an iterator over rows.
    ///
    /// # Panics
    ///
    /// Panics if rows have inconsistent lengths or capacity overflows.
    fn from_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        Self::from_row_iter(iter)
    }
}

impl<T, O> From<Matrix<T, O>> for Box<[Box<[T]>]>
where
    O: Order,
{
    fn from(value: Matrix<T, O>) -> Self {
        value.into_rows()
    }
}

impl<T, O> From<Matrix<T, O>> for Vec<Box<[T]>>
where
    O: Order,
{
    fn from(value: Matrix<T, O>) -> Self {
        value.into_rows()
    }
}

impl<T, O> From<Matrix<T, O>> for Box<[Vec<T>]>
where
    O: Order,
{
    fn from(value: Matrix<T, O>) -> Self {
        value.into_rows()
    }
}

impl<T, O> From<Matrix<T, O>> for Vec<Vec<T>>
where
    O: Order,
{
    fn from(value: Matrix<T, O>) -> Self {
        value.into_rows()
    }
}

#[cfg(test)]
mod tests {
    use super::super::layout::{ColMajor, RowMajor};
    use super::*;
    use crate::dispatch_unary;
    use alloc::vec;

    #[test]
    fn test_try_from() {
        const MAX: usize = isize::MAX as usize;

        let expected = Matrix::<i32, RowMajor>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();

        dispatch_unary! {{
            let seq: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32; 3]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let output = Matrix::<i32, O>::try_from(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [[(); MAX]; 2] = [[(); MAX], [(); MAX]];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[[(); MAX]; 2]> = Box::new([[(); MAX], [(); MAX]]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[[(); MAX]]> = Box::new([[(); MAX], [(); MAX]]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Vec<[(); MAX]> = vec![[(); MAX], [(); MAX]];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: [Box<[(); MAX]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[Box<[(); MAX]>; 2]> =
                Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[Box<[(); MAX]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Vec<Box<[(); MAX]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: [Box<[()]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[Box<[()]>; 2]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[Box<[()]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Vec<Box<[()]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: [Vec<()>; 2] = [vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[Vec<()>; 2]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::<(), O>::try_from(seq).is_ok());

            let seq: [[(); MAX]; 3] = [[(); MAX], [(); MAX], [(); MAX]];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[[(); MAX]; 3]> = Box::new([[(); MAX], [(); MAX], [(); MAX]]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[[(); MAX]]> = Box::new([[(); MAX], [(); MAX], [(); MAX]]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<[(); MAX]> = vec![[(); MAX], [(); MAX], [(); MAX]];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Box<[(); MAX]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[(); MAX]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[(); MAX]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Box<[(); MAX]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Box<[()]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[()]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[()]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Box<[()]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Vec<()>; 3] = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Vec<()>; 3]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::<(), O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            // Unable to cover.
            // let seq: [Box<[u8; MAX]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8; MAX]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8; MAX]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Box<[u8; MAX]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: [Box<[u8]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Box<[u8]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: [Vec<u8>; 2] = [vec![0; MAX], vec![0; MAX]];
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Vec<u8>; 2]> = Box::new([vec![0; MAX], vec![0; MAX]]);
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Vec<u8>]> = Box::new([vec![0; MAX], vec![0; MAX]]);
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Vec<u8>> = vec![vec![0; MAX], vec![0; MAX]];
            // let error = Matrix::<u8, O>::try_from(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            let seq: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::<i32, O>::try_from(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);
        }}
    }

    #[test]
    fn test_from_iter() {
        let expected = Matrix::<i32, RowMajor>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();

        dispatch_unary! {{
            let iter = [[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::from_iter(iter);
            assert_eq!(output, expected);
        }}
    }

    #[test]
    #[should_panic]
    fn test_from_iter_fails_row_major() {
        let iter = [vec![1, 2, 3], vec![4, 5]];
        Matrix::<i32, RowMajor>::from_iter(iter);
    }

    #[test]
    #[should_panic]
    fn test_from_iter_fails_col_major() {
        let iter = [vec![1, 2, 3], vec![4, 5]];
        Matrix::<i32, ColMajor>::from_iter(iter);
    }

    #[test]
    fn test_into() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Box<[Box<[i32]>]> = matrix.into();
            let expected: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Vec<Box<[i32]>> = matrix.into();
            let expected: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Box<[Vec<i32>]> = matrix.into();
            let expected: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            assert_eq!(output, expected);

            let matrix = Matrix::<i32, O>::try_from_rows([[1, 2, 3], [4, 5, 6]]).unwrap();
            let output: Vec<Vec<i32>> = matrix.into();
            let expected: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            assert_eq!(output, expected);
        }}
    }
}
