use super::super::Matrix;
use super::super::layout::{ColMajor, Layout, Order};
use crate::convert::{FromColIterator, FromCols, TryFromCols};
use crate::error::{Error, Result};
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::vec::Vec;

impl<T, O, const R: usize, const C: usize> FromCols<[[T; R]; C]> for Matrix<T, O>
where
    O: Order,
{
    fn from_cols(value: [[T; R]; C]) -> Self {
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<T, ColMajor>::from_shape_unchecked(shape);
        let data = value.into_iter().flatten().collect();
        Matrix { layout, data }.with_order()
    }
}

impl<T, O, const R: usize, const C: usize> FromCols<Box<[[T; R]; C]>> for Matrix<T, O>
where
    O: Order,
{
    fn from_cols(value: Box<[[T; R]; C]>) -> Self {
        Self::from_cols(value as Box<[[T; R]]>)
    }
}

impl<T, O, const R: usize> FromCols<Box<[[T; R]]>> for Matrix<T, O>
where
    O: Order,
{
    fn from_cols(value: Box<[[T; R]]>) -> Self {
        Self::from_cols(value.into_vec())
    }
}

impl<T, O, const R: usize> FromCols<Vec<[T; R]>> for Matrix<T, O>
where
    O: Order,
{
    fn from_cols(value: Vec<[T; R]>) -> Self {
        let nrows = R;
        let ncols = value.len();
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<T, ColMajor>::from_shape_unchecked(shape);
        let data = value.into_iter().flatten().collect();
        Matrix { layout, data }.with_order()
    }
}

impl<T, O, const R: usize, const C: usize> TryFromCols<[Box<[T; R]>; C]> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: [Box<[T; R]>; C]) -> Result<Self> {
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<T, ColMajor>::from_shape(shape)?;
        let data = value.into_iter().flat_map(|col| col as Box<[T]>).collect();
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize, const C: usize> TryFromCols<Box<[Box<[T; R]>; C]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Box<[Box<[T; R]>; C]>) -> Result<Self> {
        Self::try_from_cols(value as Box<[Box<[T; R]>]>)
    }
}

impl<T, O, const R: usize> TryFromCols<Box<[Box<[T; R]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Box<[Box<[T; R]>]>) -> Result<Self> {
        Self::try_from_cols(value.into_vec())
    }
}

impl<T, O, const R: usize> TryFromCols<Vec<Box<[T; R]>>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Vec<Box<[T; R]>>) -> Result<Self> {
        let nrows = R;
        let ncols = value.len();
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<T, ColMajor>::from_shape(shape)?;
        let data = value.into_iter().flat_map(|col| col as Box<[T]>).collect();
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const C: usize> TryFromCols<[Box<[T]>; C]> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: [Box<[T]>; C]) -> Result<Self> {
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = first.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, ColMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const C: usize> TryFromCols<Box<[Box<[T]>; C]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Box<[Box<[T]>; C]>) -> Result<Self> {
        Self::try_from_cols(value as Box<[Box<[T]>]>)
    }
}

impl<T, O> TryFromCols<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Box<[Box<[T]>]>) -> Result<Self> {
        Self::try_from_cols(value.into_vec())
    }
}

impl<T, O> TryFromCols<Vec<Box<[T]>>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Vec<Box<[T]>>) -> Result<Self> {
        let ncols = value.len();
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, ColMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const C: usize> TryFromCols<[Vec<T>; C]> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: [Vec<T>; C]) -> Result<Self> {
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = first.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, ColMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const C: usize> TryFromCols<Box<[Vec<T>; C]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Box<[Vec<T>; C]>) -> Result<Self> {
        Self::try_from_cols(value as Box<[Vec<T>]>)
    }
}

impl<T, O> TryFromCols<Box<[Vec<T>]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Box<[Vec<T>]>) -> Result<Self> {
        Self::try_from_cols(value.into_vec())
    }
}

impl<T, O> TryFromCols<Vec<Vec<T>>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_cols(value: Vec<Vec<T>>) -> Result<Self> {
        let ncols = value.len();
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<T, ColMajor>::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);
        data.extend(first);
        for col in iter {
            if col.len() != nrows {
                return Err(Error::LengthInconsistent);
            }
            data.extend(col);
        }
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, C> FromColIterator<C, T> for Matrix<T, O>
where
    O: Order,
    C: IntoIterator<Item = T>,
{
    fn from_col_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = C>,
    {
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else {
            return Self::new();
        };
        // Could panic if capacity overflows.
        let mut data: Vec<T> = first.into_iter().collect();
        let nrows = data.len();
        let mut ncols = 1;
        let mut size = nrows;
        for col in iter {
            // Could panic if capacity overflows.
            data.extend(col);
            if data.len() - size != nrows {
                panic!("{}", Error::LengthInconsistent);
            }
            ncols += 1;
            size = data.len();
        }
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<T, ColMajor>::from_shape_unchecked(shape);
        Matrix { layout, data }.with_order()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::layout::RowMajor;
    use super::*;
    use crate::convert::FromRows;
    use crate::dispatch_unary;
    use alloc::vec;

    #[test]
    fn test_from_cols() {
        let expected = Matrix::<i32, RowMajor>::from_rows([[1, 4], [2, 5], [3, 6]]);

        dispatch_unary! {{
            let seq: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::from_cols(seq);
            assert_eq!(output, expected);

            let seq: Box<[[i32; 3]; 2]> = Box::new([[1, 2, 3], [4, 5, 6]]);
            let output = Matrix::<i32, O>::from_cols(seq);
            assert_eq!(output, expected);

            let seq: Box<[[i32; 3]]> = Box::new([[1, 2, 3], [4, 5, 6]]);
            let output = Matrix::<i32, O>::from_cols(seq);
            assert_eq!(output, expected);

            let seq: Vec<[i32; 3]> = vec![[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::from_cols(seq);
            assert_eq!(output, expected);
        }}
    }

    #[test]
    fn test_try_from_cols() {
        const MAX: usize = isize::MAX as usize;

        let expected = Matrix::<i32, RowMajor>::from_rows([[1, 4], [2, 5], [3, 6]]);

        dispatch_unary! {{
            let seq: [Box<[i32; 3]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32; 3]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32; 3]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Box<[i32; 3]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5, 6])]);
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5, 6])];
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5, 6]];
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5, 6]]);
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            let seq: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let output = Matrix::<i32, O>::try_from_cols(seq).unwrap();
            assert_eq!(output, expected);

            // Unable to cover.
            // let seq: [Box<[(); MAX]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            // assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            // let seq: Box<[Box<[(); MAX]>; 2]> =
            //     Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            // assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            // let seq: Box<[Box<[(); MAX]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            // assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            // let seq: Vec<Box<[(); MAX]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            // assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: [Box<[()]>; 2] = [Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: Box<[Box<[()]>; 2]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: Box<[Box<[()]>]> = Box::new([Box::new([(); MAX]), Box::new([(); MAX])]);
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: Vec<Box<[()]>> = vec![Box::new([(); MAX]), Box::new([(); MAX])];
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: [Vec<()>; 2] = [vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: Box<[Vec<()>; 2]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX]]);
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX]];
            assert!(Matrix::<(), O>::try_from_cols(seq).is_ok());

            let seq: [Box<[(); MAX]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[(); MAX]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[(); MAX]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Box<[(); MAX]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Box<[()]>; 3] = [
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[()]>; 3]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Box<[()]>]> = Box::new([
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ]);
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Box<[()]>> = vec![
                Box::new([(); MAX]),
                Box::new([(); MAX]),
                Box::new([(); MAX]),
            ];
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: [Vec<()>; 3] = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Vec<()>; 3]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Box<[Vec<()>]> = Box::new([vec![(); MAX], vec![(); MAX], vec![(); MAX]]);
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let seq: Vec<Vec<()>> = vec![vec![(); MAX], vec![(); MAX], vec![(); MAX]];
            let error = Matrix::<(), O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            // Unable to cover.
            // let seq: [Box<[u8; MAX]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8; MAX]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8; MAX]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Box<[u8; MAX]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: [Box<[u8]>; 2] = [Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8]>; 2]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Box<[u8]>]> = Box::new([Box::new([0; MAX]), Box::new([0; MAX])]);
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Box<[u8]>> = vec![Box::new([0; MAX]), Box::new([0; MAX])];
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: [Vec<u8>; 2] = [vec![0; MAX], vec![0; MAX]];
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Vec<u8>; 2]> = Box::new([vec![0; MAX], vec![0; MAX]]);
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Box<[Vec<u8>]> = Box::new([vec![0; MAX], vec![0; MAX]]);
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            // let seq: Vec<Vec<u8>> = vec![vec![0; MAX], vec![0; MAX]];
            // let error = Matrix::<u8, O>::try_from_cols(seq).unwrap_err();
            // assert_eq!(error, Error::CapacityOverflow);

            let seq: [Box<[i32]>; 2] = [Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Box<[i32]>]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Box<[i32]>; 2]> = Box::new([Box::new([1, 2, 3]), Box::new([4, 5])]);
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Vec<Box<[i32]>> = vec![Box::new([1, 2, 3]), Box::new([4, 5])];
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: [Vec<i32>; 2] = [vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Vec<i32>; 2]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Box<[Vec<i32>]> = Box::new([vec![1, 2, 3], vec![4, 5]]);
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);

            let seq: Vec<Vec<i32>> = vec![vec![1, 2, 3], vec![4, 5]];
            let error = Matrix::<i32, O>::try_from_cols(seq).unwrap_err();
            assert_eq!(error, Error::LengthInconsistent);
        }}
    }

    #[test]
    fn test_from_col_iter() {
        let expected = Matrix::<i32, RowMajor>::from_rows([[1, 4], [2, 5], [3, 6]]);

        dispatch_unary! {{
            let iter = [[1, 2, 3], [4, 5, 6]];
            let output = Matrix::<i32, O>::from_col_iter(iter);
            assert_eq!(output, expected);
        }}
    }

    #[test]
    #[should_panic]
    fn test_from_col_iter_fails_row_major() {
        let iter = [vec![1, 2, 3], vec![4, 5]];
        Matrix::<i32, RowMajor>::from_col_iter(iter);
    }

    #[test]
    #[should_panic]
    fn test_from_col_iter_fails_col_major() {
        let iter = [vec![1, 2, 3], vec![4, 5]];
        Matrix::<i32, ColMajor>::from_col_iter(iter);
    }
}
