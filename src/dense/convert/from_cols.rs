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
        let layout = Layout::<_, ColMajor>::from_shape_unchecked(shape);
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
        let layout = Layout::<_, ColMajor>::from_shape_unchecked(shape);
        let data = value.into_iter().flatten().collect();
        Matrix { layout, data }.with_order()
    }
}

impl<T, O, C> TryFromCols<C> for Matrix<T, O>
where
    Self: FromCols<C>,
    O: Order,
{
    fn try_from_cols(value: C) -> Result<Self> {
        Ok(Self::from_cols(value))
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
        let layout = Layout::<_, ColMajor>::from_shape(shape)?;
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
        let layout = Layout::<_, ColMajor>::from_shape(shape)?;
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
        let (layout, size) = Layout::<_, ColMajor>::from_shape_with_size(shape)?;
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
        let (layout, size) = Layout::<_, ColMajor>::from_shape_with_size(shape)?;
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
        let (layout, size) = Layout::<_, ColMajor>::from_shape_with_size(shape)?;
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
        let (layout, size) = Layout::<_, ColMajor>::from_shape_with_size(shape)?;
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

impl<T, O, V> FromColIterator<T, V> for Matrix<T, O>
where
    O: Order,
    V: IntoIterator<Item = T>,
{
    fn from_col_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
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
        let layout = Layout::<_, ColMajor>::from_shape_unchecked(shape);
        Matrix { layout, data }.with_order()
    }
}
