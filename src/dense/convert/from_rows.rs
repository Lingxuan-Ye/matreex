use super::super::Matrix;
use super::super::layout::{Layout, Order, RowMajor};
use crate::convert::{FromRowIterator, FromRows, TryFromRows};
use crate::error::{Error, Result};
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::vec::Vec;

impl<T, O, const R: usize, const C: usize> FromRows<[[T; C]; R]> for Matrix<T, O>
where
    O: Order,
{
    fn from_rows(value: [[T; C]; R]) -> Self {
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<_, RowMajor>::from_shape_unchecked(shape);
        let data = value.into_iter().flatten().collect();
        Matrix { layout, data }.with_order()
    }
}

impl<T, O, const R: usize, const C: usize> FromRows<Box<[[T; C]; R]>> for Matrix<T, O>
where
    O: Order,
{
    fn from_rows(value: Box<[[T; C]; R]>) -> Self {
        Self::from_rows(value as Box<[[T; C]]>)
    }
}

impl<T, O, const C: usize> FromRows<Box<[[T; C]]>> for Matrix<T, O>
where
    O: Order,
{
    fn from_rows(value: Box<[[T; C]]>) -> Self {
        Self::from_rows(value.into_vec())
    }
}

impl<T, O, const C: usize> FromRows<Vec<[T; C]>> for Matrix<T, O>
where
    O: Order,
{
    fn from_rows(value: Vec<[T; C]>) -> Self {
        let nrows = value.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<_, RowMajor>::from_shape_unchecked(shape);
        let data = value.into_iter().flatten().collect();
        Matrix { layout, data }.with_order()
    }
}

impl<T, O, R> TryFromRows<R> for Matrix<T, O>
where
    Self: FromRows<R>,
    O: Order,
{
    fn try_from_rows(value: R) -> Result<Self> {
        Ok(Self::from_rows(value))
    }
}

impl<T, O, const R: usize, const C: usize> TryFromRows<[Box<[T; C]>; R]> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: [Box<[T; C]>; R]) -> Result<Self> {
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<_, RowMajor>::from_shape(shape)?;
        let data = value.into_iter().flat_map(|row| row as Box<[T]>).collect();
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize, const C: usize> TryFromRows<Box<[Box<[T; C]>; R]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: Box<[Box<[T; C]>; R]>) -> Result<Self> {
        Self::try_from_rows(value as Box<[Box<[T; C]>]>)
    }
}

impl<T, O, const C: usize> TryFromRows<Box<[Box<[T; C]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: Box<[Box<[T; C]>]>) -> Result<Self> {
        Self::try_from_rows(value.into_vec())
    }
}

impl<T, O, const C: usize> TryFromRows<Vec<Box<[T; C]>>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: Vec<Box<[T; C]>>) -> Result<Self> {
        let nrows = value.len();
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let layout = Layout::<_, RowMajor>::from_shape(shape)?;
        let data = value.into_iter().flat_map(|row| row as Box<[T]>).collect();
        Ok(Matrix { layout, data }.with_order())
    }
}

impl<T, O, const R: usize> TryFromRows<[Box<[T]>; R]> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: [Box<[T]>; R]) -> Result<Self> {
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = R;
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<_, RowMajor>::from_shape_with_size(shape)?;
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
    fn try_from_rows(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(value as Box<[Box<[T]>]>)
    }
}

impl<T, O> TryFromRows<Box<[Box<[T]>]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: Box<[Box<[T]>]>) -> Result<Self> {
        Self::try_from_rows(value.into_vec())
    }
}

impl<T, O> TryFromRows<Vec<Box<[T]>>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: Vec<Box<[T]>>) -> Result<Self> {
        let nrows = value.len();
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<_, RowMajor>::from_shape_with_size(shape)?;
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
    fn try_from_rows(value: [Vec<T>; R]) -> Result<Self> {
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let nrows = R;
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<_, RowMajor>::from_shape_with_size(shape)?;
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
    fn try_from_rows(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(value as Box<[Vec<T>]>)
    }
}

impl<T, O> TryFromRows<Box<[Vec<T>]>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: Box<[Vec<T>]>) -> Result<Self> {
        Self::try_from_rows(value.into_vec())
    }
}

impl<T, O> TryFromRows<Vec<Vec<T>>> for Matrix<T, O>
where
    O: Order,
{
    fn try_from_rows(value: Vec<Vec<T>>) -> Result<Self> {
        let nrows = value.len();
        let mut iter = value.into_iter();
        let Some(first) = iter.next() else {
            return Ok(Self::new());
        };
        let ncols = first.len();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::<_, RowMajor>::from_shape_with_size(shape)?;
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

impl<T, O, V> FromRowIterator<T, V> for Matrix<T, O>
where
    O: Order,
    V: IntoIterator<Item = T>,
{
    fn from_row_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
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
        let layout = Layout::<_, RowMajor>::from_shape_unchecked(shape);
        Matrix { layout, data }.with_order()
    }
}
