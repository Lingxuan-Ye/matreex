use crate::Matrix;
use crate::error::{Error, Result};
use crate::order::Order;
use crate::shape::{AxisShape, Shape};
use alloc::boxed::Box;
use alloc::vec::Vec;

pub trait FromRows<T>: Sized {
    fn from_rows(value: T) -> Self;
}

pub trait TryFromRows<T>: Sized {
    fn try_from_rows(value: T) -> Result<Self>;
}

pub trait FromRowIterator<T, V>: Sized
where
    V: IntoIterator<Item = T>,
{
    fn from_row_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>;
}

impl<T, const R: usize, const C: usize> FromRows<[[T; C]; R]> for Matrix<T> {
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
    #[inline]
    fn from_rows(value: Box<[[T; C]; R]>) -> Self {
        Self::from_rows(*value)
    }
}

impl<T, const R: usize, const C: usize> FromRows<Box<[Box<[T; C]>; R]>> for Matrix<T> {
    #[inline]
    fn from_rows(value: Box<[Box<[T; C]>; R]>) -> Self {
        Self::from_rows(*value)
    }
}

impl<T, const C: usize> FromRows<Box<[[T; C]]>> for Matrix<T> {
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
    #[inline]
    fn from_rows(value: Vec<[T; C]>) -> Self {
        Self::from_rows(value.into_boxed_slice())
    }
}

impl<T, const C: usize> FromRows<Vec<Box<[T; C]>>> for Matrix<T> {
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
    #[inline]
    fn try_from_rows(value: Box<[Box<[T]>; R]>) -> Result<Self> {
        Self::try_from_rows(*value)
    }
}

impl<T, const R: usize> TryFromRows<Box<[Vec<T>; R]>> for Matrix<T> {
    #[inline]
    fn try_from_rows(value: Box<[Vec<T>; R]>) -> Result<Self> {
        Self::try_from_rows(*value)
    }
}

impl<T> TryFromRows<Box<[Box<[T]>]>> for Matrix<T> {
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
    #[inline]
    fn try_from_rows(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_rows(value.into_boxed_slice())
    }
}

impl<T> TryFromRows<Vec<Vec<T>>> for Matrix<T> {
    #[inline]
    fn try_from_rows(value: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_rows(value.into_boxed_slice())
    }
}

impl<T, V> FromRowIterator<T, V> for Matrix<T>
where
    V: IntoIterator<Item = T>,
{
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
