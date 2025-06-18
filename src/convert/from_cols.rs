use crate::Matrix;
use crate::error::{Error, Result};
use crate::order::Order;
use crate::shape::{AxisShape, Shape};
use alloc::boxed::Box;
use alloc::vec::Vec;

pub trait FromCols<T>: Sized {
    fn from_cols(value: T) -> Self;
}

pub trait TryFromCols<T>: Sized {
    fn try_from_cols(value: T) -> Result<Self>;
}

pub trait FromColIterator<T, V>: Sized
where
    V: IntoIterator<Item = T>,
{
    fn from_col_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>;
}

impl<T, const R: usize, const C: usize> FromCols<[[T; R]; C]> for Matrix<T> {
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

impl<T, const R: usize, const C: usize> FromCols<[Box<[T; R]>; C]> for Matrix<T> {
    fn from_cols(value: [Box<[T; R]>; C]) -> Self {
        let order = Order::ColMajor;
        let nrows = R;
        let ncols = C;
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flat_map(|col| *col).collect();
        Self { order, shape, data }
    }
}

impl<T, const R: usize, const C: usize> FromCols<Box<[[T; R]; C]>> for Matrix<T> {
    #[inline]
    fn from_cols(value: Box<[[T; R]; C]>) -> Self {
        Self::from_cols(*value)
    }
}

impl<T, const R: usize, const C: usize> FromCols<Box<[Box<[T; R]>; C]>> for Matrix<T> {
    #[inline]
    fn from_cols(value: Box<[Box<[T; R]>; C]>) -> Self {
        Self::from_cols(*value)
    }
}

impl<T, const R: usize> FromCols<Box<[[T; R]]>> for Matrix<T> {
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

impl<T, const R: usize> FromCols<Box<[Box<[T; R]>]>> for Matrix<T> {
    fn from_cols(value: Box<[Box<[T; R]>]>) -> Self {
        let order = Order::ColMajor;
        let nrows = R;
        let ncols = value.len();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let data = value.into_iter().flat_map(|col| *col).collect();
        Self { order, shape, data }
    }
}

impl<T, const R: usize> FromCols<Vec<[T; R]>> for Matrix<T> {
    #[inline]
    fn from_cols(value: Vec<[T; R]>) -> Self {
        Self::from_cols(value.into_boxed_slice())
    }
}

impl<T, const R: usize> FromCols<Vec<Box<[T; R]>>> for Matrix<T> {
    #[inline]
    fn from_cols(value: Vec<Box<[T; R]>>) -> Self {
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

impl<T, const C: usize> TryFromCols<[Box<[T]>; C]> for Matrix<T> {
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

impl<T, const C: usize> TryFromCols<[Vec<T>; C]> for Matrix<T> {
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

impl<T, const C: usize> TryFromCols<Box<[Box<[T]>; C]>> for Matrix<T> {
    #[inline]
    fn try_from_cols(value: Box<[Box<[T]>; C]>) -> Result<Self> {
        Self::try_from_cols(*value)
    }
}

impl<T, const C: usize> TryFromCols<Box<[Vec<T>; C]>> for Matrix<T> {
    #[inline]
    fn try_from_cols(value: Box<[Vec<T>; C]>) -> Result<Self> {
        Self::try_from_cols(*value)
    }
}

impl<T> TryFromCols<Box<[Box<[T]>]>> for Matrix<T> {
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

impl<T> TryFromCols<Box<[Vec<T>]>> for Matrix<T> {
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
    #[inline]
    fn try_from_cols(value: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_cols(value.into_boxed_slice())
    }
}

impl<T> TryFromCols<Vec<Box<[T]>>> for Matrix<T> {
    #[inline]
    fn try_from_cols(value: Vec<Box<[T]>>) -> Result<Self> {
        Self::try_from_cols(value.into_boxed_slice())
    }
}

impl<T, V> FromColIterator<T, V> for Matrix<T>
where
    V: IntoIterator<Item = T>,
{
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
