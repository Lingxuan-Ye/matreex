use super::order::Order;
use super::shape::{AxisShape, Shape};
use super::Matrix;
use crate::error::{Error, Result};

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(value: [[T; C]; R]) -> Self {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked(Shape::new(R, C), order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T: Clone, const C: usize> From<Vec<[T; C]>> for Matrix<T> {
    fn from(value: Vec<[T; C]>) -> Self {
        let order = Order::default();
        let nrows = value.len();
        let shape = AxisShape::from_shape_unchecked(Shape::new(nrows, C), order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T: Clone, const C: usize> From<&[[T; C]]> for Matrix<T> {
    fn from(value: &[[T; C]]) -> Self {
        let order = Order::default();
        let nrows = value.len();
        let shape = AxisShape::from_shape_unchecked(Shape::new(nrows, C), order);
        let data = value.iter().flatten().cloned().collect();
        Self { order, shape, data }
    }
}

impl<T: Clone, const C: usize> TryFrom<[Vec<T>; C]> for Matrix<T> {
    type Error = Error;

    fn try_from(value: [Vec<T>; C]) -> Result<Self> {
        let order = Order::default();
        let nrows = C;
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = AxisShape::try_from_shape(Shape::new(nrows, ncols), order)?;
        Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(shape.size());
        for row in value {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T: Clone> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = Error;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self> {
        let order = Order::default();
        let nrows = value.len();
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = AxisShape::try_from_shape(Shape::new(nrows, ncols), order)?;
        Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(shape.size());
        for row in value {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T: Clone> TryFrom<&[Vec<T>]> for Matrix<T> {
    type Error = Error;

    fn try_from(value: &[Vec<T>]) -> Result<Self> {
        let order = Order::default();
        let nrows = value.len();
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = AxisShape::try_from_shape(Shape::new(nrows, ncols), order)?;
        Self::check_size(shape.size())?;
        let mut data = Vec::with_capacity(shape.size());
        for row in value {
            if row.len() != ncols {
                return Err(Error::LengthInconsistent);
            }
            data.extend_from_slice(row);
        }
        Ok(Self { order, shape, data })
    }
}

impl<T: Clone, const N: usize> From<[T; N]> for Matrix<T> {
    fn from(value: [T; N]) -> Self {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked(Shape::new(1, N), order);
        let data = value.to_vec();
        Self { order, shape, data }
    }
}

impl<T: Clone> From<Vec<T>> for Matrix<T> {
    fn from(value: Vec<T>) -> Self {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked(Shape::new(1, value.len()), order);
        let data = value;
        Self { order, shape, data }
    }
}

impl<T: Clone> From<&[T]> for Matrix<T> {
    fn from(value: &[T]) -> Self {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked(Shape::new(1, value.len()), order);
        let data = value.to_vec();
        Self { order, shape, data }
    }
}

