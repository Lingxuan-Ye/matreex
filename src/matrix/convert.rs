use super::order::Order;
use super::shape::AxisShape;
use super::Matrix;
use crate::error::{Error, Result};

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(value: [[T; C]; R]) -> Self {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked((R, C), order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> From<Vec<[T; C]>> for Matrix<T> {
    fn from(value: Vec<[T; C]>) -> Self {
        let order = Order::default();
        let nrows = value.len();
        let shape = AxisShape::from_shape_unchecked((nrows, C), order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T: Clone, const C: usize> From<&[[T; C]]> for Matrix<T> {
    fn from(value: &[[T; C]]) -> Self {
        let order = Order::default();
        let nrows = value.len();
        let shape = AxisShape::from_shape_unchecked((nrows, C), order);
        let data = value.iter().flatten().cloned().collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> TryFrom<[Vec<T>; C]> for Matrix<T> {
    type Error = Error;

    fn try_from(value: [Vec<T>; C]) -> Result<Self> {
        let order = Order::default();
        let nrows = C;
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = AxisShape::try_from_shape((nrows, ncols), order)?;
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

impl<T> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = Error;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self> {
        let order = Order::default();
        let nrows = value.len();
        let ncols = value.first().map_or(0, |row| row.len());
        let shape = AxisShape::try_from_shape((nrows, ncols), order)?;
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
        let shape = AxisShape::try_from_shape((nrows, ncols), order)?;
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

impl<T, V> FromIterator<V> for Matrix<T>
where
    V: IntoIterator<Item = T>,
{
    /// Creates a new [`Matrix<T>`] from an iterator over matrix rows.
    ///
    /// # Panics
    ///
    /// Panics if length in each iteration is inconsistent, or memory runs out.
    fn from_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        let mut nrows;
        let ncols;
        let mut data_len;
        let mut data = Vec::new();
        let mut iter = iter.into_iter();
        match iter.next() {
            None => {
                return Self::new();
            }
            Some(row) => {
                data.extend(row);
                nrows = 1;
                ncols = data.len();
                data_len = data.len();
            }
        }
        for row in iter {
            data.extend(row);
            if data.len() - data_len != ncols {
                panic!("{}", Error::LengthInconsistent);
            }
            nrows += 1;
            data_len = data.len();
        }
        data.shrink_to_fit();
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked((nrows, ncols), order);
        Self { order, shape, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_from_2darray() {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked((2, 3), order);
        let data = vec![0, 1, 2, 3, 4, 5];
        let mut expected = Matrix { order, shape, data };

        let array = [[0, 1, 2], [3, 4, 5]];
        assert_eq!(Matrix::from(array), expected);
        assert_eq!(Matrix::from(array.to_vec()), expected);
        assert_eq!(Matrix::from(&array[..]), expected);
        assert_eq!(matrix![[0, 1, 2], [3, 4, 5]], expected);

        let array = [[0, 3], [1, 4], [2, 5]];
        assert_ne!(Matrix::from(array), expected);
        assert_ne!(Matrix::from(array.to_vec()), expected);
        assert_ne!(Matrix::from(&array[..]), expected);
        assert_ne!(matrix![[0, 3], [1, 4], [2, 5]], expected);
        expected.transpose();
        assert_eq!(Matrix::from(array), expected);
        assert_eq!(Matrix::from(array.to_vec()), expected);
        assert_eq!(Matrix::from(&array[..]), expected);
        assert_eq!(matrix![[0, 3], [1, 4], [2, 5]], expected);
    }

    #[test]
    fn test_try_from_array_of_arrays() {
        const MAX: usize = isize::MAX as usize;

        let expected = matrix![[0, 1, 2], [3, 4, 5]];

        let aoa = [vec![0, 1, 2], vec![3, 4, 5]];
        assert_eq!(Matrix::try_from(aoa.clone()).unwrap(), expected);
        assert_eq!(Matrix::try_from(aoa.to_vec()).unwrap(), expected);
        assert_eq!(Matrix::try_from(&aoa[..]).unwrap(), expected);

        let aoa = [vec![0, 1, 2]];
        assert_ne!(Matrix::try_from(aoa.clone()).unwrap(), expected);
        assert_ne!(Matrix::try_from(aoa.to_vec()).unwrap(), expected);
        assert_ne!(Matrix::try_from(&aoa[..]).unwrap(), expected);

        let aoa = [vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
        assert_ne!(Matrix::try_from(aoa.clone()).unwrap(), expected);
        assert_ne!(Matrix::try_from(aoa.to_vec()).unwrap(), expected);
        assert_ne!(Matrix::try_from(&aoa[..]).unwrap(), expected);

        let aoa = [vec![0, 1], vec![2, 3], vec![4, 5]];
        assert_ne!(Matrix::try_from(aoa.clone()).unwrap(), expected);
        assert_ne!(Matrix::try_from(aoa.to_vec()).unwrap(), expected);
        assert_ne!(Matrix::try_from(&aoa[..]).unwrap(), expected);

        let aoa = [vec![(); MAX], vec![(); MAX]];
        assert!(Matrix::try_from(aoa.clone()).is_ok());
        assert!(Matrix::try_from(aoa.to_vec()).is_ok());
        assert!(Matrix::try_from(&aoa[..]).is_ok());

        let aoa = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
        assert_eq!(Matrix::try_from(aoa.clone()), Err(Error::SizeOverflow));
        assert_eq!(Matrix::try_from(aoa.to_vec()), Err(Error::SizeOverflow));
        assert_eq!(Matrix::try_from(&aoa[..]), Err(Error::SizeOverflow));

        // unable to cover (run out of memory)
        // let aoa = [vec![0u8; MAX], vec![0u8; MAX]];
        // assert_eq!(Matrix::try_from(aoa.clone()), Err(Error::CapacityExceeded));
        // assert_eq!(Matrix::try_from(aoa.to_vec()), Err(Error::CapacityExceeded));
        // assert_eq!(Matrix::try_from(&aoa[..]), Err(Error::CapacityExceeded));

        let aoa = [vec![0, 1, 2], vec![3, 4]];
        assert_eq!(
            Matrix::try_from(aoa.clone()),
            Err(Error::LengthInconsistent)
        );
        assert_eq!(
            Matrix::try_from(aoa.to_vec()),
            Err(Error::LengthInconsistent)
        );
        assert_eq!(Matrix::try_from(&aoa[..]), Err(Error::LengthInconsistent));
    }

    #[test]
    fn test_from_iterator() {
        let expected = matrix![[0, 1, 2], [3, 4, 5]];

        let iterable = [[0, 1, 2], [3, 4, 5]];
        assert_eq!(Matrix::from_iter(iterable), expected);

        let iterable = [[0, 1], [2, 3], [4, 5]];
        assert_ne!(Matrix::from_iter(iterable), expected);
    }

    #[test]
    #[should_panic]
    fn test_from_iterator_fails() {
        let iterable = [vec![0, 1, 2], vec![3, 4]];
        Matrix::from_iter(iterable);
    }
}
