use super::order::Order;
use super::shape::{AxisShape, Shape};
use super::Matrix;
use crate::error::{Error, Result};

impl<T> Matrix<T> {
    /// Creates a new [`Matrix<T>`] instance from the given 2D array.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Matrix;
    ///
    /// let array = [[0, 1, 2], [3, 4, 5]];
    /// let matrix = Matrix::from_2darray(array);
    /// ```
    pub fn from_2darray<const R: usize, const C: usize>(value: [[T; C]; R]) -> Self {
        Self::from(value)
    }

    /// Creates a new [`Matrix<T>`] instance from the given slice.
    ///
    /// # Notes
    ///
    /// The matrix returned will always have `1` row and `src.len()` columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Matrix};
    ///
    /// let slice = [0, 1, 2, 3, 4, 5];
    /// let matrix = Matrix::from_slice(&slice);
    ///
    /// assert_eq!(matrix, matrix![[0, 1, 2, 3, 4, 5]]);
    /// ```
    pub fn from_slice(value: &[T]) -> Self
    where
        T: Clone,
    {
        Self::from(value)
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(value: [[T; C]; R]) -> Self {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked(Shape::new(R, C), order);
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

impl<T: Clone> From<&[T]> for Matrix<T> {
    fn from(value: &[T]) -> Self {
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked(Shape::new(1, value.len()), order);
        let data = value.to_vec();
        Self { order, shape, data }
    }
}

impl<T, V> FromIterator<V> for Matrix<T>
where
    V: IntoIterator<Item = T>,
{
    /// Creates a new [`Matrix<T>`] instance from an iterator over matrix rows.
    ///
    /// # Panics
    ///
    /// Panics if length in each iteration is inconsistent.
    fn from_iter<M: IntoIterator<Item = V>>(iter: M) -> Self {
        let mut data = Vec::with_capacity(0x1000);
        let mut data_len;
        let mut nrows;
        let ncols;
        let mut iter = iter.into_iter();
        match iter.next() {
            None => {
                return Self::empty();
            }
            Some(row) => {
                nrows = 1;
                data.extend(row);
                data_len = data.len();
                ncols = data_len;
            }
        }
        for row in iter {
            data.extend(row);
            if data.len() - data_len != ncols {
                panic!("{}", Error::LengthInconsistent);
            }
            data_len = data.len();
            nrows += 1;
        }
        data.shrink_to_fit();
        let order = Order::default();
        let shape = AxisShape::from_shape_unchecked(Shape::new(nrows, ncols), order);
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
        let shape = AxisShape::from_shape_unchecked(Shape::new(2, 3), order);
        let data = vec![0, 1, 2, 3, 4, 5];
        let expected = Matrix { order, shape, data };

        let array = [[0, 1, 2], [3, 4, 5]];
        assert_eq!(Matrix::from(array), expected);
        assert_eq!(Matrix::from_2darray(array), expected);
        assert_eq!(matrix![[0, 1, 2], [3, 4, 5]], expected);

        let array = [[0, 1], [2, 3], [4, 5]];
        assert_ne!(Matrix::from(array), expected);
        assert_ne!(Matrix::from_2darray(array), expected);
        assert_ne!(matrix![[0, 1], [2, 3], [4, 5]], expected);
    }

    #[test]
    fn test_from_slice_of_arrays() {
        let expected = matrix![[0, 1, 2], [3, 4, 5]];

        let arrays = [[0, 1, 2], [3, 4, 5]];
        assert_eq!(Matrix::from(&arrays[..]), expected);

        let arrays = [[0, 1, 2]];
        assert_ne!(Matrix::from(&arrays[..]), expected);

        let arrays = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
        assert_ne!(Matrix::from(&arrays[..]), expected);

        let arrays = [[0, 1], [2, 3], [4, 5]];
        assert_ne!(Matrix::from(&arrays[..]), expected);
    }

    #[test]
    fn test_try_from_slice_of_vecs() {
        const MAX: usize = isize::MAX as usize;

        let expected = matrix![[0, 1, 2], [3, 4, 5]];

        let vecs = [vec![0, 1, 2], vec![3, 4, 5]];
        assert_eq!(Matrix::try_from(&vecs[..]).unwrap(), expected);

        let vecs = [vec![0, 1, 2]];
        assert_ne!(Matrix::try_from(&vecs[..]).unwrap(), expected);

        let vecs = [vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
        assert_ne!(Matrix::try_from(&vecs[..]).unwrap(), expected);

        let vecs = [vec![0, 1], vec![2, 3], vec![4, 5]];
        assert_ne!(Matrix::try_from(&vecs[..]).unwrap(), expected);

        let vecs = [vec![(); MAX], vec![(); MAX]];
        assert!(Matrix::<()>::try_from(&vecs[..]).is_ok());

        let vecs = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
        assert_eq!(Matrix::<()>::try_from(&vecs[..]), Err(Error::SizeOverflow));

        // unable to cover (run out of memory)
        // let vecs = [vec![0u8; MAX], vec![0u8; MAX]];
        // assert_eq!(Matrix::<u8>::try_from(&vecs[..]), Err(Error::CapacityExceeded));

        let vecs = [vec![0, 1, 2], vec![3, 4]];
        assert_eq!(
            Matrix::<i32>::try_from(&vecs[..]),
            Err(Error::LengthInconsistent)
        );
    }

    #[test]
    fn test_from_slice() {
        let expected = matrix![[0, 1, 2, 3, 4, 5]];

        let array = [0, 1, 2, 3, 4, 5];
        assert_eq!(Matrix::from(&array[..]), expected);
        assert_eq!(Matrix::from_slice(&array), expected);

        let array = [0; 6];
        assert_ne!(Matrix::from(&array[..]), expected);
        assert_ne!(Matrix::from_slice(&array), expected);

        let vec = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(Matrix::from(&vec[..]), expected);
        assert_eq!(Matrix::from_slice(&vec), expected);

        let vec = vec![0; 6];
        assert_ne!(Matrix::from(&vec[..]), expected);
        assert_ne!(Matrix::from_slice(&vec), expected);
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
