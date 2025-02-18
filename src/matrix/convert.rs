use super::order::Order;
use super::shape::Shape;
use super::Matrix;
use crate::error::{Error, Result};

impl<T> Matrix<T> {
    /// Creates a new single-row [`Matrix<T>`] from a vector.
    ///
    /// # Examples
    /// ```
    /// use matreex::{matrix, Matrix};
    ///
    /// let row_vec = Matrix::from_row(vec![0, 1, 2]);
    /// assert_eq!(row_vec, matrix![[0, 1, 2]]);
    /// ```
    pub fn from_row(row: Vec<T>) -> Self {
        let order = Order::default();
        let shape = Shape::new(1, row.len()).to_axis_shape_unchecked(order);
        let data = row;
        Self { order, shape, data }
    }

    /// Creates a new single-column [`Matrix<T>`] from a vector.
    ///
    /// # Examples
    /// ```
    /// use matreex::{matrix, Matrix};
    ///
    /// let col_vec = Matrix::from_col(vec![0, 1, 2]);
    /// assert_eq!(col_vec, matrix![[0], [1], [2]]);
    /// ```
    pub fn from_col(col: Vec<T>) -> Self {
        let order = Order::default();
        let shape = Shape::new(col.len(), 1).to_axis_shape_unchecked(order);
        let data = col;
        Self { order, shape, data }
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(value: [[T; C]; R]) -> Self {
        let order = Order::default();
        let shape = Shape::new(R, C).to_axis_shape_unchecked(order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T, const C: usize> From<Vec<[T; C]>> for Matrix<T> {
    fn from(value: Vec<[T; C]>) -> Self {
        let order = Order::default();
        let nrows = value.len();
        let shape = Shape::new(nrows, C).to_axis_shape_unchecked(order);
        let data = value.into_iter().flatten().collect();
        Self { order, shape, data }
    }
}

impl<T: Clone, const C: usize> From<&[[T; C]]> for Matrix<T> {
    fn from(value: &[[T; C]]) -> Self {
        let order = Order::default();
        let nrows = value.len();
        let shape = Shape::new(nrows, C).to_axis_shape_unchecked(order);
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
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
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
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
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
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
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
    /// Creates a new [`Matrix<T>`] from an iterator over rows.
    ///
    /// # Panics
    ///
    /// Panics if rows have inconsistent lengths or if memory allocation fails.
    fn from_iter<M>(iter: M) -> Self
    where
        M: IntoIterator<Item = V>,
    {
        let mut iter = iter.into_iter();
        let Some(row) = iter.next() else {
            return Self::new();
        };
        let mut data: Vec<T> = row.into_iter().collect(); // could panic for running out of memory
        let mut nrows = 1;
        let ncols = data.len();
        let mut size = ncols;
        for row in iter {
            data.extend(row); // could panic for running out of memory
            if data.len() - size != ncols {
                panic!("{}", Error::LengthInconsistent);
            }
            nrows += 1;
            size = data.len();
        }
        data.shrink_to_fit();
        let order = Order::default();
        let shape = Shape::new(nrows, ncols).to_axis_shape_unchecked(order);
        Self { order, shape, data }
    }
}

#[cfg(test)]
mod tests {
    use super::super::order::Order;
    use super::*;
    use crate::matrix;

    #[test]
    fn test_from_row() {
        let row_vec: Matrix<i32> = Matrix::from_row(Vec::new());
        assert_eq!(row_vec, Matrix::with_default((1, 0)).unwrap());

        let row_vec = Matrix::from_row(vec![1, 2, 3]);
        assert_eq!(row_vec, matrix![[1, 2, 3]]);
    }

    #[test]
    fn test_from_col() {
        let col_vec: Matrix<i32> = Matrix::from_col(Vec::new());
        assert_eq!(col_vec, Matrix::with_default((0, 1)).unwrap());

        let col_vec = Matrix::from_col(vec![1, 2, 3]);
        assert_eq!(col_vec, matrix![[1], [2], [3]]);
    }

    #[test]
    fn test_from_arrays() {
        // avoid using `matrix!` to prevent circular validation
        let order = Order::default();
        let shape = Shape::new(2, 3).to_axis_shape_unchecked(order);
        let data = vec![1, 2, 3, 4, 5, 6];
        let mut expected = Matrix { order, shape, data };

        let arrays = [[1, 2, 3], [4, 5, 6]];
        assert_eq!(Matrix::from(arrays), expected);
        assert_eq!(Matrix::from(arrays.to_vec()), expected);
        assert_eq!(Matrix::from(&arrays[..]), expected);
        assert_eq!(matrix![[1, 2, 3], [4, 5, 6]], expected);

        let arrays = [[1, 4], [2, 5], [3, 6]];
        assert_ne!(Matrix::from(arrays), expected);
        assert_ne!(Matrix::from(arrays.to_vec()), expected);
        assert_ne!(Matrix::from(&arrays[..]), expected);
        assert_ne!(matrix![[1, 4], [2, 5], [3, 6]], expected);
        expected.transpose();
        assert_eq!(Matrix::from(arrays), expected);
        assert_eq!(Matrix::from(arrays.to_vec()), expected);
        assert_eq!(Matrix::from(&arrays[..]), expected);
        assert_eq!(matrix![[1, 4], [2, 5], [3, 6]], expected);
    }

    #[test]
    fn test_try_from_vectors() {
        const MAX: usize = isize::MAX as usize;

        let expected = matrix![[1, 2, 3], [4, 5, 6]];

        let vectors = [vec![1, 2, 3], vec![4, 5, 6]];
        assert_eq!(Matrix::try_from(vectors.clone()).unwrap(), expected);
        assert_eq!(Matrix::try_from(vectors.to_vec()).unwrap(), expected);
        assert_eq!(Matrix::try_from(&vectors[..]).unwrap(), expected);

        let vectors = [vec![1, 2, 3]];
        assert_ne!(Matrix::try_from(vectors.clone()).unwrap(), expected);
        assert_ne!(Matrix::try_from(vectors.to_vec()).unwrap(), expected);
        assert_ne!(Matrix::try_from(&vectors[..]).unwrap(), expected);

        let vectors = [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        assert_ne!(Matrix::try_from(vectors.clone()).unwrap(), expected);
        assert_ne!(Matrix::try_from(vectors.to_vec()).unwrap(), expected);
        assert_ne!(Matrix::try_from(&vectors[..]).unwrap(), expected);

        let vectors = [vec![1, 2], vec![3, 4], vec![5, 6]];
        assert_ne!(Matrix::try_from(vectors.clone()).unwrap(), expected);
        assert_ne!(Matrix::try_from(vectors.to_vec()).unwrap(), expected);
        assert_ne!(Matrix::try_from(&vectors[..]).unwrap(), expected);

        let vectors = [vec![(); MAX], vec![(); MAX]];
        assert!(Matrix::try_from(vectors.clone()).is_ok());
        assert!(Matrix::try_from(vectors.to_vec()).is_ok());
        assert!(Matrix::try_from(&vectors[..]).is_ok());

        let vectors = [vec![(); MAX], vec![(); MAX], vec![(); MAX]];
        assert_eq!(Matrix::try_from(vectors.clone()), Err(Error::SizeOverflow));
        assert_eq!(Matrix::try_from(vectors.to_vec()), Err(Error::SizeOverflow));
        assert_eq!(Matrix::try_from(&vectors[..]), Err(Error::SizeOverflow));

        // unable to cover (run out of memory)
        // let vectors = [vec![0u8; MAX], vec![0u8; MAX]];
        // assert_eq!(Matrix::try_from(vectors.clone()), Err(Error::CapacityOverflow));
        // assert_eq!(Matrix::try_from(vectors.to_vec()), Err(Error::CapacityOverflow));
        // assert_eq!(Matrix::try_from(&vectors[..]), Err(Error::CapacityOverflow));

        let vectors = [vec![1, 2, 3], vec![4, 5]];
        assert_eq!(
            Matrix::try_from(vectors.clone()),
            Err(Error::LengthInconsistent)
        );
        assert_eq!(
            Matrix::try_from(vectors.to_vec()),
            Err(Error::LengthInconsistent)
        );
        assert_eq!(
            Matrix::try_from(&vectors[..]),
            Err(Error::LengthInconsistent)
        );
    }

    #[test]
    fn test_from_iterator() {
        let expected = matrix![[1, 2, 3], [4, 5, 6]];

        let iterable = [[1, 2, 3], [4, 5, 6]];
        assert_eq!(Matrix::from_iter(iterable), expected);

        let iterable = [[1, 2], [3, 4], [5, 6]];
        assert_ne!(Matrix::from_iter(iterable), expected);
    }

    #[test]
    #[should_panic]
    fn test_from_iterator_fails() {
        let iterable = [vec![1, 2, 3], vec![4, 5]];
        Matrix::from_iter(iterable);
    }
}
