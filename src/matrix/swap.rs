use super::index::{AxisIndex, Index};
use super::order::Order;
use super::Matrix;
use crate::error::{Error, Result};

impl<T> Matrix<T> {
    /// Swaps the elements at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error};
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.swap((0, 0), (1, 1))?;
    /// assert_eq!(matrix, matrix![[4, 1, 2], [3, 0, 5]]);
    ///
    /// let result = matrix.swap((0, 0), (2, 2));
    /// assert_eq!(result, Err(Error::IndexOutOfBounds));
    /// # Ok(())
    /// # }
    /// ```
    pub fn swap<I, J>(&mut self, i: I, j: J) -> Result<&mut Self>
    where
        I: Into<Index>,
        J: Into<Index>,
    {
        let index = AxisIndex::from_index(i, self.order);
        let jndex = AxisIndex::from_index(j, self.order);
        index.ensure_in_bounds(self.shape)?;
        jndex.ensure_in_bounds(self.shape)?;
        let index = index.to_flattened(self.shape);
        let jndex = jndex.to_flattened(self.shape);
        self.data.swap(index, jndex);
        Ok(self)
    }

    /// Swaps the rows at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error};
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.swap_rows(0, 1)?;
    /// assert_eq!(matrix, matrix![[3, 4, 5], [0, 1, 2]]);
    ///
    /// let result = matrix.swap_rows(0, 2);
    /// assert_eq!(result, Err(Error::IndexOutOfBounds));
    /// # Ok(())
    /// # }
    /// ```
    pub fn swap_rows(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match self.order {
            Order::RowMajor => self.swap_major_axis_vectors(m, n),
            Order::ColMajor => self.swap_minor_axis_vectors(m, n),
        }
    }

    /// Swaps the columns at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Error};
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    ///
    /// matrix.swap_cols(0, 1)?;
    /// assert_eq!(matrix, matrix![[1, 0, 2], [4, 3, 5]]);
    ///
    /// let result = matrix.swap_cols(0, 3);
    /// assert_eq!(result, Err(Error::IndexOutOfBounds));
    /// # Ok(())
    /// # }
    /// ```
    pub fn swap_cols(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match self.order {
            Order::RowMajor => self.swap_minor_axis_vectors(m, n),
            Order::ColMajor => self.swap_major_axis_vectors(m, n),
        }
    }
}

impl<T> Matrix<T> {
    fn swap_major_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.major() || n >= self.major() {
            return Err(Error::IndexOutOfBounds);
        }
        let mut index = m * self.major_stride();
        let mut jndex = n * self.major_stride();
        for _ in 0..self.minor() {
            self.data.swap(index, jndex);
            index += 1;
            jndex += 1;
        }
        Ok(self)
    }

    fn swap_minor_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.minor() || n >= self.minor() {
            return Err(Error::IndexOutOfBounds);
        }
        let mut index = m;
        let mut jndex = n;
        for _ in 0..self.major() {
            self.data.swap(index, jndex);
            index += self.major_stride();
            jndex += self.major_stride();
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_swap() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix.swap((0, 0), (1, 1)).unwrap();
        assert_eq!(matrix, matrix![[4, 1, 2], [3, 0, 5]]);

        matrix.swap((0, 0), (1, 1)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.swap((1, 1), (0, 0)).unwrap();
        assert_eq!(matrix, matrix![[4, 1, 2], [3, 0, 5]]);

        matrix.swap((1, 1), (0, 0)).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        let unchanged = matrix.clone();

        let error = matrix.swap((0, 0), (2, 2)).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);

        let error = matrix.swap((2, 2), (0, 0)).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);

        let error = matrix.swap((2, 3), (3, 2)).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);
    }

    #[test]
    fn test_swap_rows() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix.swap_rows(0, 1).unwrap();
        assert_eq!(matrix, matrix![[3, 4, 5], [0, 1, 2]]);

        matrix.swap_rows(0, 1).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.swap_rows(1, 0).unwrap();
        assert_eq!(matrix, matrix![[3, 4, 5], [0, 1, 2]]);

        matrix.swap_rows(1, 0).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        let unchanged = matrix.clone();

        let error = matrix.swap_rows(0, 2).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);

        let error = matrix.swap_rows(2, 0).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);

        let error = matrix.swap_rows(2, 3).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);
    }

    #[test]
    fn test_swap_cols() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];

        matrix.swap_cols(0, 1).unwrap();
        assert_eq!(matrix, matrix![[1, 0, 2], [4, 3, 5]]);

        matrix.swap_cols(0, 1).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.swap_cols(1, 0).unwrap();
        assert_eq!(matrix, matrix![[1, 0, 2], [4, 3, 5]]);

        matrix.swap_cols(1, 0).unwrap();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        let unchanged = matrix.clone();

        let error = matrix.swap_cols(0, 3).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);

        let error = matrix.swap_cols(3, 0).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);

        let error = matrix.swap_cols(3, 4).unwrap_err();
        assert_eq!(error, Error::IndexOutOfBounds);
        assert_eq!(matrix, unchanged);
    }
}
