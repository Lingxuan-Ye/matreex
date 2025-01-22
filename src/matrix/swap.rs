use super::index::MatrixIndex;
use super::order::Order;
use super::Matrix;
use crate::error::{Error, Result};
use std::ptr::swap;

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
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix.swap((0, 0), (1, 1))?;
    /// assert_eq!(matrix, matrix![[4, 1, 2], [3, 0, 5]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn swap<I, J>(&mut self, i: I, j: J) -> Result<&mut Self>
    where
        I: MatrixIndex<T, Output = T>,
        J: MatrixIndex<T, Output = T>,
    {
        i.ensure_in_bounds(self)?;
        j.ensure_in_bounds(self)?;
        unsafe {
            let element_i = i.get_unchecked_mut(self);
            let element_j = j.get_unchecked_mut(self);
            swap(element_i, element_j);
        }
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
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix.swap_rows(0, 1)?;
    /// assert_eq!(matrix, matrix![[3, 4, 5], [0, 1, 2]]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
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
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// matrix.swap_cols(0, 1)?;
    /// assert_eq!(matrix, matrix![[1, 0, 2], [4, 3, 5]]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
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

        matrix.switch_order();
        matrix.swap((0, 0), (1, 1)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[4, 1, 2], [3, 0, 5]]);

        matrix.switch_order();
        matrix.swap((0, 0), (1, 1)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.switch_order();
        matrix.swap((1, 1), (0, 0)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[4, 1, 2], [3, 0, 5]]);

        matrix.switch_order();
        matrix.swap((1, 1), (0, 0)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        let unchanged = matrix.clone();

        assert_eq!(matrix.swap((0, 0), (2, 2)), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        assert_eq!(matrix.swap((2, 2), (0, 0)), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        assert_eq!(matrix.swap((2, 2), (3, 3)), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap((0, 0), (2, 2)), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap((2, 2), (0, 0)), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap((2, 2), (3, 3)), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
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

        matrix.switch_order();
        matrix.swap_rows(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[3, 4, 5], [0, 1, 2]]);

        matrix.switch_order();
        matrix.swap_rows(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.switch_order();
        matrix.swap_rows(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[3, 4, 5], [0, 1, 2]]);

        matrix.switch_order();
        matrix.swap_rows(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        let unchanged = matrix.clone();

        assert_eq!(matrix.swap_rows(0, 2), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        assert_eq!(matrix.swap_rows(2, 0), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        assert_eq!(matrix.swap_rows(2, 3), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap_rows(0, 2), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap_rows(2, 0), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap_rows(2, 3), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
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

        matrix.switch_order();
        matrix.swap_cols(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 0, 2], [4, 3, 5]]);

        matrix.switch_order();
        matrix.swap_cols(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        matrix.switch_order();
        matrix.swap_cols(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 0, 2], [4, 3, 5]]);

        matrix.switch_order();
        matrix.swap_cols(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[0, 1, 2], [3, 4, 5]]);

        let unchanged = matrix.clone();

        assert_eq!(matrix.swap_cols(0, 3), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        assert_eq!(matrix.swap_cols(3, 0), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        assert_eq!(matrix.swap_cols(3, 4), Err(Error::IndexOutOfBounds));
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap_cols(0, 3), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap_cols(3, 0), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
        assert_eq!(matrix, unchanged);

        matrix.switch_order();
        assert_eq!(matrix.swap_cols(3, 4), Err(Error::IndexOutOfBounds));
        matrix.switch_order();
        assert_eq!(matrix, unchanged);
    }
}
