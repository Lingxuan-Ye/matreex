use super::index::MatrixIndex;
use super::order::Order;
use super::Matrix;
use crate::error::{Error, Result};
use std::ptr;

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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.swap((0, 0), (1, 1))?;
    /// assert_eq!(matrix, matrix![[5, 2, 3], [4, 1, 6]]);
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
            let x = i.get_unchecked_mut(self);
            let y = j.get_unchecked_mut(self);
            ptr::swap(x, y);
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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.swap_rows(0, 1)?;
    /// assert_eq!(matrix, matrix![[4, 5, 6], [1, 2, 3]]);
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
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.swap_cols(0, 1)?;
    /// assert_eq!(matrix, matrix![[2, 1, 3], [5, 4, 6]]);
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
        let base = self.data.as_mut_ptr();
        let index = m * self.major_stride();
        let jndex = n * self.major_stride();
        unsafe {
            let x = base.add(index);
            let y = base.add(jndex);
            let count = self.minor();
            ptr::swap_nonoverlapping(x, y, count);
        }
        Ok(self)
    }

    fn swap_minor_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.minor() || n >= self.minor() {
            return Err(Error::IndexOutOfBounds);
        }
        let base = self.data.as_mut_ptr();
        let mut index = m;
        let mut jndex = n;
        for _ in 0..self.major() {
            unsafe {
                let x = base.add(index);
                let y = base.add(jndex);
                ptr::swap(x, y);
            }
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
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        matrix.swap((0, 0), (1, 1)).unwrap();
        assert_eq!(matrix, matrix![[5, 2, 3], [4, 1, 6]]);

        matrix.swap((0, 0), (1, 1)).unwrap();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.swap((1, 1), (0, 0)).unwrap();
        assert_eq!(matrix, matrix![[5, 2, 3], [4, 1, 6]]);

        matrix.swap((1, 1), (0, 0)).unwrap();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();
        matrix.swap((0, 0), (1, 1)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[5, 2, 3], [4, 1, 6]]);

        matrix.switch_order();
        matrix.swap((0, 0), (1, 1)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();
        matrix.swap((1, 1), (0, 0)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[5, 2, 3], [4, 1, 6]]);

        matrix.switch_order();
        matrix.swap((1, 1), (0, 0)).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

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
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        matrix.swap_rows(0, 1).unwrap();
        assert_eq!(matrix, matrix![[4, 5, 6], [1, 2, 3]]);

        matrix.swap_rows(0, 1).unwrap();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.swap_rows(1, 0).unwrap();
        assert_eq!(matrix, matrix![[4, 5, 6], [1, 2, 3]]);

        matrix.swap_rows(1, 0).unwrap();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();
        matrix.swap_rows(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[4, 5, 6], [1, 2, 3]]);

        matrix.switch_order();
        matrix.swap_rows(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();
        matrix.swap_rows(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[4, 5, 6], [1, 2, 3]]);

        matrix.switch_order();
        matrix.swap_rows(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

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
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        matrix.swap_cols(0, 1).unwrap();
        assert_eq!(matrix, matrix![[2, 1, 3], [5, 4, 6]]);

        matrix.swap_cols(0, 1).unwrap();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.swap_cols(1, 0).unwrap();
        assert_eq!(matrix, matrix![[2, 1, 3], [5, 4, 6]]);

        matrix.swap_cols(1, 0).unwrap();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();
        matrix.swap_cols(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[2, 1, 3], [5, 4, 6]]);

        matrix.switch_order();
        matrix.swap_cols(0, 1).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

        matrix.switch_order();
        matrix.swap_cols(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[2, 1, 3], [5, 4, 6]]);

        matrix.switch_order();
        matrix.swap_cols(1, 0).unwrap();
        matrix.switch_order();
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);

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
