use crate::Matrix;
use crate::error::{Error, Result};
use crate::index::MatrixIndex;
use crate::order::Order;
use core::ptr;

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
    /// # use matreex::Result;
    /// use matreex::matrix;
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
        let x = self.get_mut(i)? as *mut T;
        let y = self.get_mut(j)? as *mut T;

        if x == y {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let x = base.with_addr(x.addr());
        let y = base.with_addr(y.addr());

        unsafe {
            ptr::swap_nonoverlapping(x, y, 1);
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
    /// # use matreex::Result;
    /// use matreex::matrix;
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
    /// # use matreex::Result;
    /// use matreex::matrix;
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
        } else if m == n {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let index = m * stride.major();
        let jndex = n * stride.major();

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
        } else if m == n {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let mut index = m * stride.minor();
        let mut jndex = n * stride.minor();

        for _ in 0..self.major() {
            unsafe {
                let x = base.add(index);
                let y = base.add(jndex);
                ptr::swap_nonoverlapping(x, y, 1);
            }
            index += stride.major();
            jndex += stride.major();
        }

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;
    use crate::testkit;

    #[test]
    fn test_swap() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];

        testkit::for_each_order_unary(matrix, |mut matrix| {
            matrix.swap((0, 0), (0, 0)).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap((0, 0), (1, 1)).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap((0, 0), (1, 1)).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap((1, 1), (0, 0)).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap((1, 1), (0, 0)).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            let unchanged = matrix.clone();

            let error = matrix.swap((0, 0), (2, 2)).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);

            let error = matrix.swap((2, 2), (0, 0)).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);

            let error = matrix.swap((2, 2), (3, 3)).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);
        });
    }

    #[test]
    fn test_swap_rows() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];

        testkit::for_each_order_unary(matrix, |mut matrix| {
            matrix.swap_rows(0, 0).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_rows(0, 1).unwrap();
            let expected = matrix![[4, 5, 6], [1, 2, 3]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_rows(0, 1).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_rows(1, 0).unwrap();
            let expected = matrix![[4, 5, 6], [1, 2, 3]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_rows(1, 0).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            let unchanged = matrix.clone();

            let error = matrix.swap_rows(0, 2).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);

            let error = matrix.swap_rows(2, 0).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);

            let error = matrix.swap_rows(2, 3).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);
        });
    }

    #[test]
    fn test_swap_cols() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];

        testkit::for_each_order_unary(matrix, |mut matrix| {
            matrix.swap_cols(0, 0).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_cols(0, 1).unwrap();
            let expected = matrix![[2, 1, 3], [5, 4, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_cols(0, 1).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_cols(1, 0).unwrap();
            let expected = matrix![[2, 1, 3], [5, 4, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            matrix.swap_cols(1, 0).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            testkit::assert_loose_eq(&matrix, &expected);

            let unchanged = matrix.clone();

            let error = matrix.swap_cols(0, 3).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);

            let error = matrix.swap_cols(3, 0).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);

            let error = matrix.swap_cols(3, 4).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            testkit::assert_loose_eq(&matrix, &unchanged);
        });
    }
}
