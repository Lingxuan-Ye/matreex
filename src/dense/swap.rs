use super::Matrix;
use super::layout::{Order, OrderKind};
use crate::error::{Error, Result};
use crate::index::MatrixIndex;
use core::ptr;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
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
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let _ = matrix.swap((0, 0), (1, 1));
    /// assert_eq!(matrix, matrix![[5, 2, 3], [4, 1, 6]]);
    /// ```
    pub fn swap<I, J>(&mut self, i: I, j: J) -> Result<&mut Self>
    where
        I: for<'a> MatrixIndex<Self, OutputMut<'a> = &'a mut T>,
        J: for<'a> MatrixIndex<Self, OutputMut<'a> = &'a mut T>,
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
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let _ = matrix.swap_rows(0, 1);
    /// assert_eq!(matrix, matrix![[4, 5, 6], [1, 2, 3]]);
    /// ```
    pub fn swap_rows(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match O::KIND {
            OrderKind::RowMajor => self.swap_major_axis_vectors(m, n),
            OrderKind::ColMajor => self.swap_minor_axis_vectors(m, n),
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
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let _ = matrix.swap_cols(0, 1);
    /// assert_eq!(matrix, matrix![[2, 1, 3], [5, 4, 6]]);
    /// ```
    pub fn swap_cols(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match O::KIND {
            OrderKind::RowMajor => self.swap_minor_axis_vectors(m, n),
            OrderKind::ColMajor => self.swap_major_axis_vectors(m, n),
        }
    }

    fn swap_major_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.major() || n >= self.major() {
            return Err(Error::IndexOutOfBounds);
        } else if m == n || self.minor() == 0 {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let index = m * stride.major();
        let jndex = n * stride.major();

        unsafe {
            let x = base.add(index);
            let y = base.add(jndex);
            let count = self.minor() * stride.minor();
            ptr::swap_nonoverlapping(x, y, count);
        }

        Ok(self)
    }

    fn swap_minor_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.minor() || n >= self.minor() {
            return Err(Error::IndexOutOfBounds);
        } else if m == n || self.major() == 0 {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let index = m * stride.minor();
        let jndex = n * stride.minor();

        unsafe {
            let mut x = base.add(index);
            let mut y = base.add(jndex);
            ptr::swap_nonoverlapping(x, y, stride.minor());
            for _ in 1..self.major() {
                x = x.add(stride.major());
                y = y.add(stride.major());
                ptr::swap_nonoverlapping(x, y, stride.minor());
            }
        }

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Index, WrappingIndex};
    use crate::{dispatch_unary, matrix};

    #[test]
    fn test_swap() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = Index::new(0, 0);
            let jndex = Index::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = Index::new(0, 0);
            let jndex = Index::new(1, 1);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = Index::new(1, 1);
            let jndex = Index::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = Index::new(0, 0);
            let jndex = WrappingIndex::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = Index::new(0, 0);
            let jndex = WrappingIndex::new(1, 1);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = Index::new(1, 1);
            let jndex = WrappingIndex::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = WrappingIndex::new(0, 0);
            let jndex = Index::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = WrappingIndex::new(0, 0);
            let jndex = Index::new(1, 1);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = WrappingIndex::new(1, 1);
            let jndex = Index::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = WrappingIndex::new(0, 0);
            let jndex = WrappingIndex::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = WrappingIndex::new(0, 0);
            let jndex = WrappingIndex::new(1, 1);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let index = WrappingIndex::new(1, 1);
            let jndex = WrappingIndex::new(0, 0);
            matrix.swap(index, jndex).unwrap();
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let index = Index::new(0, 0);
            let jndex = Index::new(2, 2);
            let error = matrix.swap(index, jndex).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let index = Index::new(2, 2);
            let jndex = Index::new(0, 0);
            let error = matrix.swap(index, jndex).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let index = Index::new(2, 2);
            let jndex = Index::new(3, 3);
            let error = matrix.swap(index, jndex).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);
        }}
    }

    #[test]
    fn test_swap_rows() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_rows(0, 0).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_rows(0, 1).unwrap();
            let expected = matrix![[4, 5, 6], [1, 2, 3]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_rows(1, 0).unwrap();
            let expected = matrix![[4, 5, 6], [1, 2, 3]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let error = matrix.swap_rows(0, 2).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let error = matrix.swap_rows(2, 0).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let error = matrix.swap_rows(2, 3).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            // Asserts no undefined behavior.
            let mut matrix = matrix![[0; 0]; 2].with_order::<O>();
            matrix.swap_rows(0, 1).unwrap();
            let expected = matrix![[0; 0]; 2];
            assert_eq!(matrix, expected);
        }}
    }

    #[test]
    fn test_swap_cols() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_cols(0, 0).unwrap();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_cols(0, 1).unwrap();
            let expected = matrix![[2, 1, 3], [5, 4, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_cols(1, 0).unwrap();
            let expected = matrix![[2, 1, 3], [5, 4, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let error = matrix.swap_cols(0, 3).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let error = matrix.swap_cols(3, 0).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let error = matrix.swap_cols(3, 4).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            // Asserts no undefined behavior.
            let mut matrix = matrix![[0; 3]; 0].with_order::<O>();
            matrix.swap_cols(0, 1).unwrap();
            let expected = matrix![[0; 3]; 0];
            assert_eq!(matrix, expected);
        }}
    }
}
