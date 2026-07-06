use super::Matrix;
use super::index::MatrixIndex;
use super::order::{Order, OrderKind};
use crate::error::{Error, Result};
use core::ptr;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Swaps the elements at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if either index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.swap((0, 0), (1, 1))?;
    /// assert_eq!(matrix, matrix![[5, 2, 3], [4, 1, 6]]);
    /// #
    /// # Ok::<(), matreex::Error>(())
    /// ```
    pub fn swap<I, J>(&mut self, i: I, j: J) -> Result<&mut Self>
    where
        I: for<'a> MatrixIndex<T, O, Output = T>,
        J: for<'a> MatrixIndex<T, O, Output = T>,
    {
        if i.is_out_of_bounds(self) || j.is_out_of_bounds(self) {
            return Err(Error::IndexOutOfBounds);
        }

        unsafe {
            let x = i.get_unchecked_mut(self);
            let y = j.get_unchecked_mut(self);
            if x != y {
                ptr::swap_nonoverlapping(x, y, 1);
            }
        }

        Ok(self)
    }

    /// Swaps the rows at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if either index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.swap_rows(0, 1)?;
    /// assert_eq!(matrix, matrix![[4, 5, 6], [1, 2, 3]]);
    /// #
    /// # Ok::<(), matreex::Error>(())
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
    /// - [`Error::IndexOutOfBounds`] if either index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.swap_cols(0, 1)?;
    /// assert_eq!(matrix, matrix![[2, 1, 3], [5, 4, 6]]);
    /// #
    /// # Ok::<(), matreex::Error>(())
    /// ```
    pub fn swap_cols(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        match O::KIND {
            OrderKind::RowMajor => self.swap_minor_axis_vectors(m, n),
            OrderKind::ColMajor => self.swap_major_axis_vectors(m, n),
        }
    }

    /// Swaps the major-axis vectors at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if either index is out of bounds.
    fn swap_major_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.major() || n >= self.major() {
            return Err(Error::IndexOutOfBounds);
        } else if m == n || self.minor() == 0 {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let x = m * stride.major();
        let y = n * stride.major();

        unsafe {
            let x = base.add(x);
            let y = base.add(y);
            let count = self.minor() * stride.minor();
            ptr::swap_nonoverlapping(x, y, count);
        }

        Ok(self)
    }

    /// Swaps the minor-axis vectors at the given indices.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if either index is out of bounds.
    fn swap_minor_axis_vectors(&mut self, m: usize, n: usize) -> Result<&mut Self> {
        if m >= self.minor() || n >= self.minor() {
            return Err(Error::IndexOutOfBounds);
        } else if m == n || self.major() == 0 {
            return Ok(self);
        }

        let base = self.data.as_mut_ptr();
        let stride = self.stride();
        let x = m * stride.minor();
        let y = n * stride.minor();

        unsafe {
            let mut x = base.add(x);
            let mut y = base.add(y);
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
    fn test_swap() -> Result<()> {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = Index::new(0, 0);
            let j = Index::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = Index::new(1, 1);
            let j = Index::new(0, 0);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = Index::new(1, 1);
            let j = Index::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = Index::new(0, 0);
            let j = WrappingIndex::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = Index::new(1, 1);
            let j = WrappingIndex::new(0, 0);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = Index::new(1, 1);
            let j = WrappingIndex::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = WrappingIndex::new(0, 0);
            let j = Index::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = WrappingIndex::new(1, 1);
            let j = Index::new(0, 0);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = WrappingIndex::new(1, 1);
            let j = Index::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = WrappingIndex::new(0, 0);
            let j = WrappingIndex::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = WrappingIndex::new(1, 1);
            let j = WrappingIndex::new(0, 0);
            matrix.swap(i, j)?;
            let expected = matrix![[5, 2, 3], [4, 1, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let i = WrappingIndex::new(1, 1);
            let j = WrappingIndex::new(1, 1);
            matrix.swap(i, j)?;
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let i = Index::new(0, 0);
            let j = Index::new(2, 2);
            let error = matrix.swap(i, j).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let i = Index::new(2, 2);
            let j = Index::new(0, 0);
            let error = matrix.swap(i, j).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let unchanged = matrix.clone();
            let i = Index::new(2, 2);
            let j = Index::new(3, 3);
            let error = matrix.swap(i, j).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
            assert_eq!(matrix, unchanged);
        }}

        #[cfg(miri)]
        {
            extern crate std;

            use alloc::boxed::Box;
            use alloc::vec::Vec;
            use core::cell::RefCell;
            use std::thread_local;

            // Use thread-local storage here because raw pointers are not `Send`.
            thread_local! {
                // Leaked memory will be automatically reclaimed once the test process exits,
                // but we free it manually only to silence Miri's leak detection.
                static LEAKED: RefCell<Vec<*mut u8>> = const { RefCell::new(Vec::new()) };
            }

            struct I(Index);

            unsafe impl<O> MatrixIndex<u8, O> for I
            where
                O: Order,
            {
                type Output = u8;

                fn is_out_of_bounds(&self, matrix: &Matrix<u8, O>) -> bool {
                    self.0.is_out_of_bounds(matrix)
                }

                unsafe fn get_unchecked(self, _: *const Matrix<u8, O>) -> *const Self::Output {
                    unimplemented!()
                }

                unsafe fn get_unchecked_mut(self, _: *mut Matrix<u8, O>) -> *mut Self::Output {
                    LEAKED.with(|cell| {
                        let ptr = Box::into_raw(Box::new(0));
                        cell.borrow_mut().push(ptr);
                        ptr
                    })
                }
            }

            dispatch_unary! {{
                let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
                let i = I(Index::new(0, 0));
                let j = I(Index::new(1, 1));
                matrix.swap(i, j)?;
            }}

            let _ = LEAKED.try_with(|cell| {
                for &ptr in cell.borrow().iter() {
                    let _ = unsafe { Box::from_raw(ptr) };
                }
            });
        }

        Ok(())
    }

    #[test]
    fn test_swap_rows() -> Result<()> {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_rows(0, 0)?;
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_rows(0, 1)?;
            let expected = matrix![[4, 5, 6], [1, 2, 3]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_rows(1, 0)?;
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
            matrix.swap_rows(0, 1)?;
            let expected = matrix![[0; 0]; 2];
            assert_eq!(matrix, expected);
        }}

        Ok(())
    }

    #[test]
    fn test_swap_cols() -> Result<()> {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_cols(0, 0)?;
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_cols(0, 1)?;
            let expected = matrix![[2, 1, 3], [5, 4, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.swap_cols(1, 0)?;
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
            matrix.swap_cols(0, 1)?;
            let expected = matrix![[0; 3]; 0];
            assert_eq!(matrix, expected);
        }}

        Ok(())
    }
}
