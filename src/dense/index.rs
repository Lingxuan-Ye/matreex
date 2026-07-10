//! Matrix indexing operations.

use super::Matrix;
use super::layout::Stride;
use super::order::{Order, OrderKind};
use crate::error::Error;
use crate::index::{AsIndex, Index, WrappingIndex};

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Returns a shared reference to the output at this location, if in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(matrix.get((1, 1)), Some(&5));
    /// assert_eq!(matrix.get((2, 3)), None);
    /// ```
    pub fn get<I>(&self, index: I) -> Option<&I::Output>
    where
        I: MatrixIndex<T, O>,
    {
        index.get(self)
    }

    /// Returns a mutable reference to the output at this location, if in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(matrix.get_mut((1, 1)), Some(&mut 5));
    /// assert_eq!(matrix.get_mut((2, 3)), None);
    /// ```
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut I::Output>
    where
        I: MatrixIndex<T, O>,
    {
        index.get_mut(self)
    }

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Panics
    ///
    /// Panic behavior is implementation-specific and documented by each
    /// implementation of [`MatrixIndex<T, O>`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(unsafe { matrix.get_unchecked((1, 1)) }, &5);
    /// ```
    ///
    /// [`get`]: Matrix::get
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
    where
        I: MatrixIndex<T, O>,
    {
        unsafe { &*index.get_unchecked(self) }
    }

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// # Panics
    ///
    /// Panic behavior is implementation-specific and documented by each
    /// implementation of [`MatrixIndex<T, O>`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(unsafe { matrix.get_unchecked_mut((1, 1)) }, &mut 5);
    /// ```
    ///
    /// [`get_mut`]: Matrix::get_mut
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
    where
        I: MatrixIndex<T, O>,
    {
        unsafe { &mut *index.get_unchecked_mut(self) }
    }
}

impl<T, O, I> core::ops::Index<I> for Matrix<T, O>
where
    O: Order,
    I: MatrixIndex<T, O, Output = T>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        match self.get(index) {
            None => panic!("{}", Error::IndexOutOfBounds),
            Some(output) => output,
        }
    }
}

impl<T, O, I> core::ops::IndexMut<I> for Matrix<T, O>
where
    O: Order,
    I: MatrixIndex<T, O, Output = T>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        match self.get_mut(index) {
            None => panic!("{}", Error::IndexOutOfBounds),
            Some(output) => output,
        }
    }
}

/// A helper trait for dense matrix indexing operations.
///
/// # Safety
///
/// Implementations of this trait have to promise that:
///
/// - If the argument to [`get_unchecked`] or [`get_unchecked_mut`] is a safe
///   reference, then so is the result.
///
/// - If any default implementation of [`get`] or [`get_mut`] is used, then
///   [`is_out_of_bounds`] is implemented correctly.
///
/// [`is_out_of_bounds`]: MatrixIndex::is_out_of_bounds
/// [`get`]: MatrixIndex::get
/// [`get_mut`]: MatrixIndex::get_mut
/// [`get_unchecked`]: MatrixIndex::get_unchecked
/// [`get_unchecked_mut`]: MatrixIndex::get_unchecked_mut
pub unsafe trait MatrixIndex<T, O>: Sized
where
    O: Order,
{
    /// The output type returned by methods.
    type Output;

    /// Returns `true` if the index is out of bounds for the given matrix.
    fn is_out_of_bounds(&self, matrix: &Matrix<T, O>) -> bool;

    /// Returns a shared reference to the output at this location, if in bounds.
    fn get(self, matrix: &Matrix<T, O>) -> Option<&Self::Output> {
        if self.is_out_of_bounds(matrix) {
            return None;
        }
        unsafe { Some(&*self.get_unchecked(matrix)) }
    }

    /// Returns a mutable reference to the output at this location, if in bounds.
    fn get_mut(self, matrix: &mut Matrix<T, O>) -> Option<&mut Self::Output> {
        if self.is_out_of_bounds(matrix) {
            return None;
        }
        unsafe { Some(&mut *self.get_unchecked_mut(matrix)) }
    }

    /// Returns a pointer to the output at this location, without performing any
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling matrix pointer
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: *const Matrix<T, O>) -> *const Self::Output;

    /// Returns a mutable pointer to the output at this location, without performing
    /// any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling matrix pointer
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<T, O>) -> *mut Self::Output;
}

unsafe impl<T, O, I> MatrixIndex<T, O> for I
where
    O: Order,
    I: AsIndex,
{
    type Output = T;

    fn is_out_of_bounds(&self, matrix: &Matrix<T, O>) -> bool {
        let shape = matrix.shape();
        self.row() >= shape.nrows || self.col() >= shape.ncols
    }

    /// Returns a pointer to the output at this location, without performing any
    /// bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if overflow occurs when linearizing the index.
    ///
    /// If the index is in bounds, this method will never panic.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling matrix pointer
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: *const Matrix<T, O>) -> *const Self::Output {
        let matrix = unsafe { &*matrix };
        let stride = matrix.stride();
        let index = Index::new(self.row(), self.col()).to_linear::<O>(stride);
        unsafe { matrix.data.as_ptr().add(index) }
    }

    /// Returns a mutable pointer to the output at this location, without performing
    /// any bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if overflow occurs when linearizing the index.
    ///
    /// If the index is in bounds, this method will never panic.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling matrix pointer
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<T, O>) -> *mut Self::Output {
        let matrix = unsafe { &mut *matrix };
        let stride = matrix.stride();
        let index = Index::new(self.row(), self.col()).to_linear::<O>(stride);
        unsafe { matrix.data.as_mut_ptr().add(index) }
    }
}

unsafe impl<T, O> MatrixIndex<T, O> for WrappingIndex
where
    O: Order,
{
    type Output = T;

    /// Returns `true` if the index is out of bounds for the given matrix.
    ///
    /// # Notes
    ///
    /// A wrapping index is out of bounds if and only if the matrix is empty.
    fn is_out_of_bounds(&self, matrix: &Matrix<T, O>) -> bool {
        matrix.is_empty()
    }

    /// Returns a pointer to the output at this location, without performing any
    /// bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// Calling this method with a dangling matrix pointer is *[undefined behavior]*
    /// even if the resulting pointer is not used.
    ///
    /// If the matrix pointer is valid, this method is safe. An out-of-bounds index
    /// only results in a panic.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: *const Matrix<T, O>) -> *const Self::Output {
        let matrix = unsafe { &*matrix };
        let shape = matrix.shape();
        let index = Index::from_wrapping_index(self, shape);
        unsafe { index.get_unchecked(matrix) }
    }

    /// Returns a mutable pointer to the output at this location, without performing
    /// any bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// Calling this method with a dangling matrix pointer is *[undefined behavior]*
    /// even if the resulting pointer is not used.
    ///
    /// If the matrix pointer is valid, this method is safe. An out-of-bounds index
    /// only results in a panic.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<T, O>) -> *mut Self::Output {
        let matrix = unsafe { &mut *matrix };
        let shape = matrix.shape();
        let index = Index::from_wrapping_index(self, shape);
        unsafe { index.get_unchecked_mut(matrix) }
    }
}

impl Index {
    /// Constructs a new [`Index`] from a linear index.
    ///
    /// # Panics
    ///
    /// Panics if `stride.major() == 0`.
    pub(super) const fn from_linear<O>(index: usize, stride: Stride) -> Self
    where
        O: Order,
    {
        let major = index / stride.major();
        let minor = (index % stride.major()) / stride.minor();
        match O::KIND {
            OrderKind::RowMajor => Self::new(major, minor),
            OrderKind::ColMajor => Self::new(minor, major),
        }
    }

    /// Converts this index to a linear index.
    ///
    /// # Panics
    ///
    /// Panics if overflow occurs.
    pub(super) const fn to_linear<O>(self, stride: Stride) -> usize
    where
        O: Order,
    {
        let (major, minor) = match O::KIND {
            OrderKind::RowMajor => (self.row, self.col),
            OrderKind::ColMajor => (self.col, self.row),
        };
        major * stride.major() + minor * stride.minor()
    }
}

#[cfg(test)]
mod tests {
    use super::super::order::{ColMajor, RowMajor};
    use super::*;
    use crate::{dispatch_unary, matrix};

    #[test]
    fn test_matrix_get() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            assert!(matrix.get(Index::new(0, 0)).is_some_and(|element| *element == 1));
            assert!(matrix.get(Index::new(0, 1)).is_some_and(|element| *element == 2));
            assert!(matrix.get(Index::new(0, 2)).is_some_and(|element| *element == 3));
            assert!(matrix.get(Index::new(1, 0)).is_some_and(|element| *element == 4));
            assert!(matrix.get(Index::new(1, 1)).is_some_and(|element| *element == 5));
            assert!(matrix.get(Index::new(1, 2)).is_some_and(|element| *element == 6));

            assert!(matrix.get(Index::new(2, 0)).is_none());
            assert!(matrix.get(Index::new(0, 3)).is_none());
            assert!(matrix.get(Index::new(2, 3)).is_none());
        }}
    }

    #[test]
    fn test_matrix_get_mut() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            *matrix.get_mut(Index::new(0, 0)).unwrap() -= 1;
            *matrix.get_mut(Index::new(0, 1)).unwrap() -= 2;
            *matrix.get_mut(Index::new(0, 2)).unwrap() -= 3;
            *matrix.get_mut(Index::new(1, 0)).unwrap() -= 4;
            *matrix.get_mut(Index::new(1, 1)).unwrap() -= 5;
            *matrix.get_mut(Index::new(1, 2)).unwrap() -= 6;

            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            assert_eq!(matrix, expected);

            assert!(matrix.get_mut(Index::new(2, 0)).is_none());
            assert!(matrix.get_mut(Index::new(0, 3)).is_none());
            assert!(matrix.get_mut(Index::new(2, 3)).is_none());
        }}
    }

    #[test]
    fn test_matrix_get_unchecked() {
        dispatch_unary! {{
            unsafe {
                let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                assert_eq!(*matrix.get_unchecked(Index::new(0, 0)), 1);
                assert_eq!(*matrix.get_unchecked(Index::new(0, 1)), 2);
                assert_eq!(*matrix.get_unchecked(Index::new(0, 2)), 3);
                assert_eq!(*matrix.get_unchecked(Index::new(1, 0)), 4);
                assert_eq!(*matrix.get_unchecked(Index::new(1, 1)), 5);
                assert_eq!(*matrix.get_unchecked(Index::new(1, 2)), 6);
            }
        }}
    }

    #[test]
    fn test_matrix_get_unchecked_mut() {
        dispatch_unary! {{
            unsafe {
                let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                *matrix.get_unchecked_mut(Index::new(0, 0)) -= 1;
                *matrix.get_unchecked_mut(Index::new(0, 1)) -= 2;
                *matrix.get_unchecked_mut(Index::new(0, 2)) -= 3;
                *matrix.get_unchecked_mut(Index::new(1, 0)) -= 4;
                *matrix.get_unchecked_mut(Index::new(1, 1)) -= 5;
                *matrix.get_unchecked_mut(Index::new(1, 2)) -= 6;

                let expected = matrix![[0, 0, 0], [0, 0, 0]];
                assert_eq!(matrix, expected);
            }
        }}
    }

    #[test]
    fn test_matrix_index() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            assert_eq!(matrix[Index::new(0, 0)], 1);
            assert_eq!(matrix[Index::new(0, 1)], 2);
            assert_eq!(matrix[Index::new(0, 2)], 3);
            assert_eq!(matrix[Index::new(1, 0)], 4);
            assert_eq!(matrix[Index::new(1, 1)], 5);
            assert_eq!(matrix[Index::new(1, 2)], 6);
        }}
    }

    #[test]
    #[should_panic]
    fn test_matrix_index_out_of_bounds_row_major() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        let _ = matrix[Index::new(2, 3)];
    }

    #[test]
    #[should_panic]
    fn test_matrix_index_out_of_bounds_col_major() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<ColMajor>();
        let _ = matrix[Index::new(2, 3)];
    }

    #[test]
    fn test_matrix_index_mut() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            matrix[Index::new(0, 0)] -= 1;
            matrix[Index::new(0, 1)] -= 2;
            matrix[Index::new(0, 2)] -= 3;
            matrix[Index::new(1, 0)] -= 4;
            matrix[Index::new(1, 1)] -= 5;
            matrix[Index::new(1, 2)] -= 6;

            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            assert_eq!(matrix, expected);
        }}
    }

    #[test]
    #[should_panic]
    fn test_matrix_index_mut_out_of_bounds_row_major() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        matrix[Index::new(2, 3)] += 2;
    }

    #[test]
    #[should_panic]
    fn test_matrix_index_mut_out_of_bounds_col_major() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<ColMajor>();
        matrix[Index::new(2, 3)] += 2;
    }

    #[test]
    fn test_as_index_is_out_of_bounds() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            assert!(!Index::new(0, 0).is_out_of_bounds(&matrix));
            assert!(!Index::new(0, 1).is_out_of_bounds(&matrix));
            assert!(!Index::new(0, 2).is_out_of_bounds(&matrix));
            assert!(!Index::new(1, 0).is_out_of_bounds(&matrix));
            assert!(!Index::new(1, 1).is_out_of_bounds(&matrix));
            assert!(!Index::new(1, 2).is_out_of_bounds(&matrix));

            assert!(Index::new(2, 0).is_out_of_bounds(&matrix));
            assert!(Index::new(0, 3).is_out_of_bounds(&matrix));
            assert!(Index::new(2, 3).is_out_of_bounds(&matrix));
        }}
    }

    #[test]
    fn test_as_index_get() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            assert!(Index::new(0, 0).get(&matrix).is_some_and(|element| *element ==  1));
            assert!(Index::new(0, 1).get(&matrix).is_some_and(|element| *element ==  2));
            assert!(Index::new(0, 2).get(&matrix).is_some_and(|element| *element ==  3));
            assert!(Index::new(1, 0).get(&matrix).is_some_and(|element| *element ==  4));
            assert!(Index::new(1, 1).get(&matrix).is_some_and(|element| *element ==  5));
            assert!(Index::new(1, 2).get(&matrix).is_some_and(|element| *element ==  6));

            assert!(Index::new(2, 0).get(&matrix).is_none());
            assert!(Index::new(0, 3).get(&matrix).is_none());
            assert!(Index::new(2, 3).get(&matrix).is_none());
        }}
    }

    #[test]
    fn test_as_index_get_mut() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            *Index::new(0, 0).get_mut(&mut matrix).unwrap() -= 1;
            *Index::new(0, 1).get_mut(&mut matrix).unwrap() -= 2;
            *Index::new(0, 2).get_mut(&mut matrix).unwrap() -= 3;
            *Index::new(1, 0).get_mut(&mut matrix).unwrap() -= 4;
            *Index::new(1, 1).get_mut(&mut matrix).unwrap() -= 5;
            *Index::new(1, 2).get_mut(&mut matrix).unwrap() -= 6;

            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            assert_eq!(matrix, expected);

            assert!(Index::new(2, 0).get_mut(&mut matrix).is_none());
            assert!(Index::new(0, 3).get_mut(&mut matrix).is_none());
            assert!(Index::new(2, 3).get_mut(&mut matrix).is_none());
        }}
    }

    #[test]
    fn test_as_index_get_unchecked() {
        dispatch_unary! {{
            unsafe {
                let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                assert_eq!(*Index::new(0, 0).get_unchecked(&matrix), 1);
                assert_eq!(*Index::new(0, 1).get_unchecked(&matrix), 2);
                assert_eq!(*Index::new(0, 2).get_unchecked(&matrix), 3);
                assert_eq!(*Index::new(1, 0).get_unchecked(&matrix), 4);
                assert_eq!(*Index::new(1, 1).get_unchecked(&matrix), 5);
                assert_eq!(*Index::new(1, 2).get_unchecked(&matrix), 6);
            }
        }}
    }

    #[test]
    #[should_panic]
    fn test_as_index_get_unchecked_fails_row_major() {
        unsafe {
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
            Index::new(usize::MAX, usize::MAX).get_unchecked(&matrix);
        }
    }

    #[test]
    #[should_panic]
    fn test_as_index_get_unchecked_fails_col_major() {
        unsafe {
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<ColMajor>();
            Index::new(usize::MAX, usize::MAX).get_unchecked(&matrix);
        }
    }

    #[test]
    fn test_as_index_get_unchecked_mut() {
        dispatch_unary! {{
            unsafe {
                let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                *Index::new(0, 0).get_unchecked_mut(&mut matrix) -= 1;
                *Index::new(0, 1).get_unchecked_mut(&mut matrix) -= 2;
                *Index::new(0, 2).get_unchecked_mut(&mut matrix) -= 3;
                *Index::new(1, 0).get_unchecked_mut(&mut matrix) -= 4;
                *Index::new(1, 1).get_unchecked_mut(&mut matrix) -= 5;
                *Index::new(1, 2).get_unchecked_mut(&mut matrix) -= 6;

                let expected = matrix![[0, 0, 0], [0, 0, 0]];
                assert_eq!(matrix, expected);
            }
        }}
    }

    #[test]
    #[should_panic]
    fn test_as_index_get_unchecked_mut_fails_row_major() {
        unsafe {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
            Index::new(usize::MAX, usize::MAX).get_unchecked_mut(&mut matrix);
        }
    }

    #[test]
    #[should_panic]
    fn test_as_index_get_unchecked_mut_fails_col_major() {
        unsafe {
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<ColMajor>();
            Index::new(usize::MAX, usize::MAX).get_unchecked_mut(&mut matrix);
        }
    }

    #[test]
    fn test_wrapping_index_is_out_of_bounds() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(!WrappingIndex::new(row, col).is_out_of_bounds(&matrix));
                }
            }

            let matrix = Matrix::<i32, O>::new();
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(WrappingIndex::new(row, col).is_out_of_bounds(&matrix));
                }
            }
        }}
    }

    #[test]
    fn test_wrapping_index_get() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert!(
                WrappingIndex::new(0, 0)
                    .get(&matrix)
                    .is_some_and(|element| *element == 1)
            );
            assert!(
                WrappingIndex::new(0, 1)
                    .get(&matrix)
                    .is_some_and(|element| *element == 2)
            );
            assert!(
                WrappingIndex::new(0, 2)
                    .get(&matrix)
                    .is_some_and(|element| *element == 3)
            );
            assert!(
                WrappingIndex::new(1, 0)
                    .get(&matrix)
                    .is_some_and(|element| *element == 4)
            );
            assert!(
                WrappingIndex::new(1, 1)
                    .get(&matrix)
                    .is_some_and(|element| *element == 5)
            );
            assert!(
                WrappingIndex::new(1, 2)
                    .get(&matrix)
                    .is_some_and(|element| *element == 6)
            );
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(
                        WrappingIndex::new(row, col)
                            .get(&matrix)
                            .is_some_and(|element| *element == 1)
                    );
                }
            }

            let matrix = Matrix::<i32, O>::new();
            assert!(WrappingIndex::new(0, 0).get(&matrix).is_none());
            assert!(WrappingIndex::new(0, 1).get(&matrix).is_none());
            assert!(WrappingIndex::new(0, 2).get(&matrix).is_none());
            assert!(WrappingIndex::new(1, 0).get(&matrix).is_none());
            assert!(WrappingIndex::new(1, 1).get(&matrix).is_none());
            assert!(WrappingIndex::new(1, 2).get(&matrix).is_none());
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(WrappingIndex::new(row, col).get(&matrix).is_none());
                }
            }
        }}
    }

    #[test]
    fn test_wrapping_index_get_mut() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            *WrappingIndex::new(0, 0).get_mut(&mut matrix).unwrap() -= 1;
            *WrappingIndex::new(0, 1).get_mut(&mut matrix).unwrap() -= 2;
            *WrappingIndex::new(0, 2).get_mut(&mut matrix).unwrap() -= 3;
            *WrappingIndex::new(1, 0).get_mut(&mut matrix).unwrap() -= 4;
            *WrappingIndex::new(1, 1).get_mut(&mut matrix).unwrap() -= 5;
            *WrappingIndex::new(1, 2).get_mut(&mut matrix).unwrap() -= 6;
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    *WrappingIndex::new(row, col).get_mut(&mut matrix).unwrap() -= 1;
                }
            }
            let expected = matrix![[-35, 0, 0], [0, 0, 0]];
            assert_eq!(matrix, expected);

            let mut matrix = Matrix::<i32, O>::new();
            assert!(WrappingIndex::new(0, 0).get_mut(&mut matrix).is_none());
            assert!(WrappingIndex::new(0, 1).get_mut(&mut matrix).is_none());
            assert!(WrappingIndex::new(0, 2).get_mut(&mut matrix).is_none());
            assert!(WrappingIndex::new(1, 0).get_mut(&mut matrix).is_none());
            assert!(WrappingIndex::new(1, 1).get_mut(&mut matrix).is_none());
            assert!(WrappingIndex::new(1, 2).get_mut(&mut matrix).is_none());
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(WrappingIndex::new(row, col).get_mut(&mut matrix).is_none());
                }
            }
        }}
    }

    #[test]
    fn test_wrapping_index_get_unchecked() {
        dispatch_unary! {{
            unsafe {
                let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                assert_eq!(*WrappingIndex::new(0, 0).get_unchecked(&matrix), 1);
                assert_eq!(*WrappingIndex::new(0, 1).get_unchecked(&matrix), 2);
                assert_eq!(*WrappingIndex::new(0, 2).get_unchecked(&matrix), 3);
                assert_eq!(*WrappingIndex::new(1, 0).get_unchecked(&matrix), 4);
                assert_eq!(*WrappingIndex::new(1, 1).get_unchecked(&matrix), 5);
                assert_eq!(*WrappingIndex::new(1, 2).get_unchecked(&matrix), 6);

                for row in (-6..=6).step_by(2) {
                    for col in (-6..=6).step_by(3) {
                        assert_eq!(*WrappingIndex::new(row, col).get_unchecked(&matrix), 1);
                    }
                }
            }
        }}
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_get_unchecked_fails_row_major() {
        unsafe {
            let matrix = Matrix::<i32, RowMajor>::new();
            WrappingIndex::new(0, 0).get_unchecked(&matrix);
        }
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_get_unchecked_fails_col_major() {
        unsafe {
            let matrix = Matrix::<i32, ColMajor>::new();
            WrappingIndex::new(0, 0).get_unchecked(&matrix);
        }
    }

    #[test]
    fn test_wrapping_index_get_unchecked_mut() {
        dispatch_unary! {{
            unsafe {
                let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                *WrappingIndex::new(0, 0).get_unchecked_mut(&mut matrix) -= 1;
                *WrappingIndex::new(0, 1).get_unchecked_mut(&mut matrix) -= 2;
                *WrappingIndex::new(0, 2).get_unchecked_mut(&mut matrix) -= 3;
                *WrappingIndex::new(1, 0).get_unchecked_mut(&mut matrix) -= 4;
                *WrappingIndex::new(1, 1).get_unchecked_mut(&mut matrix) -= 5;
                *WrappingIndex::new(1, 2).get_unchecked_mut(&mut matrix) -= 6;
                for row in (-6..=6).step_by(2) {
                    for col in (-6..=6).step_by(3) {
                        *WrappingIndex::new(row, col).get_unchecked_mut(&mut matrix) -= 1;
                    }
                }
                let expected = matrix![[-35, 0, 0], [0, 0, 0]];
                assert_eq!(matrix, expected);
            }
        }}
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_get_unchecked_mut_fails_row_major() {
        unsafe {
            let mut matrix = Matrix::<i32, RowMajor>::new();
            *WrappingIndex::new(0, 0).get_unchecked_mut(&mut matrix) += 2;
        }
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_get_unchecked_mut_fails_col_major() {
        unsafe {
            let mut matrix = Matrix::<i32, ColMajor>::new();
            *WrappingIndex::new(0, 0).get_unchecked_mut(&mut matrix) += 2;
        }
    }
}
