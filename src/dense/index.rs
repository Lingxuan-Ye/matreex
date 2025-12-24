use super::Matrix;
use super::layout::{Order, OrderKind, Stride};
use crate::error::Result;
use crate::index::{AsIndex, Index, MatrixIndex, WrappingIndex};

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Returns a shared output at the given location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(matrix.get((1, 1)), Ok(&5));
    /// assert_eq!(matrix.get((2, 3)), Err(Error::IndexOutOfBounds));
    /// ```
    ///
    /// [`Error::IndexOutOfBounds`]: crate::error::Error::IndexOutOfBounds
    pub fn get<I>(&self, index: I) -> Result<I::Output<'_>>
    where
        I: MatrixIndex<Self>,
    {
        index.get(self)
    }

    /// Returns a mutable output at the given location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(matrix.get_mut((1, 1)), Ok(&mut 5));
    /// assert_eq!(matrix.get_mut((2, 3)), Err(Error::IndexOutOfBounds));
    /// ```
    ///
    /// [`Error::IndexOutOfBounds`]: crate::error::Error::IndexOutOfBounds
    pub fn get_mut<I>(&mut self, index: I) -> Result<I::OutputMut<'_>>
    where
        I: MatrixIndex<Self>,
    {
        index.get_mut(self)
    }

    /// Returns a shared output at the given location, without performing any
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// See the [`MatrixIndex`] implementation for the index type `I`.
    ///
    /// For a safe alternative see [`get`].
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
    pub unsafe fn get_unchecked<I>(&self, index: I) -> I::Output<'_>
    where
        I: MatrixIndex<Self>,
    {
        unsafe { index.get_unchecked(self) }
    }

    /// Returns a mutable output at the given location, without performing any
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// See the [`MatrixIndex`] implementation for the index type `I`.
    ///
    /// For a safe alternative see [`get_mut`].
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
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> I::OutputMut<'_>
    where
        I: MatrixIndex<Self>,
    {
        unsafe { index.get_unchecked_mut(self) }
    }
}

unsafe impl<T, O, I> MatrixIndex<Matrix<T, O>> for I
where
    O: Order,
    I: AsIndex,
{
    type Output<'a>
        = &'a T
    where
        Matrix<T, O>: 'a;

    type OutputMut<'a>
        = &'a mut T
    where
        Matrix<T, O>: 'a;

    fn is_out_of_bounds(&self, matrix: &Matrix<T, O>) -> bool {
        let shape = matrix.shape();
        self.row() >= shape.nrows() || self.col() >= shape.ncols()
    }

    unsafe fn get_unchecked(self, matrix: &Matrix<T, O>) -> Self::Output<'_> {
        let stride = matrix.stride();
        let index = Index::new(self.row(), self.col()).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked(index) }
    }

    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T, O>) -> Self::OutputMut<'_> {
        let stride = matrix.stride();
        let index = Index::new(self.row(), self.col()).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked_mut(index) }
    }
}

impl<T, O, I> core::ops::Index<I> for Matrix<T, O>
where
    O: Order,
    I: AsIndex,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        match self.get(index) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<T, O, I> core::ops::IndexMut<I> for Matrix<T, O>
where
    O: Order,
    I: AsIndex,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        match self.get_mut(index) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

unsafe impl<T, O> MatrixIndex<Matrix<T, O>> for WrappingIndex
where
    O: Order,
{
    type Output<'a>
        = &'a T
    where
        Matrix<T, O>: 'a;

    type OutputMut<'a>
        = &'a mut T
    where
        Matrix<T, O>: 'a;

    /// Returns `true` if the index is out of bounds for the given matrix.
    ///
    /// # Notes
    ///
    /// A wrapping index is out of bounds if and only if the matrix is empty.
    fn is_out_of_bounds(&self, matrix: &Matrix<T, O>) -> bool {
        matrix.is_empty()
    }

    /// Returns a shared output at this location, without performing any bounds
    /// checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// This method is safe, despite being marked `unsafe`. If no panic occurs,
    /// the returned output is guaranteed to be valid.
    unsafe fn get_unchecked(self, matrix: &Matrix<T, O>) -> Self::Output<'_> {
        let shape = matrix.shape();
        let stride = matrix.stride();
        let index = Index::from_wrapping_index(self, shape).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked(index) }
    }

    /// Returns a mutable output at this location, without performing any bounds
    /// checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// This method is safe, despite being marked `unsafe`. If no panic occurs,
    /// the returned output is guaranteed to be valid.
    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T, O>) -> Self::OutputMut<'_> {
        let shape = matrix.shape();
        let stride = matrix.stride();
        let index = Index::from_wrapping_index(self, shape).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked_mut(index) }
    }
}

impl<T, O> core::ops::Index<WrappingIndex> for Matrix<T, O>
where
    O: Order,
{
    type Output = T;

    fn index(&self, index: WrappingIndex) -> &Self::Output {
        match self.get(index) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<T, O> core::ops::IndexMut<WrappingIndex> for Matrix<T, O>
where
    O: Order,
{
    fn index_mut(&mut self, index: WrappingIndex) -> &mut Self::Output {
        match self.get_mut(index) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl Index {
    /// Creates a new [`Index`] from a flattened index.
    ///
    /// # Panics
    ///
    /// Panics if `stride.major() == 0`.
    pub(super) fn from_flattened<O>(index: usize, stride: Stride) -> Self
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

    /// Converts this index to a flattened index.
    pub(super) fn to_flattened<O>(self, stride: Stride) -> usize
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
    use super::super::layout::{ColMajor, RowMajor};
    use super::*;
    use crate::error::Error;
    use crate::{dispatch_unary, matrix};

    #[test]
    fn test_matrix_get() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            assert_eq!(matrix.get(Index::new(0, 0)), Ok(&1));
            assert_eq!(matrix.get(Index::new(0, 1)), Ok(&2));
            assert_eq!(matrix.get(Index::new(0, 2)), Ok(&3));
            assert_eq!(matrix.get(Index::new(1, 0)), Ok(&4));
            assert_eq!(matrix.get(Index::new(1, 1)), Ok(&5));
            assert_eq!(matrix.get(Index::new(1, 2)), Ok(&6));

            assert_eq!(matrix.get(Index::new(2, 0)), Err(Error::IndexOutOfBounds));
            assert_eq!(matrix.get(Index::new(0, 3)), Err(Error::IndexOutOfBounds));
            assert_eq!(matrix.get(Index::new(2, 3)), Err(Error::IndexOutOfBounds));
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

            assert_eq!(
                matrix.get_mut(Index::new(2, 0)),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                matrix.get_mut(Index::new(0, 3)),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                matrix.get_mut(Index::new(2, 3)),
                Err(Error::IndexOutOfBounds)
            );
        }}
    }

    #[test]
    fn test_matrix_get_unchecked() {
        dispatch_unary! {{
            unsafe {
                let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                assert_eq!(matrix.get_unchecked(Index::new(0, 0)), &1);
                assert_eq!(matrix.get_unchecked(Index::new(0, 1)), &2);
                assert_eq!(matrix.get_unchecked(Index::new(0, 2)), &3);
                assert_eq!(matrix.get_unchecked(Index::new(1, 0)), &4);
                assert_eq!(matrix.get_unchecked(Index::new(1, 1)), &5);
                assert_eq!(matrix.get_unchecked(Index::new(1, 2)), &6);
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
    fn test_as_index_ensure_in_bounds() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            assert!(Index::new(0, 0).ensure_in_bounds(&matrix).is_ok());
            assert!(Index::new(0, 1).ensure_in_bounds(&matrix).is_ok());
            assert!(Index::new(0, 2).ensure_in_bounds(&matrix).is_ok());
            assert!(Index::new(1, 0).ensure_in_bounds(&matrix).is_ok());
            assert!(Index::new(1, 1).ensure_in_bounds(&matrix).is_ok());
            assert!(Index::new(1, 2).ensure_in_bounds(&matrix).is_ok());

            assert_eq!(
                Index::new(2, 0).ensure_in_bounds(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                Index::new(0, 3).ensure_in_bounds(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                Index::new(2, 3).ensure_in_bounds(&matrix),
                Err(Error::IndexOutOfBounds)
            );
        }}
    }

    #[test]
    fn test_as_index_get() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

            assert_eq!(Index::new(0, 0).get(&matrix), Ok(&1));
            assert_eq!(Index::new(0, 1).get(&matrix), Ok(&2));
            assert_eq!(Index::new(0, 2).get(&matrix), Ok(&3));
            assert_eq!(Index::new(1, 0).get(&matrix), Ok(&4));
            assert_eq!(Index::new(1, 1).get(&matrix), Ok(&5));
            assert_eq!(Index::new(1, 2).get(&matrix), Ok(&6));

            assert_eq!(Index::new(2, 0).get(&matrix), Err(Error::IndexOutOfBounds));
            assert_eq!(Index::new(0, 3).get(&matrix), Err(Error::IndexOutOfBounds));
            assert_eq!(Index::new(2, 3).get(&matrix), Err(Error::IndexOutOfBounds));
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

            assert_eq!(Index::new(2, 0).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
            assert_eq!(Index::new(0, 3).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
            assert_eq!(Index::new(2, 3).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
        }}
    }

    #[test]
    fn test_as_index_get_unchecked() {
        dispatch_unary! {{
            unsafe {
                let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                assert_eq!(Index::new(0, 0).get_unchecked(&matrix), &1);
                assert_eq!(Index::new(0, 1).get_unchecked(&matrix), &2);
                assert_eq!(Index::new(0, 2).get_unchecked(&matrix), &3);
                assert_eq!(Index::new(1, 0).get_unchecked(&matrix), &4);
                assert_eq!(Index::new(1, 1).get_unchecked(&matrix), &5);
                assert_eq!(Index::new(1, 2).get_unchecked(&matrix), &6);
            }
        }}
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
    fn test_wrapping_index_ensure_in_bounds() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(
                        WrappingIndex::new(row, col)
                            .ensure_in_bounds(&matrix)
                            .is_ok()
                    );
                }
            }

            let matrix = Matrix::<i32, O>::new();
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert_eq!(
                        WrappingIndex::new(row, col).ensure_in_bounds(&matrix),
                        Err(Error::IndexOutOfBounds)
                    );
                }
            }
        }}
    }

    #[test]
    fn test_wrapping_index_get() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert_eq!(WrappingIndex::new(0, 0).get(&matrix), Ok(&1));
            assert_eq!(WrappingIndex::new(0, 1).get(&matrix), Ok(&2));
            assert_eq!(WrappingIndex::new(0, 2).get(&matrix), Ok(&3));
            assert_eq!(WrappingIndex::new(1, 0).get(&matrix), Ok(&4));
            assert_eq!(WrappingIndex::new(1, 1).get(&matrix), Ok(&5));
            assert_eq!(WrappingIndex::new(1, 2).get(&matrix), Ok(&6));
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert_eq!(WrappingIndex::new(row, col).get(&matrix), Ok(&1));
                }
            }

            let matrix = Matrix::<i32, O>::new();
            assert_eq!(
                WrappingIndex::new(0, 0).get(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(0, 1).get(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(0, 2).get(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(1, 0).get(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(1, 1).get(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(1, 2).get(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert_eq!(
                        WrappingIndex::new(row, col).get(&matrix),
                        Err(Error::IndexOutOfBounds)
                    );
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
            assert_eq!(
                WrappingIndex::new(0, 0).get_mut(&mut matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(0, 1).get_mut(&mut matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(0, 2).get_mut(&mut matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(1, 0).get_mut(&mut matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(1, 1).get_mut(&mut matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                WrappingIndex::new(1, 2).get_mut(&mut matrix),
                Err(Error::IndexOutOfBounds)
            );
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert_eq!(
                        WrappingIndex::new(row, col).get_mut(&mut matrix),
                        Err(Error::IndexOutOfBounds)
                    );
                }
            }
        }}
    }

    #[test]
    fn test_wrapping_index_get_unchecked() {
        dispatch_unary! {{
            unsafe {
                let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();

                assert_eq!(WrappingIndex::new(0, 0).get_unchecked(&matrix), &1);
                assert_eq!(WrappingIndex::new(0, 1).get_unchecked(&matrix), &2);
                assert_eq!(WrappingIndex::new(0, 2).get_unchecked(&matrix), &3);
                assert_eq!(WrappingIndex::new(1, 0).get_unchecked(&matrix), &4);
                assert_eq!(WrappingIndex::new(1, 1).get_unchecked(&matrix), &5);
                assert_eq!(WrappingIndex::new(1, 2).get_unchecked(&matrix), &6);
                for row in (-6..=6).step_by(2) {
                    for col in (-6..=6).step_by(3) {
                        assert_eq!(WrappingIndex::new(row, col).get_unchecked(&matrix), &1);
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
