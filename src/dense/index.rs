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
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
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
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
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
    /// Refer to the [`MatrixIndex`] implementation for the index type `I`.
    ///
    /// For a safe alternative see [`get`].
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
    /// Refer to the [`MatrixIndex`] implementation for the index type `I`.
    ///
    /// For a safe alternative see [`get_mut`].
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
    /// the output returned is guaranteed to be valid.
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
    /// the output returned is guaranteed to be valid.
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
