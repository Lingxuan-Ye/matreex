use super::Matrix;
use super::layout::{Order, OrderKind, Stride};
use crate::error::Result;
use crate::index::{AsIndex, Index, MatrixIndex, WrappingIndex};

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Returns a reference to the [`MatrixIndex::Output`]
    /// at given location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    pub fn get<I>(&self, index: I) -> Result<&I::Output>
    where
        I: MatrixIndex<Self>,
    {
        index.get(self)
    }

    /// Returns a mutable reference to the [`MatrixIndex::Output`]
    /// at given location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    pub fn get_mut<I>(&mut self, index: I) -> Result<&mut I::Output>
    where
        I: MatrixIndex<Self>,
    {
        index.get_mut(self)
    }

    /// Returns a reference to the [`MatrixIndex::Output`]
    /// at given location, without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined
    /// behavior]* even if the resulting reference is not used.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`get`]: Matrix::get
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
    where
        I: MatrixIndex<Self>,
    {
        unsafe { index.get_unchecked(self) }
    }

    /// Returns a mutable reference to the [`MatrixIndex::Output`]
    /// at given location, without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined
    /// behavior]* even if the resulting reference is not used.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`get_mut`]: Matrix::get_mut
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
    where
        I: MatrixIndex<Self>,
    {
        unsafe { index.get_unchecked_mut(self) }
    }
}

impl<T, O, I> core::ops::Index<I> for Matrix<T, O>
where
    O: Order,
    I: MatrixIndex<Self>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        index.index(self)
    }
}

impl<T, O, I> core::ops::IndexMut<I> for Matrix<T, O>
where
    O: Order,
    I: MatrixIndex<Self>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        index.index_mut(self)
    }
}

unsafe impl<T, O, I> MatrixIndex<Matrix<T, O>> for I
where
    O: Order,
    I: AsIndex,
{
    type Output = T;

    fn is_out_of_bounds(&self, matrix: &Matrix<T, O>) -> bool {
        let shape = matrix.shape();
        self.row() >= shape.nrows() || self.col() >= shape.ncols()
    }

    unsafe fn get_unchecked(self, matrix: &Matrix<T, O>) -> &Self::Output {
        let stride = matrix.stride();
        let index = Index::from_as_index(self).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked(index) }
    }

    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T, O>) -> &mut Self::Output {
        let stride = matrix.stride();
        let index = Index::from_as_index(self).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked_mut(index) }
    }
}

unsafe impl<T, O> MatrixIndex<Matrix<T, O>> for WrappingIndex
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

    /// Returns a shared reference to the output at this location,
    /// without performing any bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// This method is safe, despite being marked `unsafe`. If no panic
    /// occurs, the output returned is guaranteed to be valid.
    unsafe fn get_unchecked(self, matrix: &Matrix<T, O>) -> &Self::Output {
        let shape = matrix.shape();
        let stride = matrix.stride();
        let index = Index::from_wrapping_index(self, shape).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked(index) }
    }

    /// Returns a mutable reference to the output at this location,
    /// without performing any bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// This method is safe, despite being marked `unsafe`. If no panic
    /// occurs, the output returned is guaranteed to be valid.
    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T, O>) -> &mut Self::Output {
        let shape = matrix.shape();
        let stride = matrix.stride();
        let index = Index::from_wrapping_index(self, shape).to_flattened::<O>(stride);
        unsafe { matrix.data.get_unchecked_mut(index) }
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
