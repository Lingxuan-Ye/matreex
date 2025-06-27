//! Defines indexing operations.

use crate::Matrix;
use crate::error::{Error, Result};
use crate::order::Order;
use crate::shape::{AxisShape, Stride};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

impl<T> Matrix<T> {
    /// Returns a reference to the [`MatrixIndex::Output`]
    /// at given location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
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
    #[inline]
    pub fn get<I>(&self, index: I) -> Result<&I::Output>
    where
        I: MatrixIndex<T>,
    {
        index.get(self)
    }

    /// Returns a mutable reference to the [`MatrixIndex::Output`]
    /// at given location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
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
    #[inline]
    pub fn get_mut<I>(&mut self, index: I) -> Result<&mut I::Output>
    where
        I: MatrixIndex<T>,
    {
        index.get_mut(self)
    }

    /// Returns a reference to the [`MatrixIndex::Output`]
    /// at given location, without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
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
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`get`]: Matrix::get
    #[inline]
    pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
    where
        I: MatrixIndex<T>,
    {
        unsafe { index.get_unchecked(self) }
    }

    /// Returns a mutable reference to the [`MatrixIndex::Output`]
    /// at given location, without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
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
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`get_mut`]: Matrix::get_mut
    #[inline]
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
    where
        I: MatrixIndex<T>,
    {
        unsafe { index.get_unchecked_mut(self) }
    }
}

impl<T, I> core::ops::Index<I> for Matrix<T>
where
    I: MatrixIndex<T>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        index.index(self)
    }
}

impl<T, I> core::ops::IndexMut<I> for Matrix<T>
where
    I: MatrixIndex<T>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        index.index_mut(self)
    }
}

/// A helper trait for indexing operations on a [`Matrix<T>`].
///
/// This trait is a poor imitation of [`SliceIndex`].
///
/// # Safety
///
/// Implementations of this trait have to promise that if any default
/// implementations of [`get`], [`get_mut`], [`index`] or [`index_mut`]
/// are used, then [`is_out_of_bounds`] is implemented correctly and
/// [`ensure_in_bounds`] is not overridden. Failing to do so may result
/// in an out-of-bounds memory access, leading to *[undefined behavior]*.
///
/// [`SliceIndex`]: core::slice::SliceIndex
/// [`is_out_of_bounds`]: MatrixIndex::is_out_of_bounds
/// [`ensure_in_bounds`]: MatrixIndex::ensure_in_bounds
/// [`get`]: MatrixIndex::get
/// [`get_mut`]: MatrixIndex::get_mut
/// [`index`]: MatrixIndex::index
/// [`index_mut`]: MatrixIndex::index_mut
/// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
pub unsafe trait MatrixIndex<T>: Sized + internal::Sealed {
    /// The output type returned by methods.
    type Output: ?Sized;

    /// Returns `true` if the index is out of bounds for the given matrix.
    fn is_out_of_bounds(&self, matrix: &Matrix<T>) -> bool;

    /// Ensures the index is in bounds for the given matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn ensure_in_bounds(&self, matrix: &Matrix<T>) -> Result<&Self> {
        if self.is_out_of_bounds(matrix) {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self)
        }
    }

    /// Returns a shared reference to the output at this location, if in
    /// bounds.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output> {
        self.ensure_in_bounds(matrix)?;
        unsafe { Ok(self.get_unchecked(matrix)) }
    }

    /// Returns a mutable reference to the output at this location, if in
    /// bounds.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        self.ensure_in_bounds(matrix)?;
        unsafe { Ok(self.get_unchecked_mut(matrix)) }
    }

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output;

    /// Returns a shared reference to the output at this location.
    ///
    /// # Panics
    ///
    /// Panics if out of bounds.
    fn index(self, matrix: &Matrix<T>) -> &Self::Output {
        match self.get(matrix) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }

    /// Returns a mutable reference to the output at this location.
    ///
    /// # Panics
    ///
    /// Panics if out of bounds.
    fn index_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        match self.get_mut(matrix) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

/// A struct representing the index of an element in a [`Matrix<T>`].
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Index {
    /// The row index.
    pub row: usize,

    /// The column index.
    pub col: usize,
}

impl Index {
    /// Creates a new [`Index`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Index;
    ///
    /// let index = Index::new(2, 3);
    /// assert_eq!(index.row, 2);
    /// assert_eq!(index.col, 3);
    /// ```
    #[inline]
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    /// Swaps the row and column indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Index;
    ///
    /// let mut index = Index::new(2, 3);
    /// index.swap();
    /// assert_eq!(index, Index::new(3, 2));
    /// ```
    #[inline]
    pub fn swap(&mut self) -> &mut Self {
        (self.row, self.col) = (self.col, self.row);
        self
    }

    pub(crate) fn from_flattened(index: usize, order: Order, stride: Stride) -> Self {
        AxisIndex::from_flattened(index, stride).to_index(order)
    }

    pub(crate) fn to_flattened(self, order: Order, stride: Stride) -> usize {
        AxisIndex::from_index(self, order).to_flattened(stride)
    }
}

impl From<(usize, usize)> for Index {
    #[inline]
    fn from(value: (usize, usize)) -> Self {
        let (row, col) = value;
        Self { row, col }
    }
}

impl From<[usize; 2]> for Index {
    #[inline]
    fn from(value: [usize; 2]) -> Self {
        let [row, col] = value;
        Self { row, col }
    }
}

/// A trait for single-element indexing operations on a [`Matrix<T>`].
///
/// # Examples
///
/// ```
/// use matreex::index::AsIndex;
/// use matreex::matrix;
///
/// struct I(usize, usize);
///
/// impl AsIndex for I {
///     fn row(&self) -> usize {
///         self.0
///     }
///
///     fn col(&self) -> usize {
///         self.1
///     }
/// }
///
/// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
/// assert_eq!(matrix.get(I(1, 1)), Ok(&5));
/// ```
pub trait AsIndex {
    /// Returns the row index.
    fn row(&self) -> usize;

    /// Returns the column index.
    fn col(&self) -> usize;
}

unsafe impl<T, I> MatrixIndex<T> for I
where
    I: AsIndex,
{
    type Output = T;

    #[inline]
    fn is_out_of_bounds(&self, matrix: &Matrix<T>) -> bool {
        let shape = matrix.shape();
        self.row() >= shape.nrows() || self.col() >= shape.ncols()
    }

    #[inline]
    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output> {
        let index = AxisIndex::from_index(self, matrix.order);
        index.get(matrix)
    }

    #[inline]
    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        let index = AxisIndex::from_index(self, matrix.order);
        index.get_mut(matrix)
    }

    #[inline]
    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output {
        let index = AxisIndex::from_index(self, matrix.order);
        unsafe { index.get_unchecked(matrix) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        let index = AxisIndex::from_index(self, matrix.order);
        unsafe { index.get_unchecked_mut(matrix) }
    }
}

impl AsIndex for Index {
    #[inline]
    fn row(&self) -> usize {
        self.row
    }

    #[inline]
    fn col(&self) -> usize {
        self.col
    }
}

impl AsIndex for (usize, usize) {
    #[inline]
    fn row(&self) -> usize {
        self.0
    }

    #[inline]
    fn col(&self) -> usize {
        self.1
    }
}

impl AsIndex for [usize; 2] {
    #[inline]
    fn row(&self) -> usize {
        self[0]
    }

    #[inline]
    fn col(&self) -> usize {
        self[1]
    }
}

/// A struct representing the wrapping index of an element in a
/// [`Matrix<T>`].
///
/// [`WrappingIndex`] is the only type that exhibits wrapping indexing
/// behavior. It does not implement [`AsIndex`], and there is no wrapping
/// equivalent for that trait. You cannot pass a `(isize, isize)` or a
/// `[isize; 2]` to methods expecting an index, or more precisely, a type
/// that implements [`MatrixIndex<T>`].
///
/// The design choice is based on the following considerations:
/// - Wrapping indexing does not follow standard indexing conventions,
///   and should always be used explicitly.
/// - Both `(isize, isize)` and `[isize; 2]` are not sufficiently
///   distinguishable from their `usize` counterparts, which would
///   introduce ambiguity and prevent type inference, making type
///   annotations necessary.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct WrappingIndex {
    /// The row index.
    pub row: isize,

    /// The column index.
    pub col: isize,
}

impl WrappingIndex {
    /// Creates a new [`WrappingIndex`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::WrappingIndex;
    ///
    /// let index = WrappingIndex::new(2, 3);
    /// assert_eq!(index.row, 2);
    /// assert_eq!(index.col, 3);
    /// ```
    #[inline]
    pub fn new(row: isize, col: isize) -> Self {
        Self { row, col }
    }

    /// Swaps the row and column indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::WrappingIndex;
    ///
    /// let mut index = WrappingIndex::new(2, 3);
    /// index.swap();
    /// assert_eq!(index, WrappingIndex::new(3, 2));
    /// ```
    #[inline]
    pub fn swap(&mut self) -> &mut Self {
        (self.row, self.col) = (self.col, self.row);
        self
    }
}

unsafe impl<T> MatrixIndex<T> for WrappingIndex {
    type Output = T;

    /// Returns `true` if the index is out of bounds for the given matrix.
    ///
    /// # Notes
    ///
    /// A wrapping index is out of bounds if and only if the matrix is empty.
    #[inline]
    fn is_out_of_bounds(&self, matrix: &Matrix<T>) -> bool {
        matrix.is_empty()
    }

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// This method is safe regardless of it being marked unsafe.
    /// If no panic occurs, the output returned is guaranteed to be valid.
    #[inline]
    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output {
        let index = AxisIndex::from_wrapping_index(self, matrix.order, matrix.shape);
        unsafe { index.get_unchecked(matrix) }
    }

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty.
    ///
    /// # Safety
    ///
    /// This method is safe regardless of it being marked unsafe.
    /// If no panic occurs, the output returned is guaranteed to be valid.
    #[inline]
    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        let index = AxisIndex::from_wrapping_index(self, matrix.order, matrix.shape);
        unsafe { index.get_unchecked_mut(matrix) }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct AxisIndex {
    pub(crate) major: usize,
    pub(crate) minor: usize,
}

impl AxisIndex {
    pub(crate) fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }

    pub(crate) fn from_index<I>(index: I, order: Order) -> Self
    where
        I: AsIndex,
    {
        let (major, minor) = match order {
            Order::RowMajor => (index.row(), index.col()),
            Order::ColMajor => (index.col(), index.row()),
        };
        Self { major, minor }
    }

    pub(crate) fn to_index(self, order: Order) -> Index {
        let (row, col) = match order {
            Order::RowMajor => (self.major, self.minor),
            Order::ColMajor => (self.minor, self.major),
        };
        Index { row, col }
    }

    /// # Panics
    ///
    /// Panics if the size of the `shape` is zero.
    pub(crate) fn from_wrapping_index(
        index: WrappingIndex,
        order: Order,
        shape: AxisShape,
    ) -> Self {
        let (major, minor) = match order {
            Order::RowMajor => (index.row, index.col),
            Order::ColMajor => (index.col, index.row),
        };
        let major = if major < 0 {
            (shape.major() - major.unsigned_abs() % shape.major()) % shape.major()
        } else {
            major as usize % shape.major()
        };
        let minor = if minor < 0 {
            (shape.minor() - minor.unsigned_abs() % shape.minor()) % shape.minor()
        } else {
            minor as usize % shape.minor()
        };
        Self { major, minor }
    }

    // `to_wapping_index` is not implemented for two reasons:
    // - it is a one-to-many mapping
    // - it serves no practical purpose

    pub(crate) fn from_flattened(index: usize, stride: Stride) -> Self {
        let major = index / stride.major();
        let minor = (index % stride.major()) / stride.minor();
        Self { major, minor }
    }

    pub(crate) fn to_flattened(self, stride: Stride) -> usize {
        self.major * stride.major() + self.minor * stride.minor()
    }
}

unsafe impl<T> MatrixIndex<T> for AxisIndex {
    type Output = T;

    fn is_out_of_bounds(&self, matrix: &Matrix<T>) -> bool {
        self.major >= matrix.major() || self.minor >= matrix.minor()
    }

    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output {
        let index = self.to_flattened(matrix.stride());
        unsafe { matrix.data.get_unchecked(index) }
    }

    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        let index = self.to_flattened(matrix.stride());
        unsafe { matrix.data.get_unchecked_mut(index) }
    }
}

mod internal {
    use super::{AsIndex, AxisIndex, WrappingIndex};
    pub trait Sealed {}
    impl<I> Sealed for I where I: AsIndex {}
    impl Sealed for WrappingIndex {}
    impl Sealed for AxisIndex {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;
    use crate::testkit;

    #[test]
    fn test_matrix_get() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert_eq!(matrix.get(Index::new(0, 0)), Ok(&1));
            assert_eq!(matrix.get(Index::new(0, 1)), Ok(&2));
            assert_eq!(matrix.get(Index::new(0, 2)), Ok(&3));
            assert_eq!(matrix.get(Index::new(1, 0)), Ok(&4));
            assert_eq!(matrix.get(Index::new(1, 1)), Ok(&5));
            assert_eq!(matrix.get(Index::new(1, 2)), Ok(&6));

            assert_eq!(matrix.get(Index::new(2, 0)), Err(Error::IndexOutOfBounds));
            assert_eq!(matrix.get(Index::new(0, 3)), Err(Error::IndexOutOfBounds));
            assert_eq!(matrix.get(Index::new(2, 3)), Err(Error::IndexOutOfBounds));
        });
    }

    #[test]
    fn test_matrix_get_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            *matrix.get_mut(Index::new(0, 0)).unwrap() -= 1;
            *matrix.get_mut(Index::new(0, 1)).unwrap() -= 2;
            *matrix.get_mut(Index::new(0, 2)).unwrap() -= 3;
            *matrix.get_mut(Index::new(1, 0)).unwrap() -= 4;
            *matrix.get_mut(Index::new(1, 1)).unwrap() -= 5;
            *matrix.get_mut(Index::new(1, 2)).unwrap() -= 6;
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            testkit::assert_loose_eq(&matrix, &expected);

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
        });
    }

    #[test]
    fn test_matrix_get_unchecked() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| unsafe {
            assert_eq!(matrix.get_unchecked(Index::new(0, 0)), &1);
            assert_eq!(matrix.get_unchecked(Index::new(0, 1)), &2);
            assert_eq!(matrix.get_unchecked(Index::new(0, 2)), &3);
            assert_eq!(matrix.get_unchecked(Index::new(1, 0)), &4);
            assert_eq!(matrix.get_unchecked(Index::new(1, 1)), &5);
            assert_eq!(matrix.get_unchecked(Index::new(1, 2)), &6);
        });
    }

    #[test]
    fn test_matrix_get_unchecked_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| unsafe {
            *matrix.get_unchecked_mut(Index::new(0, 0)) -= 1;
            *matrix.get_unchecked_mut(Index::new(0, 1)) -= 2;
            *matrix.get_unchecked_mut(Index::new(0, 2)) -= 3;
            *matrix.get_unchecked_mut(Index::new(1, 0)) -= 4;
            *matrix.get_unchecked_mut(Index::new(1, 1)) -= 5;
            *matrix.get_unchecked_mut(Index::new(1, 2)) -= 6;
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }

    #[test]
    fn test_matrix_index() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert_eq!(matrix[Index::new(0, 0)], 1);
            assert_eq!(matrix[Index::new(0, 1)], 2);
            assert_eq!(matrix[Index::new(0, 2)], 3);
            assert_eq!(matrix[Index::new(1, 0)], 4);
            assert_eq!(matrix[Index::new(1, 1)], 5);
            assert_eq!(matrix[Index::new(1, 2)], 6);
        });
    }

    #[test]
    #[should_panic]
    fn test_matrix_index_out_of_bounds() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let _ = matrix[Index::new(2, 3)];
    }

    #[test]
    fn test_matrix_index_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            matrix[Index::new(0, 0)] -= 1;
            matrix[Index::new(0, 1)] -= 2;
            matrix[Index::new(0, 2)] -= 3;
            matrix[Index::new(1, 0)] -= 4;
            matrix[Index::new(1, 1)] -= 5;
            matrix[Index::new(1, 2)] -= 6;
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }

    #[test]
    #[should_panic]
    fn test_matrix_index_mut_out_of_bounds() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        matrix[Index::new(2, 3)] += 2;
    }

    #[test]
    fn test_index_new() {
        let index = Index::new(2, 3);
        assert_eq!(index.row, 2);
        assert_eq!(index.col, 3);
    }

    #[test]
    fn test_index_swap() {
        let mut index = Index::new(2, 3);
        index.swap();
        assert_eq!(index, Index::new(3, 2));
    }

    #[test]
    fn test_as_index_is_out_of_bounds() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert!(!(0, 0).is_out_of_bounds(&matrix));
            assert!(!(0, 1).is_out_of_bounds(&matrix));
            assert!(!(0, 2).is_out_of_bounds(&matrix));
            assert!(!(1, 0).is_out_of_bounds(&matrix));
            assert!(!(1, 1).is_out_of_bounds(&matrix));
            assert!(!(1, 2).is_out_of_bounds(&matrix));

            assert!((2, 0).is_out_of_bounds(&matrix));
            assert!((0, 3).is_out_of_bounds(&matrix));
            assert!((2, 3).is_out_of_bounds(&matrix));
        });
    }

    #[test]
    fn test_as_index_ensure_in_bounds() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert!((0, 0).ensure_in_bounds(&matrix).is_ok());
            assert!((0, 1).ensure_in_bounds(&matrix).is_ok());
            assert!((0, 2).ensure_in_bounds(&matrix).is_ok());
            assert!((1, 0).ensure_in_bounds(&matrix).is_ok());
            assert!((1, 1).ensure_in_bounds(&matrix).is_ok());
            assert!((1, 2).ensure_in_bounds(&matrix).is_ok());

            assert_eq!(
                (2, 0).ensure_in_bounds(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                (0, 3).ensure_in_bounds(&matrix),
                Err(Error::IndexOutOfBounds)
            );
            assert_eq!(
                (2, 3).ensure_in_bounds(&matrix),
                Err(Error::IndexOutOfBounds)
            );
        });
    }

    #[test]
    fn test_as_index_get() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert_eq!((0, 0).get(&matrix), Ok(&1));
            assert_eq!((0, 1).get(&matrix), Ok(&2));
            assert_eq!((0, 2).get(&matrix), Ok(&3));
            assert_eq!((1, 0).get(&matrix), Ok(&4));
            assert_eq!((1, 1).get(&matrix), Ok(&5));
            assert_eq!((1, 2).get(&matrix), Ok(&6));

            assert_eq!((2, 0).get(&matrix), Err(Error::IndexOutOfBounds));
            assert_eq!((0, 3).get(&matrix), Err(Error::IndexOutOfBounds));
            assert_eq!((2, 3).get(&matrix), Err(Error::IndexOutOfBounds));
        });
    }

    #[test]
    fn test_as_index_get_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            *(0, 0).get_mut(&mut matrix).unwrap() -= 1;
            *(0, 1).get_mut(&mut matrix).unwrap() -= 2;
            *(0, 2).get_mut(&mut matrix).unwrap() -= 3;
            *(1, 0).get_mut(&mut matrix).unwrap() -= 4;
            *(1, 1).get_mut(&mut matrix).unwrap() -= 5;
            *(1, 2).get_mut(&mut matrix).unwrap() -= 6;
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            testkit::assert_loose_eq(&matrix, &expected);

            assert_eq!((2, 0).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
            assert_eq!((0, 3).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
            assert_eq!((2, 3).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
        });
    }

    #[test]
    fn test_as_index_get_unchecked() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| unsafe {
            assert_eq!((0, 0).get_unchecked(&matrix), &1);
            assert_eq!((0, 1).get_unchecked(&matrix), &2);
            assert_eq!((0, 2).get_unchecked(&matrix), &3);
            assert_eq!((1, 0).get_unchecked(&matrix), &4);
            assert_eq!((1, 1).get_unchecked(&matrix), &5);
            assert_eq!((1, 2).get_unchecked(&matrix), &6);
        });
    }

    #[test]
    fn test_as_index_get_unchecked_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| unsafe {
            *(0, 0).get_unchecked_mut(&mut matrix) -= 1;
            *(0, 1).get_unchecked_mut(&mut matrix) -= 2;
            *(0, 2).get_unchecked_mut(&mut matrix) -= 3;
            *(1, 0).get_unchecked_mut(&mut matrix) -= 4;
            *(1, 1).get_unchecked_mut(&mut matrix) -= 5;
            *(1, 2).get_unchecked_mut(&mut matrix) -= 6;
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }

    #[test]
    fn test_as_index_index() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert_eq!((0, 0).index(&matrix), &1);
            assert_eq!((0, 1).index(&matrix), &2);
            assert_eq!((0, 2).index(&matrix), &3);
            assert_eq!((1, 0).index(&matrix), &4);
            assert_eq!((1, 1).index(&matrix), &5);
            assert_eq!((1, 2).index(&matrix), &6);
        });
    }

    #[test]
    #[should_panic]
    fn test_as_index_index_out_of_bounds() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        (2, 3).index(&matrix);
    }

    #[test]
    fn test_as_index_index_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            *(0, 0).index_mut(&mut matrix) -= 1;
            *(0, 1).index_mut(&mut matrix) -= 2;
            *(0, 2).index_mut(&mut matrix) -= 3;
            *(1, 0).index_mut(&mut matrix) -= 4;
            *(1, 1).index_mut(&mut matrix) -= 5;
            *(1, 2).index_mut(&mut matrix) -= 6;
            let expected = matrix![[0, 0, 0], [0, 0, 0]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }

    #[test]
    #[should_panic]
    fn test_as_index_index_mut_out_of_bounds() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
        *(2, 3).index_mut(&mut matrix) += 2;
    }

    #[test]
    fn test_wrapping_index_new() {
        let index = WrappingIndex::new(2, 3);
        assert_eq!(index.row, 2);
        assert_eq!(index.col, 3);
    }

    #[test]
    fn test_wrapping_index_swap() {
        let mut index = WrappingIndex::new(2, 3);
        index.swap();
        assert_eq!(index, WrappingIndex::new(3, 2));
    }

    #[test]
    fn test_wrapping_index_is_out_of_bounds() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(!WrappingIndex::new(row, col).is_out_of_bounds(&matrix));
                }
            }
        });

        let matrix = Matrix::<i32>::new();
        testkit::for_each_order_unary(matrix, |matrix| {
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(WrappingIndex::new(row, col).is_out_of_bounds(&matrix));
                }
            }
        });
    }

    #[test]
    fn test_wrapping_index_ensure_in_bounds() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert!(
                        WrappingIndex::new(row, col)
                            .ensure_in_bounds(&matrix)
                            .is_ok()
                    );
                }
            }
        });

        let matrix = Matrix::<i32>::new();
        testkit::for_each_order_unary(matrix, |matrix| {
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert_eq!(
                        WrappingIndex::new(row, col).ensure_in_bounds(&matrix),
                        Err(Error::IndexOutOfBounds)
                    );
                }
            }
        });
    }

    #[test]
    fn test_wrapping_index_get() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
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
        });

        let matrix = Matrix::<i32>::new();
        testkit::for_each_order_unary(matrix, |matrix| {
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
        });
    }

    #[test]
    fn test_wrapping_index_get_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
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
            testkit::assert_loose_eq(&matrix, &expected);
        });

        let matrix = Matrix::<i32>::new();
        testkit::for_each_order_unary(matrix, |mut matrix| {
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
        });
    }

    #[test]
    fn test_wrapping_index_get_unchecked() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| unsafe {
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
        });
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_get_unchecked_fails() {
        let matrix = Matrix::<i32>::new();
        unsafe {
            WrappingIndex::new(0, 0).get_unchecked(&matrix);
        }
    }

    #[test]
    fn test_wrapping_index_get_unchecked_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| unsafe {
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
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_get_unchecked_mut_fails() {
        let mut matrix = Matrix::<i32>::new();
        unsafe {
            *WrappingIndex::new(0, 0).get_unchecked_mut(&mut matrix) += 2;
        }
    }

    #[test]
    fn test_wrapping_index_index() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert_eq!(WrappingIndex::new(0, 0).index(&matrix), &1);
            assert_eq!(WrappingIndex::new(0, 1).index(&matrix), &2);
            assert_eq!(WrappingIndex::new(0, 2).index(&matrix), &3);
            assert_eq!(WrappingIndex::new(1, 0).index(&matrix), &4);
            assert_eq!(WrappingIndex::new(1, 1).index(&matrix), &5);
            assert_eq!(WrappingIndex::new(1, 2).index(&matrix), &6);
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    assert_eq!(WrappingIndex::new(row, col).index(&matrix), &1);
                }
            }
        });
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_index_out_of_bounds() {
        let matrix = Matrix::<i32>::new();
        WrappingIndex::new(0, 0).index(&matrix);
    }

    #[test]
    fn test_wrapping_index_index_mut() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            *WrappingIndex::new(0, 0).index_mut(&mut matrix) -= 1;
            *WrappingIndex::new(0, 1).index_mut(&mut matrix) -= 2;
            *WrappingIndex::new(0, 2).index_mut(&mut matrix) -= 3;
            *WrappingIndex::new(1, 0).index_mut(&mut matrix) -= 4;
            *WrappingIndex::new(1, 1).index_mut(&mut matrix) -= 5;
            *WrappingIndex::new(1, 2).index_mut(&mut matrix) -= 6;
            for row in (-6..=6).step_by(2) {
                for col in (-6..=6).step_by(3) {
                    *WrappingIndex::new(row, col).index_mut(&mut matrix) -= 1;
                }
            }
            let expected = matrix![[-35, 0, 0], [0, 0, 0]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }

    #[test]
    #[should_panic]
    fn test_wrapping_index_index_mut_out_of_bounds() {
        let mut matrix = Matrix::<i32>::new();
        *WrappingIndex::new(0, 0).index_mut(&mut matrix) += 2;
    }
}
