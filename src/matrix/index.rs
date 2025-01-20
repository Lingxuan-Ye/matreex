use super::order::Order;
use super::shape::AxisShape;
use super::Matrix;
use crate::error::{Error, Result};

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
    /// use matreex::{matrix, Error};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.get((1, 1)), Ok(&4));
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
    /// use matreex::{matrix, Error};
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(matrix.get_mut((1, 1)), Ok(&mut 4));
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
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(unsafe { matrix.get_unchecked((1, 1)) }, &4);
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
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(unsafe { matrix.get_unchecked_mut((1, 1)) }, &mut 4);
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

impl<T, I> std::ops::Index<I> for Matrix<T>
where
    I: MatrixIndex<T>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        index.index(self)
    }
}

impl<T, I> std::ops::IndexMut<I> for Matrix<T>
where
    I: MatrixIndex<T>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        index.index_mut(self)
    }
}

/// A helper trait used for [`Matrix<T>`] indexing.
///
/// # Safety
///
/// Marking this trait as `unsafe` originates from a poor imitation
/// of [`SliceIndex`]. In another words, I have no idea what I'm doing.
///
/// [`SliceIndex`]: core::slice::SliceIndex
pub unsafe trait MatrixIndex<T>: internal::Sealed {
    /// The output type returned by methods.
    type Output;

    /// Returns a reference to the output at this location,
    /// if in bounds.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output>;

    /// Returns a mutable reference to the output at this location,
    /// if in bounds.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output>;

    /// Returns a reference to the output at this location,
    /// without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output;

    /// Returns a mutable reference to the output at this location,
    /// without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output;

    /// Returns a reference to the output at this location.
    ///
    /// # Panics
    ///
    /// Panics if out of bounds.
    fn index(self, matrix: &Matrix<T>) -> &Self::Output;

    /// Returns a mutable reference to the output at this location.
    ///
    /// # Panics
    ///
    /// Panics if out of bounds.
    fn index_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output;
}

/// A structure representing the index of an element in a [`Matrix<T>`].
///
/// # Notes
///
/// Any type that implements [`Into<Index>`] can be used as an index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Index {
    /// The row index of the element.
    pub row: usize,

    /// The column index of the element.
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
    /// ```
    #[inline]
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    /// Returns `true` if the index is out of bounds for the given matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Index};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let index = Index::new(2, 3);
    /// assert!(index.is_out_of_bounds(&matrix));
    /// ```
    #[inline]
    pub fn is_out_of_bounds<T>(&self, matrix: &Matrix<T>) -> bool {
        let shape = matrix.shape();
        self.row >= shape.nrows() || self.col >= shape.ncols()
    }

    /// Ensures the index is in bounds for the given matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{matrix, Index};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// let index = Index::new(0, 0);
    /// assert!(index.ensure_in_bounds(&matrix).is_ok());
    /// ```
    #[inline]
    pub fn ensure_in_bounds<T>(&self, matrix: &Matrix<T>) -> Result<&Self> {
        if self.is_out_of_bounds(matrix) {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self)
        }
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
}

impl Index {
    pub(super) fn from_axis_index(index: AxisIndex, order: Order) -> Self {
        let (row, col) = match order {
            Order::RowMajor => (index.major, index.minor),
            Order::ColMajor => (index.minor, index.major),
        };
        Self { row, col }
    }

    pub(super) fn to_axis_index(self, order: Order) -> AxisIndex {
        let (major, minor) = match order {
            Order::RowMajor => (self.row, self.col),
            Order::ColMajor => (self.col, self.row),
        };
        AxisIndex { major, minor }
    }

    pub(super) fn from_flattened(index: usize, order: Order, shape: AxisShape) -> Self {
        Self::from_axis_index(AxisIndex::from_flattened(index, shape), order)
    }

    pub(super) fn to_flattened(self, order: Order, shape: AxisShape) -> usize {
        self.to_axis_index(order).to_flattened(shape)
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

unsafe impl<T, I> MatrixIndex<T> for I
where
    I: Into<Index>,
{
    type Output = T;

    #[inline]
    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output> {
        let index = self.into().to_axis_index(matrix.order);
        index.get(matrix)
    }

    #[inline]
    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        let index = self.into().to_axis_index(matrix.order);
        index.get_mut(matrix)
    }

    #[inline]
    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output {
        let index = self.into().to_axis_index(matrix.order);
        unsafe { index.get_unchecked(matrix) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        let index = self.into().to_axis_index(matrix.order);
        unsafe { index.get_unchecked_mut(matrix) }
    }

    #[inline]
    fn index(self, matrix: &Matrix<T>) -> &Self::Output {
        let index = self.into().to_axis_index(matrix.order);
        index.index(matrix)
    }

    #[inline]
    fn index_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        let index = self.into().to_axis_index(matrix.order);
        index.index_mut(matrix)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AxisIndex {
    pub(super) major: usize,
    pub(super) minor: usize,
}

impl AxisIndex {
    pub(super) fn is_out_of_bounds(&self, shape: AxisShape) -> bool {
        self.major >= shape.major() || self.minor >= shape.minor()
    }

    pub(super) fn ensure_in_bounds(&self, shape: AxisShape) -> Result<&Self> {
        if self.is_out_of_bounds(shape) {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self)
        }
    }

    pub(super) fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }

    pub(super) fn from_flattened(index: usize, shape: AxisShape) -> Self {
        let major = index / shape.major_stride();
        // let minor = (index % shape.major_stride()) / shape.minor_stride();
        let minor = index % shape.major_stride();
        Self { major, minor }
    }

    pub(super) fn to_flattened(self, shape: AxisShape) -> usize {
        // self.major * shape.major_stride() + self.minor * shape.minor_stride()
        self.major * shape.major_stride() + self.minor
    }
}

unsafe impl<T> MatrixIndex<T> for AxisIndex {
    type Output = T;

    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output> {
        self.ensure_in_bounds(matrix.shape)?;
        unsafe { Ok(self.get_unchecked(matrix)) }
    }

    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        self.ensure_in_bounds(matrix.shape)?;
        unsafe { Ok(self.get_unchecked_mut(matrix)) }
    }

    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output {
        let index = self.to_flattened(matrix.shape);
        unsafe { matrix.data.get_unchecked(index) }
    }

    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        let index = self.to_flattened(matrix.shape);
        unsafe { matrix.data.get_unchecked_mut(index) }
    }

    fn index(self, matrix: &Matrix<T>) -> &Self::Output {
        if let Err(error) = self.ensure_in_bounds(matrix.shape) {
            panic!("{error}");
        }
        unsafe { self.get_unchecked(matrix) }
    }

    fn index_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        if let Err(error) = self.ensure_in_bounds(matrix.shape) {
            panic!("{error}");
        }
        unsafe { self.get_unchecked_mut(matrix) }
    }
}

#[inline(always)]
pub(super) fn map_flattened_index_for_transpose(index: usize, mut shape: AxisShape) -> usize {
    let mut index = AxisIndex::from_flattened(index, shape);
    index.swap();
    shape.transpose();
    index.to_flattened(shape)
}

mod internal {
    use super::{AxisIndex, Index};

    pub trait Sealed {}

    impl<I> Sealed for I where I: Into<Index> {}

    impl Sealed for AxisIndex {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_get() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert_eq!(matrix.get((0, 0)), Ok(&0));
        assert_eq!(matrix.get((0, 1)), Ok(&1));
        assert_eq!(matrix.get((0, 2)), Ok(&2));
        assert_eq!(matrix.get((1, 0)), Ok(&3));
        assert_eq!(matrix.get((1, 1)), Ok(&4));
        assert_eq!(matrix.get((1, 2)), Ok(&5));
        assert_eq!(matrix.get((2, 0)), Err(Error::IndexOutOfBounds));
    }

    #[test]
    fn test_get_mut() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert_eq!(matrix.get_mut((0, 0)), Ok(&mut 0));
        assert_eq!(matrix.get_mut((0, 1)), Ok(&mut 1));
        assert_eq!(matrix.get_mut((0, 2)), Ok(&mut 2));
        assert_eq!(matrix.get_mut((1, 0)), Ok(&mut 3));
        assert_eq!(matrix.get_mut((1, 1)), Ok(&mut 4));
        assert_eq!(matrix.get_mut((1, 2)), Ok(&mut 5));
        assert_eq!(matrix.get_mut((2, 0)), Err(Error::IndexOutOfBounds));
    }

    #[test]
    fn test_get_unchecked() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        unsafe {
            assert_eq!(matrix.get_unchecked((0, 0)), &0);
            assert_eq!(matrix.get_unchecked((0, 1)), &1);
            assert_eq!(matrix.get_unchecked((0, 2)), &2);
            assert_eq!(matrix.get_unchecked((1, 0)), &3);
            assert_eq!(matrix.get_unchecked((1, 1)), &4);
            assert_eq!(matrix.get_unchecked((1, 2)), &5);
        }
    }

    #[test]
    fn test_get_unchecked_mut() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        unsafe {
            assert_eq!(matrix.get_unchecked_mut((0, 0)), &mut 0);
            assert_eq!(matrix.get_unchecked_mut((0, 1)), &mut 1);
            assert_eq!(matrix.get_unchecked_mut((0, 2)), &mut 2);
            assert_eq!(matrix.get_unchecked_mut((1, 0)), &mut 3);
            assert_eq!(matrix.get_unchecked_mut((1, 1)), &mut 4);
            assert_eq!(matrix.get_unchecked_mut((1, 2)), &mut 5);
        }
    }

    #[test]
    fn test_index() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert_eq!(matrix[(0, 0)], 0);
        assert_eq!(matrix[(0, 1)], 1);
        assert_eq!(matrix[(0, 2)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
        assert_eq!(matrix[(1, 2)], 5);
    }

    #[test]
    fn test_index_mut() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        matrix[(0, 0)] += 1;
        matrix[(0, 1)] += 1;
        matrix[(0, 2)] += 1;
        matrix[(1, 0)] += 1;
        matrix[(1, 1)] += 1;
        matrix[(1, 2)] += 1;
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]])
    }

    #[test]
    #[should_panic]
    fn test_row_out_of_bounds() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let _ = matrix[(2, 0)];
    }

    #[test]
    #[should_panic]
    fn test_col_out_of_bounds() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let _ = matrix[(0, 3)];
    }

    #[test]
    fn test_struct_index_new() {
        let index = Index::new(2, 3);
        assert_eq!(index.row, 2);
        assert_eq!(index.col, 3);
    }

    #[test]
    fn test_struct_index_is_out_of_bounds() {
        let matrix = Matrix::<i32>::with_default((2, 3)).unwrap();

        let index = Index::new(0, 0);
        assert!(!index.is_out_of_bounds(&matrix));

        let index = Index::new(1, 2);
        assert!(!index.is_out_of_bounds(&matrix));

        let index = Index::new(1, 3);
        assert!(index.is_out_of_bounds(&matrix));

        let index = Index::new(2, 2);
        assert!(index.is_out_of_bounds(&matrix));

        let index = Index::new(2, 3);
        assert!(index.is_out_of_bounds(&matrix));
    }

    #[test]
    fn test_struct_index_ensure_in_bounds() {
        let matrix = Matrix::<i32>::with_default((2, 3)).unwrap();

        let index = Index::new(0, 0);
        assert_eq!(index.ensure_in_bounds(&matrix), Ok(&index));

        let index = Index::new(1, 2);
        assert_eq!(index.ensure_in_bounds(&matrix), Ok(&index));

        let index = Index::new(1, 3);
        assert_eq!(
            index.ensure_in_bounds(&matrix),
            Err(Error::IndexOutOfBounds)
        );

        let index = Index::new(2, 2);
        assert_eq!(
            index.ensure_in_bounds(&matrix),
            Err(Error::IndexOutOfBounds)
        );

        let index = Index::new(2, 3);
        assert_eq!(
            index.ensure_in_bounds(&matrix),
            Err(Error::IndexOutOfBounds)
        );
    }

    #[test]
    fn test_struct_index_swap() {
        let mut index = Index::new(2, 3);
        index.swap();
        assert_eq!(index, Index::new(3, 2));
    }
}
