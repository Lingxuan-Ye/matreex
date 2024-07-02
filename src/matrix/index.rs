use super::order::Order;
use super::shape::AxisShape;
use super::Matrix;
use crate::error::{Error, Result};

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
    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output>;

    /// Returns a mutable reference to the output at this location,
    /// if in bounds.
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
    /// use matreex::{matrix, Error};
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// unsafe { assert_eq!(matrix.get_unchecked((1, 1)), &4); }
    /// ```
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`get`]: Matrix::get
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
    /// use matreex::{matrix, Error};
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// unsafe { assert_eq!(matrix.get_unchecked_mut((1, 1)), &mut 4); }
    /// ```
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`get_mut`]: Matrix::get_mut
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

    fn index(&self, index: I) -> &Self::Output {
        index.index(self)
    }
}

impl<T, I> std::ops::IndexMut<I> for Matrix<T>
where
    I: MatrixIndex<T>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        index.index_mut(self)
    }
}

/// Any type that implements this trait can be used to index a [`Matrix<T>`].
///
/// # Examples
///
/// ```
/// use matreex::{matrix, Index};
///
/// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
///
/// assert_eq!(matrix[Index::new(1, 1)], 4);
/// assert_eq!(matrix[(1, 1)], 4);
/// assert_eq!(matrix[[1, 1]], 4);
/// ```
pub trait IndexLike {
    /// Returns the row of the index.
    fn row(&self) -> usize;

    /// Returns the column of the index.
    fn col(&self) -> usize;

    /// Returns `true` if the index is out of bounds for given matrix.
    fn is_out_of_bounds<T>(&self, matrix: &Matrix<T>) -> bool {
        let shape = matrix.shape();
        self.row() >= shape.nrows || self.col() >= shape.ncols
    }
}

/// A structure that represents the index of an element in a [`Matrix<T>`].
///
/// # Notes
///
/// You might prefer using `(usize, usize)` for matrix indexing instead.
/// Refer to [`IndexLike`] for more information.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Index {
    /// The row index of the element.
    pub row: usize,

    /// The column index of the element.
    pub col: usize,
}

impl Index {
    /// Creates a new [`Index`] instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Index;
    ///
    /// let index = Index::new(2, 3);
    /// ```
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

impl std::fmt::Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

impl IndexLike for Index {
    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }
}

impl IndexLike for (usize, usize) {
    fn row(&self) -> usize {
        self.0
    }

    fn col(&self) -> usize {
        self.1
    }
}

impl IndexLike for [usize; 2] {
    fn row(&self) -> usize {
        self[0]
    }

    fn col(&self) -> usize {
        self[1]
    }
}

unsafe impl<T, I> MatrixIndex<T> for I
where
    I: IndexLike,
{
    type Output = T;

    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output> {
        AxisIndex::from_index(self, matrix.order).get(matrix)
    }

    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        AxisIndex::from_index(self, matrix.order).get_mut(matrix)
    }

    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output {
        unsafe { AxisIndex::from_index(self, matrix.order).get_unchecked(matrix) }
    }

    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        unsafe { AxisIndex::from_index(self, matrix.order).get_unchecked_mut(matrix) }
    }

    fn index(self, matrix: &Matrix<T>) -> &Self::Output {
        AxisIndex::from_index(self, matrix.order).index(matrix)
    }

    fn index_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        AxisIndex::from_index(self, matrix.order).index_mut(matrix)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AxisIndex {
    pub(super) major: usize,
    pub(super) minor: usize,
}

impl AxisIndex {
    pub(super) fn new(major: usize, minor: usize) -> Self {
        Self { major, minor }
    }

    pub(super) fn is_out_of_bounds(&self, shape: AxisShape) -> bool {
        self.major >= shape.major() || self.minor >= shape.minor()
    }

    pub(super) fn transpose(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }
}

// For `IndexLike`, `AxisIndex`, and `usize` (flattened index), we assume that
// the flattened index is always valid. Therefore, in the conversions among
// these three, only conversions to a flattened index require boundary checks.
impl AxisIndex {
    pub(super) fn from_index<I: IndexLike>(index: I, order: Order) -> Self {
        let (major, minor) = match order {
            Order::RowMajor => (index.row(), index.col()),
            Order::ColMajor => (index.col(), index.row()),
        };
        Self { major, minor }
    }

    pub(super) fn from_flattened(index: usize, shape: AxisShape) -> Self {
        let major = index / shape.major_stride();
        // let minor = (index % shape.major_stride()) / shape.minor_stride();
        let minor = index % shape.major_stride();
        Self { major, minor }
    }

    pub(super) fn into_index(self, order: Order) -> Index {
        match order {
            Order::RowMajor => Index::new(self.major, self.minor),
            Order::ColMajor => Index::new(self.minor, self.major),
        }
    }

    pub(super) fn into_flattened_unchecked(self, shape: AxisShape) -> usize {
        // self.major * shape.major_stride() + self.minor * shape.minor_stride()
        self.major * shape.major_stride() + self.minor
    }

    pub(super) fn try_into_flattened(self, shape: AxisShape) -> Result<usize> {
        if self.is_out_of_bounds(shape) {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.into_flattened_unchecked(shape))
        }
    }
}

unsafe impl<T> MatrixIndex<T> for AxisIndex {
    type Output = T;

    fn get(self, matrix: &Matrix<T>) -> Result<&Self::Output> {
        let index = self.try_into_flattened(matrix.shape)?;
        unsafe { Ok(matrix.data.get_unchecked(index)) }
    }

    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        let index = self.try_into_flattened(matrix.shape)?;
        unsafe { Ok(matrix.data.get_unchecked_mut(index)) }
    }

    unsafe fn get_unchecked(self, matrix: &Matrix<T>) -> &Self::Output {
        let index = self.into_flattened_unchecked(matrix.shape);
        unsafe { matrix.data.get_unchecked(index) }
    }

    unsafe fn get_unchecked_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        let index = self.into_flattened_unchecked(matrix.shape);
        unsafe { matrix.data.get_unchecked_mut(index) }
    }

    fn index(self, matrix: &Matrix<T>) -> &Self::Output {
        match self.try_into_flattened(matrix.shape) {
            Err(error) => panic!("{error}"),
            Ok(index) => unsafe { matrix.data.get_unchecked(index) },
        }
    }

    fn index_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        match self.try_into_flattened(matrix.shape) {
            Err(error) => panic!("{error}"),
            Ok(index) => unsafe { matrix.data.get_unchecked_mut(index) },
        }
    }
}

impl<T> Matrix<T> {
    pub(super) fn unflatten_index(index: usize, order: Order, shape: AxisShape) -> Index {
        AxisIndex::from_flattened(index, shape).into_index(order)
    }

    pub(super) fn flatten_index_unchecked<I: IndexLike>(
        index: I,
        order: Order,
        shape: AxisShape,
    ) -> usize {
        AxisIndex::from_index(index, order).into_flattened_unchecked(shape)
    }

    #[allow(dead_code)]
    pub(super) fn try_flatten_index<I: IndexLike>(
        index: I,
        order: Order,
        shape: AxisShape,
    ) -> Result<usize> {
        AxisIndex::from_index(index, order).try_into_flattened(shape)
    }

    #[inline]
    pub(super) fn reindex_to_different_order_unchecked(
        index: usize,
        src_shape: AxisShape,
    ) -> usize {
        // This implementation is based on the idea that the element at the
        // same position remains the same across different orders. Assuming
        // that the original order is `src_order`, and given that the `Index`
        // instance representing the position is invariant, we have:
        //
        // ```
        // let src_flattened_index = index;
        // let src_axis_index = AxisIndex::from_flattened(src_flattened_index, src_shape);
        //
        // // invariant
        // let position = match src_order {
        //     Order::RowMajor => Index::new(src_axis_index.major, src_axis_index.minor),
        //     Order::ColMajor => Index::new(src_axis_index.minor, src_axis_index.major),
        // };
        //
        // let dest_order = src_order.switch();
        // let dest_axis_index = match dest_order {
        //     Order::RowMajor => AxisIndex::new(position.row, position.col),
        //     Order::ColMajor => AxisIndex::new(position.col, position.row),
        // };
        // let mut dest_shape = src_shape;
        // dest_shape.transpose();
        // let dest_flattened_index = dest_axis_index.into_flattened_unchecked(dest_shape);
        // dest_flattened_index
        // ```
        //
        // Note that the variable `dest_axis_index` is always the transpose of
        // `src_axis_index`, which allows us to simplify the code to the
        // following:
        let mut index = AxisIndex::from_flattened(index, src_shape);
        index.transpose();

        let mut dest_shape = src_shape;
        dest_shape.transpose();

        index.into_flattened_unchecked(dest_shape)
    }
}

mod internal {
    pub trait Sealed {}

    impl<I: super::IndexLike> Sealed for I {}

    impl Sealed for super::AxisIndex {}
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
    fn test_trait_index_like() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];

        assert_eq!(Index::new(2, 3).row(), 2);
        assert_eq!(Index::new(2, 3).col(), 3);
        assert!(!Index::new(1, 2).is_out_of_bounds(&matrix));
        assert!(Index::new(1, 3).is_out_of_bounds(&matrix));
        assert!(Index::new(2, 2).is_out_of_bounds(&matrix));
        assert!(Index::new(2, 3).is_out_of_bounds(&matrix));

        assert_eq!((2, 3).row(), 2);
        assert_eq!((2, 3).col(), 3);
        assert!(!(1, 2).is_out_of_bounds(&matrix));
        assert!((1, 3).is_out_of_bounds(&matrix));
        assert!((2, 2).is_out_of_bounds(&matrix));
        assert!((2, 3).is_out_of_bounds(&matrix));

        assert_eq!([2, 3].row(), 2);
        assert_eq!([2, 3].col(), 3);
        assert!(![1, 2].is_out_of_bounds(&matrix));
        assert!([1, 3].is_out_of_bounds(&matrix));
        assert!([2, 2].is_out_of_bounds(&matrix));
        assert!([2, 3].is_out_of_bounds(&matrix));
    }

    #[test]
    fn test_struct_index_new() {
        let expected = Index { row: 2, col: 3 };
        assert_eq!(Index::new(2, 3), expected);
        assert_ne!(Index::new(3, 2), expected);
    }

    #[test]
    fn test_struct_index_display() {
        assert_eq!(Index::new(2, 3).to_string(), "(2, 3)");
        assert_eq!(Index::new(3, 2).to_string(), "(3, 2)");
    }
}
