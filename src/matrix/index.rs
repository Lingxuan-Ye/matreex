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
        unsafe { &*index.get_unchecked(self) }
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
        unsafe { &mut *index.get_unchecked_mut(self) }
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
        unsafe { Ok(&*self.get_unchecked(matrix)) }
    }

    /// Returns a mutable reference to the output at this location, if in
    /// bounds.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        self.ensure_in_bounds(matrix)?;
        unsafe { Ok(&mut *self.get_unchecked_mut(matrix)) }
    }

    /// Returns a pointer to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling `matrix` pointer
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: *const Matrix<T>) -> *const Self::Output;

    /// Returns a mutable pointer to the output at this location, without
    /// performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index or a dangling `matrix` pointer
    /// is *[undefined behavior]* even if the resulting pointer is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<T>) -> *mut Self::Output;

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

/// A structure representing the index of an element in a [`Matrix<T>`].
///
/// Refer to [`SingleElementIndex`] for more information.
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

    pub(super) fn from_flattened(index: usize, order: Order, shape: AxisShape) -> Self {
        AxisIndex::from_flattened(index, shape).to_index(order)
    }

    pub(super) fn to_flattened(self, order: Order, shape: AxisShape) -> usize {
        AxisIndex::from_index(&self, order).to_flattened(shape)
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

/// A trait used for single-element indexing in a [`Matrix<T>`].
///
/// # Examples
///
/// ```
/// use matreex::matrix;
/// use matreex::matrix::index::SingleElementIndex;
///
/// struct I(usize, usize);
///
/// impl SingleElementIndex for I {
///     fn row(&self) -> usize {
///         self.0
///     }
///
///     fn col(&self) -> usize {
///         self.1
///     }
/// }
///
/// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
/// assert_eq!(matrix.get(I(1, 1)), Ok(&4));
/// ```
pub trait SingleElementIndex {
    /// The row index of the element.
    fn row(&self) -> usize;

    /// The column index of the element.
    fn col(&self) -> usize;
}

unsafe impl<T, I> MatrixIndex<T> for I
where
    I: SingleElementIndex,
{
    type Output = T;

    #[inline]
    fn is_out_of_bounds(&self, matrix: &Matrix<T>) -> bool {
        AxisIndex::from_index(self, matrix.order).is_out_of_bounds(matrix)
    }

    #[inline]
    unsafe fn get_unchecked(self, matrix: *const Matrix<T>) -> *const Self::Output {
        let index = AxisIndex::from_index(&self, (*matrix).order);
        unsafe { index.get_unchecked(matrix) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<T>) -> *mut Self::Output {
        let index = AxisIndex::from_index(&self, (*matrix).order);
        unsafe { index.get_unchecked_mut(matrix) }
    }
}

impl SingleElementIndex for Index {
    #[inline]
    fn row(&self) -> usize {
        self.row
    }

    #[inline]
    fn col(&self) -> usize {
        self.col
    }
}

impl SingleElementIndex for (usize, usize) {
    #[inline]
    fn row(&self) -> usize {
        self.0
    }

    #[inline]
    fn col(&self) -> usize {
        self.1
    }
}

impl SingleElementIndex for [usize; 2] {
    #[inline]
    fn row(&self) -> usize {
        self[0]
    }

    #[inline]
    fn col(&self) -> usize {
        self[1]
    }
}

    pub row: isize,
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WrappingIndex {
    pub col: isize,
}

impl WrappingIndex {
    #[inline]
    pub fn new(row: isize, col: isize) -> Self {
        Self { row, col }
    }

    #[inline]
    pub fn swap(&mut self) -> &mut Self {
        (self.row, self.col) = (self.col, self.row);
        self
    }

    pub(super) fn to_flattened(self, order: Order, shape: AxisShape) -> usize {
        let (major, minor) = match order {
            Order::RowMajor => (self.row, self.col),
            Order::ColMajor => (self.col, self.row),
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
        AxisIndex { major, minor }.to_flattened(shape)
    }
}

impl From<(isize, isize)> for WrappingIndex {
    #[inline]
    fn from(value: (isize, isize)) -> Self {
        let (row, col) = value;
        Self { row, col }
    }
}

impl From<[isize; 2]> for WrappingIndex {
    #[inline]
    fn from(value: [isize; 2]) -> Self {
        let [row, col] = value;
        Self { row, col }
    }
}

unsafe impl<T> MatrixIndex<T> for WrappingIndex {
    type Output = T;

    #[inline]
    fn is_out_of_bounds(&self, matrix: &Matrix<T>) -> bool {
        matrix.is_empty()
    }

    #[inline]
    unsafe fn get_unchecked(self, matrix: *const Matrix<T>) -> *const Self::Output {
        let index = self.to_flattened((*matrix).order, (*matrix).shape);
        unsafe { (*matrix).data.get_unchecked(index) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<T>) -> *mut Self::Output {
        let index = self.to_flattened((*matrix).order, (*matrix).shape);
        unsafe { (*matrix).data.get_unchecked_mut(index) }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AxisIndex {
    pub(super) major: usize,
    pub(super) minor: usize,
}

impl AxisIndex {
    pub(super) fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }

    pub(super) fn from_index<I>(index: &I, order: Order) -> Self
    where
        I: SingleElementIndex,
    {
        let (major, minor) = match order {
            Order::RowMajor => (index.row(), index.col()),
            Order::ColMajor => (index.col(), index.row()),
        };
        Self { major, minor }
    }

    pub(super) fn to_index(self, order: Order) -> Index {
        let (row, col) = match order {
            Order::RowMajor => (self.major, self.minor),
            Order::ColMajor => (self.minor, self.major),
        };
        Index { row, col }
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

    fn is_out_of_bounds(&self, matrix: &Matrix<T>) -> bool {
        self.major >= matrix.major() || self.minor >= matrix.minor()
    }

    unsafe fn get_unchecked(self, matrix: *const Matrix<T>) -> *const Self::Output {
        let index = self.to_flattened((*matrix).shape);
        unsafe { (*matrix).data.get_unchecked(index) }
    }

    unsafe fn get_unchecked_mut(self, matrix: *mut Matrix<T>) -> *mut Self::Output {
        let index = self.to_flattened((*matrix).shape);
        unsafe { (*matrix).data.get_unchecked_mut(index) }
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
    use super::{AxisIndex, SingleElementIndex, WrappingIndex};

    pub trait Sealed {}

    impl<I> Sealed for I where I: SingleElementIndex {}

    impl Sealed for WrappingIndex {}

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
    fn test_struct_index_swap() {
        let mut index = Index::new(2, 3);
        index.swap();
        assert_eq!(index, Index::new(3, 2));
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct I(usize, usize);

    impl SingleElementIndex for I {
        fn row(&self) -> usize {
            self.0
        }

        fn col(&self) -> usize {
            self.1
        }
    }

    #[test]
    fn test_single_element_index() {
        let index = I(2, 3);
        assert_eq!(index.row(), 2);
        assert_eq!(index.col(), 3);
    }

    #[test]
    fn test_single_element_index_is_out_of_bounds() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert!(!I(0, 0).is_out_of_bounds(&matrix));
        assert!(!I(0, 1).is_out_of_bounds(&matrix));
        assert!(!I(0, 2).is_out_of_bounds(&matrix));
        assert!(!I(1, 0).is_out_of_bounds(&matrix));
        assert!(!I(1, 1).is_out_of_bounds(&matrix));
        assert!(!I(1, 2).is_out_of_bounds(&matrix));
        assert!(I(1, 3).is_out_of_bounds(&matrix));
        assert!(I(2, 2).is_out_of_bounds(&matrix));
        assert!(I(2, 3).is_out_of_bounds(&matrix));
    }

    #[test]
    fn test_single_element_index_ensure_in_bounds() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert!(I(0, 0).ensure_in_bounds(&matrix).is_ok());
        assert!(I(0, 1).ensure_in_bounds(&matrix).is_ok());
        assert!(I(0, 2).ensure_in_bounds(&matrix).is_ok());
        assert!(I(1, 0).ensure_in_bounds(&matrix).is_ok());
        assert!(I(1, 1).ensure_in_bounds(&matrix).is_ok());
        assert!(I(1, 2).ensure_in_bounds(&matrix).is_ok());
        assert_eq!(
            I(1, 3).ensure_in_bounds(&matrix),
            Err(Error::IndexOutOfBounds)
        );
        assert_eq!(
            I(2, 2).ensure_in_bounds(&matrix),
            Err(Error::IndexOutOfBounds)
        );
        assert_eq!(
            I(2, 3).ensure_in_bounds(&matrix),
            Err(Error::IndexOutOfBounds)
        );
    }

    #[test]
    fn test_single_element_index_get() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert_eq!(I(0, 0).get(&matrix), Ok(&0));
        assert_eq!(I(0, 1).get(&matrix), Ok(&1));
        assert_eq!(I(0, 2).get(&matrix), Ok(&2));
        assert_eq!(I(1, 0).get(&matrix), Ok(&3));
        assert_eq!(I(1, 1).get(&matrix), Ok(&4));
        assert_eq!(I(1, 2).get(&matrix), Ok(&5));
        assert_eq!(I(1, 3).get(&matrix), Err(Error::IndexOutOfBounds));
        assert_eq!(I(2, 2).get(&matrix), Err(Error::IndexOutOfBounds));
        assert_eq!(I(2, 3).get(&matrix), Err(Error::IndexOutOfBounds));
    }

    #[test]
    fn test_single_element_index_get_mut() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert_eq!(I(0, 0).get_mut(&mut matrix), Ok(&mut 0));
        assert_eq!(I(0, 1).get_mut(&mut matrix), Ok(&mut 1));
        assert_eq!(I(0, 2).get_mut(&mut matrix), Ok(&mut 2));
        assert_eq!(I(1, 0).get_mut(&mut matrix), Ok(&mut 3));
        assert_eq!(I(1, 1).get_mut(&mut matrix), Ok(&mut 4));
        assert_eq!(I(1, 2).get_mut(&mut matrix), Ok(&mut 5));
        assert_eq!(I(1, 3).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
        assert_eq!(I(2, 2).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
        assert_eq!(I(2, 3).get_mut(&mut matrix), Err(Error::IndexOutOfBounds));
    }

    #[test]
    fn test_single_element_index_get_unchecked() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        unsafe {
            assert_eq!(*I(0, 0).get_unchecked(&matrix), 0);
            assert_eq!(*I(0, 1).get_unchecked(&matrix), 1);
            assert_eq!(*I(0, 2).get_unchecked(&matrix), 2);
            assert_eq!(*I(1, 0).get_unchecked(&matrix), 3);
            assert_eq!(*I(1, 1).get_unchecked(&matrix), 4);
            assert_eq!(*I(1, 2).get_unchecked(&matrix), 5);
        }
    }

    #[test]
    fn test_single_element_index_get_unchecked_mut() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        unsafe {
            assert_eq!(*I(0, 0).get_unchecked_mut(&mut matrix), 0);
            assert_eq!(*I(0, 1).get_unchecked_mut(&mut matrix), 1);
            assert_eq!(*I(0, 2).get_unchecked_mut(&mut matrix), 2);
            assert_eq!(*I(1, 0).get_unchecked_mut(&mut matrix), 3);
            assert_eq!(*I(1, 1).get_unchecked_mut(&mut matrix), 4);
            assert_eq!(*I(1, 2).get_unchecked_mut(&mut matrix), 5);
        }
    }

    #[test]
    fn test_single_element_index_index() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        assert_eq!(I(0, 0).index(&matrix), &0);
        assert_eq!(I(0, 1).index(&matrix), &1);
        assert_eq!(I(0, 2).index(&matrix), &2);
        assert_eq!(I(1, 0).index(&matrix), &3);
        assert_eq!(I(1, 1).index(&matrix), &4);
        assert_eq!(I(1, 2).index(&matrix), &5);
    }

    #[test]
    fn test_single_element_index_index_mut() {
        let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
        *I(0, 0).index_mut(&mut matrix) += 1;
        *I(0, 1).index_mut(&mut matrix) += 1;
        *I(0, 2).index_mut(&mut matrix) += 1;
        *I(1, 0).index_mut(&mut matrix) += 1;
        *I(1, 1).index_mut(&mut matrix) += 1;
        *I(1, 2).index_mut(&mut matrix) += 1;
        assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]])
    }

    #[test]
    #[should_panic]
    fn test_single_element_index_out_of_bounds() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];
        let _ = I(2, 3).index(&matrix);
    }
}
