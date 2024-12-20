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
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(unsafe { matrix.get_unchecked((1, 1)) }, &4);
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
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[0, 1, 2], [3, 4, 5]];
    /// assert_eq!(unsafe { matrix.get_unchecked_mut((1, 1)) }, &mut 4);
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

/// Any type implementing this trait can index a [`Matrix<T>`].
///
/// # Examples
///
/// ```
/// use matreex::matrix;
///
/// let matrix = matrix![[0, 1, 2], [3, 4, 5]];
///
/// let index = (0, 0);
/// assert_eq!(matrix[index], 0);
///
/// let index = [1, 1];
/// assert_eq!(matrix[index], 4);
/// ```
pub trait Index {
    /// Returns the row number of the index.
    fn row(&self) -> usize;

    /// Returns the column number of the index.
    fn col(&self) -> usize;

    /// Returns `true` if the index is out of bounds for given matrix.
    fn is_out_of_bounds<T>(&self, matrix: &Matrix<T>) -> bool {
        let shape = matrix.shape();
        self.row() >= shape.nrows() || self.col() >= shape.ncols()
    }
}

unsafe impl<T, I> MatrixIndex<T> for I
where
    I: Index,
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

impl Index for (usize, usize) {
    fn row(&self) -> usize {
        self.0
    }

    fn col(&self) -> usize {
        self.1
    }
}

impl Index for [usize; 2] {
    fn row(&self) -> usize {
        self[0]
    }

    fn col(&self) -> usize {
        self[1]
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

    pub(super) fn transpose(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }

    pub(super) fn from_index<I: Index>(index: I, order: Order) -> Self {
        let (major, minor) = match order {
            Order::RowMajor => (index.row(), index.col()),
            Order::ColMajor => (index.col(), index.row()),
        };
        Self { major, minor }
    }

    pub(super) fn to_index(self, order: Order) -> impl Index {
        match order {
            Order::RowMajor => (self.major, self.minor),
            Order::ColMajor => (self.minor, self.major),
        }
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
        if self.is_out_of_bounds(matrix.shape) {
            Err(Error::IndexOutOfBounds)
        } else {
            unsafe { Ok(self.get_unchecked(matrix)) }
        }
    }

    fn get_mut(self, matrix: &mut Matrix<T>) -> Result<&mut Self::Output> {
        if self.is_out_of_bounds(matrix.shape) {
            Err(Error::IndexOutOfBounds)
        } else {
            unsafe { Ok(self.get_unchecked_mut(matrix)) }
        }
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
        if self.is_out_of_bounds(matrix.shape) {
            panic!("{}", Error::IndexOutOfBounds);
        }
        unsafe { self.get_unchecked(matrix) }
    }

    fn index_mut(self, matrix: &mut Matrix<T>) -> &mut Self::Output {
        if self.is_out_of_bounds(matrix.shape) {
            panic!("{}", Error::IndexOutOfBounds);
        }
        unsafe { self.get_unchecked_mut(matrix) }
    }
}

pub(super) fn unflatten_index(index: usize, order: Order, shape: AxisShape) -> impl Index {
    AxisIndex::from_flattened(index, shape).to_index(order)
}

pub(super) fn flatten_index<I: Index>(index: I, order: Order, shape: AxisShape) -> usize {
    AxisIndex::from_index(index, order).to_flattened(shape)
}

pub(super) fn transpose_flattened_index(index: usize, mut shape: AxisShape) -> usize {
    let mut index = AxisIndex::from_flattened(index, shape);
    index.transpose();
    shape.transpose();
    index.to_flattened(shape)
}

mod internal {
    pub trait Sealed {}

    impl<I: super::Index> Sealed for I {}

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
    fn test_trait_index() {
        let matrix = matrix![[0, 1, 2], [3, 4, 5]];

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
}
