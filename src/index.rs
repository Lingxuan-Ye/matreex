use crate::error::{Error, Result};
use crate::shape::AsShape;

/// A helper trait for indexing operations on a matrix.
///
/// # Safety
///
/// Implementations of this trait have to promise that if any default
/// implementation of [`get`], [`get_mut`], [`index`] or [`index_mut`]
/// is used, then [`is_out_of_bounds`] is implemented correctly and
/// [`ensure_in_bounds`] is not overridden. Failing to do so may result
/// in an out-of-bounds memory access, leading to *[undefined behavior]*.
///
/// [`is_out_of_bounds`]: MatrixIndex::is_out_of_bounds
/// [`ensure_in_bounds`]: MatrixIndex::ensure_in_bounds
/// [`get`]: MatrixIndex::get
/// [`get_mut`]: MatrixIndex::get_mut
/// [`index`]: MatrixIndex::index
/// [`index_mut`]: MatrixIndex::index_mut
/// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
pub unsafe trait MatrixIndex<M>: Sized {
    /// The output type returned by methods.
    type Output;

    /// Returns `true` if the index is out of bounds for the given matrix.
    fn is_out_of_bounds(&self, matrix: &M) -> bool;

    /// Ensures the index is in bounds for the given matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn ensure_in_bounds(&self, matrix: &M) -> Result<&Self> {
        if self.is_out_of_bounds(matrix) {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self)
        }
    }

    /// Returns a shared reference to the output at this location,
    /// if in bounds.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    ///
    /// [`Output`]: MatrixIndex::Output
    fn get(self, matrix: &M) -> Result<&Self::Output> {
        self.ensure_in_bounds(matrix)?;
        unsafe { Ok(self.get_unchecked(matrix)) }
    }

    /// Returns a mutable reference to the output at this location,
    /// if in bounds.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    ///
    /// [`OutputMut`]: MatrixIndex::OutputMut
    fn get_mut(self, matrix: &mut M) -> Result<&mut Self::Output> {
        self.ensure_in_bounds(matrix)?;
        unsafe { Ok(self.get_unchecked_mut(matrix)) }
    }

    /// Returns a shared reference to the output at this location,
    /// without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined
    /// behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: &M) -> &Self::Output;

    /// Returns a mutable reference to the output at this location,
    /// without performing any bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined
    /// behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: &mut M) -> &mut Self::Output;

    /// Returns a shared reference to the output at this location.
    ///
    /// # Panics
    ///
    /// Panics if out of bounds.
    fn index(self, matrix: &M) -> &Self::Output {
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
    fn index_mut(self, matrix: &mut M) -> &mut Self::Output {
        match self.get_mut(matrix) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Index {
    pub row: usize,
    pub col: usize,
}

impl Index {
    #[inline]
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    pub fn from_as_index<I>(index: I) -> Self
    where
        I: AsIndex,
    {
        let row = index.row();
        let col = index.col();
        Self { row, col }
    }

    pub fn from_wrapping_index<S>(index: WrappingIndex, shape: S) -> Self
    where
        S: AsShape,
    {
        fn rem_euclid(lhs: isize, rhs: usize) -> usize {
            if lhs < 0 {
                (rhs - lhs.unsigned_abs() % rhs) % rhs
            } else {
                lhs as usize % rhs
            }
        }

        let row = rem_euclid(index.row, shape.nrows());
        let col = rem_euclid(index.col, shape.ncols());
        Self { row, col }
    }

    #[inline]
    pub fn swap(&mut self) -> &mut Self {
        (self.row, self.col) = (self.col, self.row);
        self
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

pub trait AsIndex {
    fn row(&self) -> usize;

    fn col(&self) -> usize;
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

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct WrappingIndex {
    pub row: isize,
    pub col: isize,
}

impl WrappingIndex {
    #[inline]
    pub fn new(row: isize, col: isize) -> Self {
        Self { row, col }
    }

    pub fn to_index<S>(self, shape: S) -> Index
    where
        S: AsShape,
    {
        Index::from_wrapping_index(self, shape)
    }

    #[inline]
    pub fn swap(&mut self) -> &mut Self {
        (self.row, self.col) = (self.col, self.row);
        self
    }
}
