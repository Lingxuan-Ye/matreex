//! Traits and types for matrix indexing.

use crate::error::{Error, Result};
use crate::shape::AsShape;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// A helper trait for indexing operations on a matrix.
///
/// # Safety
///
/// Implementors must ensure that if any default implementation of [`get`] or
/// [`get_mut`] is used, then [`is_out_of_bounds`] is implemented correctly and
/// [`ensure_in_bounds`] is not overridden. Failing to do so may result in an
/// out-of-bounds memory access, leading to *[undefined behavior]*.
///
/// [`is_out_of_bounds`]: MatrixIndex::is_out_of_bounds
/// [`ensure_in_bounds`]: MatrixIndex::ensure_in_bounds
/// [`get`]: MatrixIndex::get
/// [`get_mut`]: MatrixIndex::get_mut
/// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
pub unsafe trait MatrixIndex<M>: Sized {
    /// The shared output type.
    type Output<'a>
    where
        M: 'a;

    /// The mutable output type.
    type OutputMut<'a>
    where
        M: 'a;

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

    /// Returns a shared output at this location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn get(self, matrix: &M) -> Result<Self::Output<'_>> {
        self.ensure_in_bounds(matrix)?;
        unsafe { Ok(self.get_unchecked(matrix)) }
    }

    /// Returns a mutable output at this location.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if out of bounds.
    fn get_mut(self, matrix: &mut M) -> Result<Self::OutputMut<'_>> {
        self.ensure_in_bounds(matrix)?;
        unsafe { Ok(self.get_unchecked_mut(matrix)) }
    }

    /// Returns a shared output at this location, without performing any bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting output is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked(self, matrix: &M) -> Self::Output<'_>;

    /// Returns a mutable output at this location, without performing any bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting output is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn get_unchecked_mut(self, matrix: &mut M) -> Self::OutputMut<'_>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn test_index_new() {
        let index = Index::new(2, 3);
        assert_eq!(index, Index { row: 2, col: 3 });

        let index = Index::new(3, 2);
        assert_eq!(index, Index { row: 3, col: 2 });
    }

    #[test]
    fn test_index_from_wrapping_index() {
        let shape = Shape::new(2, 3);
        for row in (-6..=6).step_by(2) {
            for col in (-6..=6).step_by(3) {
                let index = WrappingIndex::new(row, col);
                let index = Index::from_wrapping_index(index, shape);
                assert_eq!(index, Index::new(0, 0));
            }
        }
    }

    #[test]
    fn test_index_swap() {
        let mut index = Index::new(2, 3);
        index.swap();
        assert_eq!(index, Index::new(3, 2));

        let mut index = Index::new(3, 2);
        index.swap();
        assert_eq!(index, Index::new(2, 3));
    }

    #[test]
    fn test_as_index_row() {
        let index = Index::new(2, 3);
        assert_eq!(AsIndex::row(&index), 2);

        let index = Index::new(3, 2);
        assert_eq!(AsIndex::row(&index), 3);

        let index = (2, 3);
        assert_eq!(AsIndex::row(&index), 2);

        let index = (3, 2);
        assert_eq!(AsIndex::row(&index), 3);

        let index = [2, 3];
        assert_eq!(AsIndex::row(&index), 2);

        let index = [3, 2];
        assert_eq!(AsIndex::row(&index), 3);
    }

    #[test]
    fn test_as_index_col() {
        let index = Index::new(2, 3);
        assert_eq!(AsIndex::col(&index), 3);

        let index = Index::new(3, 2);
        assert_eq!(AsIndex::col(&index), 2);

        let index = (2, 3);
        assert_eq!(AsIndex::col(&index), 3);

        let index = (3, 2);
        assert_eq!(AsIndex::col(&index), 2);

        let index = [2, 3];
        assert_eq!(AsIndex::col(&index), 3);

        let index = [3, 2];
        assert_eq!(AsIndex::col(&index), 2);
    }

    #[test]
    fn test_wrapping_index_new() {
        let index = WrappingIndex::new(2, 3);
        assert_eq!(index, WrappingIndex { row: 2, col: 3 });

        let index = WrappingIndex::new(3, 2);
        assert_eq!(index, WrappingIndex { row: 3, col: 2 });
    }

    #[test]
    fn test_wrapping_index_to_index() {
        let shape = Shape::new(2, 3);
        for row in (-6..=6).step_by(2) {
            for col in (-6..=6).step_by(3) {
                let index = WrappingIndex::new(row, col);
                let index = index.to_index(shape);
                assert_eq!(index, Index::new(0, 0));
            }
        }
    }

    #[test]
    fn test_wrapping_index_swap() {
        let mut index = WrappingIndex::new(2, 3);
        index.swap();
        assert_eq!(index, WrappingIndex::new(3, 2));

        let mut index = WrappingIndex::new(3, 2);
        index.swap();
        assert_eq!(index, WrappingIndex::new(2, 3));
    }
}
