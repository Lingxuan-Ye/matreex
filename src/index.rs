//! Matrix index representations.

use crate::shape::AsShape;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A two-dimensional index type.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
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

    /// Creates a new [`Index`] from a [`WrappingIndex`].
    ///
    /// # Panics
    ///
    /// Panics if `shape.nrows() == 0` or `shape.ncols() == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Index, WrappingIndex};
    ///
    /// let index = WrappingIndex::new(-1, 4);
    /// let index = Index::from_wrapping_index(index, (2, 3));
    /// assert_eq!(index, Index::new(1, 1));
    /// ```
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

/// A trait for two-dimensional index types.
pub trait AsIndex {
    /// Returns the row index.
    fn row(&self) -> usize;

    /// Returns the column index.
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

impl<I> AsIndex for &I
where
    I: AsIndex,
{
    #[inline]
    fn row(&self) -> usize {
        (*self).row()
    }

    #[inline]
    fn col(&self) -> usize {
        (*self).col()
    }
}

/// A two-dimensional index type with wrapping behavior.
///
/// [`WrappingIndex`] is the only type that performs wrapping indexing.
/// The design is based on the following considerations:
///
/// - Wrapping indexing does not follow standard indexing conventions,
///   so it should always be used explicitly.
/// - Both `(isize, isize)` and `[isize; 2]` are not sufficiently
///   distinguishable from their `usize` counterparts, which would
///   introduce ambiguity and require explicit type annotations.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
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

    /// Converts this wrapping index to an [`Index`].
    ///
    /// # Panics
    ///
    /// Panics if `shape.nrows() == 0` or `shape.ncols() == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Index, WrappingIndex};
    ///
    /// let index = WrappingIndex::new(-1, 4).to_index((2, 3));
    /// assert_eq!(index, Index::new(1, 1));
    /// ```
    pub fn to_index<S>(self, shape: S) -> Index
    where
        S: AsShape,
    {
        Index::from_wrapping_index(self, shape)
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
    #[should_panic]
    fn test_index_from_wrapping_index_fails() {
        let shape = Shape::new(0, 3);
        let index = WrappingIndex::new(2, 3);
        Index::from_wrapping_index(index, shape);
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
    #[should_panic]
    fn test_wrapping_index_to_index_fails() {
        let shape = Shape::new(0, 3);
        let index = WrappingIndex::new(2, 3);
        index.to_index(shape);
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
