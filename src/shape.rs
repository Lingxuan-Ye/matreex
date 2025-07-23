//! Describes the shape of a matrix.

use crate::error::{Error, Result};
use crate::order::Order;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A struct representing the shape of a [`Matrix<T>`].
///
/// [`Matrix<T>`]: crate::Matrix
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Shape {
    nrows: usize,
    ncols: usize,
}

impl Shape {
    /// Creates a new [`Shape`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Shape;
    ///
    /// let shape = Shape::new(2, 3);
    /// assert_eq!(shape.nrows(), 2);
    /// assert_eq!(shape.ncols(), 3);
    /// ```
    #[inline]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols }
    }

    /// Returns the number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Shape;
    ///
    /// let shape = Shape::new(2, 3);
    /// assert_eq!(shape.nrows(), 2);
    /// ```
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Shape;
    ///
    /// let shape = Shape::new(2, 3);
    /// assert_eq!(shape.ncols(), 3);
    /// ```
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the size.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size exceeds [`usize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, Shape};
    ///
    /// let shape = Shape::new(2, 3);
    /// assert_eq!(shape.size(), Ok(6));
    ///
    /// let shape = Shape::new(2, usize::MAX);
    /// assert_eq!(shape.size(), Err(Error::SizeOverflow));
    /// ```
    #[inline]
    pub fn size(&self) -> Result<usize> {
        AsShape::size(self)
    }

    /// Transposes the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::Shape;
    ///
    /// let mut shape = Shape::new(2, 3);
    /// shape.transpose();
    /// assert_eq!(shape, Shape::new(3, 2));
    /// ```
    #[inline]
    pub fn transpose(&mut self) -> &mut Self {
        (self.nrows, self.ncols) = (self.ncols, self.nrows);
        self
    }
}

impl From<(usize, usize)> for Shape {
    #[inline]
    fn from(value: (usize, usize)) -> Self {
        let (nrows, ncols) = value;
        Self { nrows, ncols }
    }
}

impl From<[usize; 2]> for Shape {
    #[inline]
    fn from(value: [usize; 2]) -> Self {
        let [nrows, ncols] = value;
        Self { nrows, ncols }
    }
}

/// A trait for specifying the shape of a [`Matrix<T>`].
///
/// # Examples
///
/// ```
/// use matreex::{Matrix, matrix};
/// use matreex::shape::AsShape;
///
/// struct S(usize, usize);
///
/// impl AsShape for S {
///     fn nrows(&self) -> usize {
///         self.0
///     }
///
///     fn ncols(&self) -> usize {
///         self.1
///     }
/// }
///
/// let result = Matrix::with_value(S(2, 3), 0);
/// assert_eq!(result, Ok(matrix![[0, 0, 0], [0, 0, 0]]));
/// ```
///
/// [`Matrix<T>`]: crate::Matrix
pub trait AsShape {
    /// Returns the number of rows.
    fn nrows(&self) -> usize;

    /// Returns the number of columns.
    fn ncols(&self) -> usize;

    /// Returns the size.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size exceeds [`usize::MAX`].
    ///
    /// # Notes
    ///
    /// This method has a default implementation, overriding it is not
    /// recommended.
    #[inline]
    fn size(&self) -> Result<usize> {
        self.nrows()
            .checked_mul(self.ncols())
            .ok_or(Error::SizeOverflow)
    }
}

impl AsShape for Shape {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl AsShape for (usize, usize) {
    #[inline]
    fn nrows(&self) -> usize {
        self.0
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.1
    }
}

impl AsShape for [usize; 2] {
    #[inline]
    fn nrows(&self) -> usize {
        self[0]
    }

    #[inline]
    fn ncols(&self) -> usize {
        self[1]
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct MemoryShape {
    major: usize,
    minor: usize,
}

impl MemoryShape {
    pub(crate) fn major(&self) -> usize {
        self.major
    }

    pub(crate) fn minor(&self) -> usize {
        self.minor
    }

    pub(crate) fn nrows(&self, order: Order) -> usize {
        match order {
            Order::RowMajor => self.major,
            Order::ColMajor => self.minor,
        }
    }

    pub(crate) fn ncols(&self, order: Order) -> usize {
        match order {
            Order::RowMajor => self.minor,
            Order::ColMajor => self.major,
        }
    }

    pub(crate) fn stride(&self) -> Stride {
        Stride(self.minor)
    }

    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    pub(crate) fn size<T>(&self) -> Result<usize> {
        let size = self
            .major
            .checked_mul(self.minor)
            .ok_or(Error::SizeOverflow)?;
        if size_of::<T>().saturating_mul(size) > isize::MAX as usize {
            Err(Error::CapacityOverflow)
        } else {
            Ok(size)
        }
    }

    pub(crate) fn transpose(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }

    pub(crate) fn from_shape<S>(shape: S, order: Order) -> Self
    where
        S: AsShape,
    {
        let (major, minor) = match order {
            Order::RowMajor => (shape.nrows(), shape.ncols()),
            Order::ColMajor => (shape.ncols(), shape.nrows()),
        };
        Self { major, minor }
    }

    pub(crate) fn to_shape(self, order: Order) -> Shape {
        let (nrows, ncols) = match order {
            Order::RowMajor => (self.major, self.minor),
            Order::ColMajor => (self.minor, self.major),
        };
        Shape { nrows, ncols }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct Stride(usize);

impl Stride {
    pub(crate) fn major(&self) -> usize {
        self.0
    }

    pub(crate) fn minor(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_new() {
        let shape = Shape::new(2, 3);
        assert_eq!(shape.nrows(), 2);
        assert_eq!(shape.ncols(), 3);

        let shape = Shape::new(3, 2);
        assert_eq!(shape.nrows(), 3);
        assert_eq!(shape.ncols(), 2);
    }

    #[test]
    fn test_shape_size() {
        let shape = Shape::new(2, 3);
        assert_eq!(shape.size(), Ok(6));

        let shape = Shape::new(2, usize::MAX);
        assert_eq!(shape.size(), Err(Error::SizeOverflow));
    }

    #[test]
    fn test_shape_transpose() {
        let mut shape = Shape::new(2, 3);
        shape.transpose();
        assert_eq!(shape, Shape::new(3, 2));

        let mut shape = Shape::new(3, 2);
        shape.transpose();
        assert_eq!(shape, Shape::new(2, 3));
    }

    #[test]
    fn test_as_shape_size() {
        let shape = (2, 3);
        assert_eq!(shape.size(), Ok(6));

        let shape = (2, usize::MAX);
        assert_eq!(shape.size(), Err(Error::SizeOverflow));
    }
}
