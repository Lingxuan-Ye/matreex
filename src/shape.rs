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

    /// Returns the number of rows of the shape.
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

    /// Returns the number of columns of the shape.
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

    /// Returns the size of the shape.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
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
        self.nrows
            .checked_mul(self.ncols)
            .ok_or(Error::SizeOverflow)
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

    pub(crate) fn try_to_axis_shape(self, order: Order) -> Result<AxisShape> {
        self.size()?;
        Ok(self.to_axis_shape_unchecked(order))
    }

    pub(crate) fn to_axis_shape_unchecked(self, order: Order) -> AxisShape {
        let (major, minor) = match order {
            Order::RowMajor => (self.nrows, self.ncols),
            Order::ColMajor => (self.ncols, self.nrows),
        };
        AxisShape { major, minor }
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct AxisShape {
    major: usize,
    minor: usize,
}

impl AxisShape {
    pub(crate) fn major(&self) -> usize {
        self.major
    }

    pub(crate) fn minor(&self) -> usize {
        self.minor
    }

    pub(crate) fn major_stride(&self) -> usize {
        self.minor
    }

    pub(crate) fn minor_stride(&self) -> usize {
        1
    }

    pub(crate) fn size(&self) -> usize {
        self.major * self.minor
    }

    pub(crate) fn transpose(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
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

    pub(crate) fn to_shape(self, order: Order) -> Shape {
        let (nrows, ncols) = match order {
            Order::RowMajor => (self.major, self.minor),
            Order::ColMajor => (self.minor, self.major),
        };
        Shape { nrows, ncols }
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
}
