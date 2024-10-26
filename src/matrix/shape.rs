use super::order::Order;
use crate::error::{Error, Result};

/// Any type implementing this trait can be a `shape` argument for
/// [`Matrix<T>`] constructors.
///
/// # Examples
///
/// ```
/// use matreex::Matrix;
/// # use matreex::Result;
///
/// # fn main() -> Result<()> {
/// let foo = Matrix::<i32>::with_shape((2, 3))?;
/// let bar = Matrix::<i32>::with_shape([2, 3])?;
/// # Ok(())
/// # }
/// ```
///
/// [`Matrix<T>`]: crate::matrix::Matrix<T>
pub trait Shape {
    /// Returns the number of rows.
    fn nrows(&self) -> usize;

    /// Returns the number of columns.
    fn ncols(&self) -> usize;

    /// Returns the size of the shape.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    fn size(&self) -> Result<usize> {
        self.nrows()
            .checked_mul(self.ncols())
            .ok_or(Error::SizeOverflow)
    }

    /// Returns `true` if this shape has the same rows and columns as `other`.
    fn equal<S: Shape>(&self, other: &S) -> bool {
        self.nrows() == other.nrows() && self.ncols() == other.ncols()
    }
}

impl Shape for (usize, usize) {
    fn nrows(&self) -> usize {
        self.0
    }

    fn ncols(&self) -> usize {
        self.1
    }
}

impl Shape for [usize; 2] {
    fn nrows(&self) -> usize {
        self[0]
    }

    fn ncols(&self) -> usize {
        self[1]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AxisShape {
    major: usize,
    minor: usize,
}

impl AxisShape {
    pub(super) fn major(&self) -> usize {
        self.major
    }

    pub(super) fn minor(&self) -> usize {
        self.minor
    }

    pub(super) fn major_stride(&self) -> usize {
        self.minor
    }

    pub(super) const fn minor_stride(&self) -> usize {
        1
    }

    pub(super) fn size(&self) -> usize {
        self.major * self.minor
    }

    pub(super) fn transpose(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }

    pub(super) fn interpret(&self, order: Order) -> impl Shape {
        match order {
            Order::RowMajor => (self.major, self.minor),
            Order::ColMajor => (self.minor, self.major),
        }
    }

    pub(super) fn interpret_nrows(&self, order: Order) -> usize {
        match order {
            Order::RowMajor => self.major,
            Order::ColMajor => self.minor,
        }
    }

    pub(super) fn interpret_ncols(&self, order: Order) -> usize {
        match order {
            Order::RowMajor => self.minor,
            Order::ColMajor => self.major,
        }
    }

    pub(super) fn from_shape_unchecked<S: Shape>(shape: S, order: Order) -> Self {
        let (major, minor) = match order {
            Order::RowMajor => (shape.nrows(), shape.ncols()),
            Order::ColMajor => (shape.ncols(), shape.nrows()),
        };
        Self { major, minor }
    }

    pub(super) fn try_from_shape<S: Shape>(shape: S, order: Order) -> Result<Self> {
        shape.size()?;
        Ok(Self::from_shape_unchecked(shape, order))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_shape() {
        assert_eq!((2, 3).nrows(), 2);
        assert_eq!((2, 3).ncols(), 3);
        assert_eq!((2, 3).size(), Ok(6));
        assert_eq!((2, usize::MAX).size(), Err(Error::SizeOverflow));

        assert_eq!([2, 3].nrows(), 2);
        assert_eq!([2, 3].ncols(), 3);
        assert_eq!([2, 3].size(), Ok(6));
        assert_eq!([2, usize::MAX].size(), Err(Error::SizeOverflow));

        assert!(Shape::equal(&(2, 3), &[2, 3]));
    }
}
