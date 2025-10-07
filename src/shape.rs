//! Matrix shape representations.

use crate::error::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Shape {
    nrows: usize,
    ncols: usize,
}

impl Shape {
    #[inline]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn size(&self) -> Result<usize> {
        AsShape::size(self)
    }

    #[inline]
    pub fn swap(&mut self) -> &mut Self {
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

pub trait AsShape {
    fn nrows(&self) -> usize;

    fn ncols(&self) -> usize;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_new() {
        let shape = Shape::new(2, 3);
        assert_eq!(shape, Shape { nrows: 2, ncols: 3 });

        let shape: Shape = Shape::new(3, 2);
        assert_eq!(shape, Shape { nrows: 3, ncols: 2 });
    }

    #[test]
    fn test_shape_nrows() {
        let shape = Shape::new(2, 3);
        assert_eq!(shape.nrows(), 2);

        let shape: Shape = Shape::new(3, 2);
        assert_eq!(shape.nrows(), 3);
    }

    #[test]
    fn test_shape_ncols() {
        let shape = Shape::new(2, 3);
        assert_eq!(shape.ncols(), 3);

        let shape: Shape = Shape::new(3, 2);
        assert_eq!(shape.ncols(), 2);
    }

    #[test]
    fn test_shape_size() {
        let shape = Shape::new(2, 3);
        assert_eq!(shape.size(), Ok(6));

        let shape = Shape::new(2, usize::MAX);
        assert_eq!(shape.size(), Err(Error::SizeOverflow));

        let shape = Shape::new(usize::MAX, 3);
        assert_eq!(shape.size(), Err(Error::SizeOverflow));
    }

    #[test]
    fn test_shape_swap() {
        let mut shape = Shape::new(2, 3);
        shape.swap();
        assert_eq!(shape, Shape::new(3, 2));

        let mut shape = Shape::new(3, 2);
        shape.swap();
        assert_eq!(shape, Shape::new(2, 3));
    }

    #[test]
    fn test_as_shape_nrows() {
        let shape = Shape::new(2, 3);
        assert_eq!(AsShape::nrows(&shape), 2);

        let shape = Shape::new(3, 2);
        assert_eq!(AsShape::nrows(&shape), 3);

        let shape = (2, 3);
        assert_eq!(AsShape::nrows(&shape), 2);

        let shape = (3, 2);
        assert_eq!(AsShape::nrows(&shape), 3);

        let shape = [2, 3];
        assert_eq!(AsShape::nrows(&shape), 2);

        let shape = [3, 2];
        assert_eq!(AsShape::nrows(&shape), 3);
    }

    #[test]
    fn test_as_shape_ncols() {
        let shape = Shape::new(2, 3);
        assert_eq!(AsShape::ncols(&shape), 3);

        let shape = Shape::new(3, 2);
        assert_eq!(AsShape::ncols(&shape), 2);

        let shape = (2, 3);
        assert_eq!(AsShape::ncols(&shape), 3);

        let shape = (3, 2);
        assert_eq!(AsShape::ncols(&shape), 2);

        let shape = [2, 3];
        assert_eq!(AsShape::ncols(&shape), 3);

        let shape = [3, 2];
        assert_eq!(AsShape::ncols(&shape), 2);
    }

    #[test]
    fn test_as_shape_size() {
        let shape = Shape::new(2, 3);
        assert_eq!(AsShape::size(&shape), Ok(6));

        let shape = Shape::new(2, usize::MAX);
        assert_eq!(AsShape::size(&shape), Err(Error::SizeOverflow));

        let shape = Shape::new(usize::MAX, 3);
        assert_eq!(shape.size(), Err(Error::SizeOverflow));

        let shape = (2, 3);
        assert_eq!(AsShape::size(&shape), Ok(6));

        let shape = (2, usize::MAX);
        assert_eq!(AsShape::size(&shape), Err(Error::SizeOverflow));

        let shape = (usize::MAX, 3);
        assert_eq!(shape.size(), Err(Error::SizeOverflow));

        let shape = [2, 3];
        assert_eq!(AsShape::size(&shape), Ok(6));

        let shape = [2, usize::MAX];
        assert_eq!(AsShape::size(&shape), Err(Error::SizeOverflow));

        let shape = [usize::MAX, 3];
        assert_eq!(shape.size(), Err(Error::SizeOverflow));
    }
}
