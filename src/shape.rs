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
