//! Types for matrix layout.

use self::internal::Sealed;
use crate::error::{Error, Result};
use crate::shape::{AsShape, Shape};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

#[cfg(feature = "serde")]
mod serde;

/// A marker type representing row-major order.
///
/// In this order, elements are stored row by row, with consecutive
/// elements of a row stored contiguously in memory.
///
/// For column-major order see [`ColMajor`].
#[derive(Debug)]
pub struct RowMajor;

/// A marker type representing column-major order.
///
/// In this order, elements are stored column by column, with consecutive
/// elements of a column stored contiguously in memory.
///
/// For row-major order see [`RowMajor`].
#[derive(Debug)]
pub struct ColMajor;

/// A sealed trait restricting allowed storage orders.
///
/// The allowed orders are:
/// - [`RowMajor`]
/// - [`ColMajor`]
pub trait Order: Sealed {
    #[doc(hidden)]
    const KIND: OrderKind;
}

impl Order for RowMajor {
    const KIND: OrderKind = OrderKind::RowMajor;
}

impl Order for ColMajor {
    const KIND: OrderKind = OrderKind::ColMajor;
}

#[doc(hidden)]
#[derive(Debug, PartialEq)]
pub enum OrderKind {
    RowMajor,
    ColMajor,
}

/// A struct representing the memory layout of a [`Matrix<T, O>`].
///
/// # Invariants
///
/// - `self.major() * self.minor() <= usize::MAX`
/// - `self.major() * self.minor() * size_of::<T>() <= isize:::MAX as usize`
///
/// [`Matrix<T, O>`]: crate::dense::Matrix
#[derive(Debug)]
pub(super) struct Layout<T, O>
where
    O: Order,
{
    major: usize,
    minor: usize,
    element: PhantomData<T>,
    order: PhantomData<O>,
}

impl<T, O> Layout<T, O>
where
    O: Order,
{
    fn new(major: usize, minor: usize) -> Result<Self> {
        Self::new_with_size(major, minor).map(|(layout, _)| layout)
    }

    fn new_with_size(major: usize, minor: usize) -> Result<(Self, usize)> {
        let size = major.checked_mul(minor).ok_or(Error::SizeOverflow)?;
        if size.saturating_mul(size_of::<T>()) > isize::MAX as usize {
            Err(Error::CapacityOverflow)
        } else {
            Ok((Self::new_unchecked(major, minor), size))
        }
    }

    fn new_unchecked(major: usize, minor: usize) -> Self {
        Self {
            major,
            minor,
            element: PhantomData,
            order: PhantomData,
        }
    }

    pub(super) fn from_shape_with_size<S>(shape: S) -> Result<(Self, usize)>
    where
        S: AsShape,
    {
        match O::KIND {
            OrderKind::RowMajor => Self::new_with_size(shape.nrows(), shape.ncols()),
            OrderKind::ColMajor => Self::new_with_size(shape.ncols(), shape.nrows()),
        }
    }

    pub(super) fn from_shape_unchecked<S>(shape: S) -> Self
    where
        S: AsShape,
    {
        match O::KIND {
            OrderKind::RowMajor => Self::new_unchecked(shape.nrows(), shape.ncols()),
            OrderKind::ColMajor => Self::new_unchecked(shape.ncols(), shape.nrows()),
        }
    }

    pub(super) fn to_shape(self) -> Shape {
        match O::KIND {
            OrderKind::RowMajor => Shape::new(self.major, self.minor),
            OrderKind::ColMajor => Shape::new(self.minor, self.major),
        }
    }

    pub(super) fn with_order<P>(self) -> Layout<T, P>
    where
        P: Order,
    {
        Layout::new_unchecked(self.major, self.minor)
    }

    pub(super) fn cast<U>(self) -> Result<Layout<U, O>> {
        if self.size().saturating_mul(size_of::<U>()) > isize::MAX as usize {
            Err(Error::CapacityOverflow)
        } else {
            Ok(Layout::new_unchecked(self.major, self.minor))
        }
    }

    pub(super) fn major(&self) -> usize {
        self.major
    }

    pub(super) fn minor(&self) -> usize {
        self.minor
    }

    pub(super) fn stride(&self) -> Stride {
        Stride(self.minor)
    }

    pub(super) fn size(&self) -> usize {
        self.major * self.minor
    }

    pub(super) fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }
}

impl<T, O> Copy for Layout<T, O> where O: Order {}

impl<T, O> Clone for Layout<T, O>
where
    O: Order,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, O> Default for Layout<T, O>
where
    O: Order,
{
    fn default() -> Self {
        Self {
            major: usize::default(),
            minor: usize::default(),
            element: PhantomData,
            order: PhantomData,
        }
    }
}

impl<T, O> Hash for Layout<T, O>
where
    O: Order,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.major.hash(state);
        self.minor.hash(state);
    }
}

impl<L, R, O> PartialEq<Layout<R, O>> for Layout<L, O>
where
    O: Order,
{
    fn eq(&self, other: &Layout<R, O>) -> bool {
        self.major == other.major && self.minor == other.minor
    }
}

impl<T, O> Eq for Layout<T, O> where O: Order {}

/// A struct representing the stride of a [`Matrix<T, O>`].
///
/// [`Matrix<T, O>`]: crate::dense::Matrix
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(super) struct Stride(usize);

impl Stride {
    pub(super) fn major(&self) -> usize {
        self.0
    }

    pub(super) fn minor(&self) -> usize {
        1
    }
}

mod internal {
    use super::{ColMajor, RowMajor};

    pub trait Sealed {}

    impl Sealed for RowMajor {}
    impl Sealed for ColMajor {}
}
