use super::order::{Order, OrderKind};
use crate::error::{Error, Result};
use crate::shape::{AsShape, Shape};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::panic::{RefUnwindSafe, UnwindSafe};

#[cfg(feature = "serde")]
mod serde;

/// The memory layout of a [`Matrix<T, O>`].
///
/// # Invariants
///
/// - `self.major() * self.minor() <= usize::MAX`
/// - `self.major() * self.minor() * size_of::<T>() <= isize::MAX as usize`
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
    /// Creates a new [`Layout<T, O>`] with the specified axis lengths, returning
    /// the layout and its size.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if `major * minor` exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes for the
    ///   corresponding matrix exceeds [`isize::MAX`].
    fn new(major: usize, minor: usize) -> Result<(Self, usize)> {
        let size = major.checked_mul(minor).ok_or(Error::SizeOverflow)?;
        Self::check_size(size)?;
        Ok((Self::new_unchecked(major, minor), size))
    }

    /// Creates a new [`Layout<T, O>`] with the specified axis lengths, without
    /// performing any invariant checking.
    fn new_unchecked(major: usize, minor: usize) -> Self {
        Self {
            major,
            minor,
            element: PhantomData,
            order: PhantomData,
        }
    }

    /// Creates a new [`Layout<T, O>`] with the specified shape, returning
    /// the layout and its size.
    ///
    /// # Errors
    ///
    /// - [`Error::SizeOverflow`] if the size of the shape exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes for the
    ///   corresponding matrix exceeds [`isize::MAX`].
    pub(super) fn from_shape<S>(shape: S) -> Result<(Self, usize)>
    where
        S: AsShape,
    {
        match O::KIND {
            OrderKind::RowMajor => Self::new(shape.nrows(), shape.ncols()),
            OrderKind::ColMajor => Self::new(shape.ncols(), shape.nrows()),
        }
    }

    /// Creates a new [`Layout<T, O>`] with the specified shape, without
    /// performing any invariant checking.
    pub(super) fn from_shape_unchecked<S>(shape: S) -> Self
    where
        S: AsShape,
    {
        match O::KIND {
            OrderKind::RowMajor => Self::new_unchecked(shape.nrows(), shape.ncols()),
            OrderKind::ColMajor => Self::new_unchecked(shape.ncols(), shape.nrows()),
        }
    }

    /// Converts this layout to a [`Shape`].
    pub(super) fn to_shape(self) -> Shape {
        match O::KIND {
            OrderKind::RowMajor => Shape::new(self.major, self.minor),
            OrderKind::ColMajor => Shape::new(self.minor, self.major),
        }
    }

    /// Converts to a layout with the specified storage order.
    pub(super) fn with_order<P>(self) -> Layout<T, P>
    where
        P: Order,
    {
        Layout::new_unchecked(self.major, self.minor)
    }

    /// Converts to a layout with the specified element type.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes for the
    ///   corresponding matrix of the returned layout exceeds [`isize::MAX`].
    pub(super) fn cast<U>(self) -> Result<Layout<U, O>> {
        Layout::<U, O>::check_size(self.size())?;
        Ok(Layout::new_unchecked(self.major, self.minor))
    }

    /// Returns the major axis length.
    pub(super) fn major(&self) -> usize {
        self.major
    }

    /// Returns the minor axis length.
    pub(super) fn minor(&self) -> usize {
        self.minor
    }

    /// Returns the stride.
    pub(super) fn stride(&self) -> Stride {
        Stride(self.minor)
    }

    /// Returns the size.
    pub(super) fn size(&self) -> usize {
        self.major * self.minor
    }

    /// Swaps the major and minor axes.
    pub(super) fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }
}

impl<T, O> Layout<T, O>
where
    O: Order,
{
    /// The maximum layout size.
    const MAX_SIZE: usize = if size_of::<T>() == 0 {
        usize::MAX
    } else {
        isize::MAX as usize / size_of::<T>()
    };

    /// Checks if the given size is valid.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes for a
    ///   matrix with the given number of elements exceeds [`isize::MAX`].
    pub(super) fn check_size(size: usize) -> Result<()> {
        if size > Self::MAX_SIZE {
            Err(Error::CapacityOverflow)
        } else {
            Ok(())
        }
    }
}

unsafe impl<T, O> Send for Layout<T, O> where O: Order {}
unsafe impl<T, O> Sync for Layout<T, O> where O: Order {}
impl<T, O> Unpin for Layout<T, O> where O: Order {}
impl<T, O> UnwindSafe for Layout<T, O> where O: Order {}
impl<T, O> RefUnwindSafe for Layout<T, O> where O: Order {}

impl<T, O> Clone for Layout<T, O>
where
    O: Order,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, O> Copy for Layout<T, O> where O: Order {}

impl<T, O> Default for Layout<T, O>
where
    O: Order,
{
    fn default() -> Self {
        Self::new_unchecked(0, 0)
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
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub(super) struct Stride(usize);

impl Stride {
    /// Returns the major axis stride.
    pub(super) fn major(&self) -> usize {
        self.0
    }

    /// Returns the minor axis stride.
    pub(super) fn minor(&self) -> usize {
        1
    }
}
