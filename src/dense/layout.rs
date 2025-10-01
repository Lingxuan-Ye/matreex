use self::internal::Sealed;
use crate::error::{Error, Result};
use crate::index::{AsIndex, Index};
use crate::shape::{AsShape, Shape};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

#[derive(Debug)]
pub struct RowMajor;

#[derive(Debug)]
pub struct ColMajor;

pub trait Order: Sealed {
    type Alternate: Order;
    const KIND: OrderKind;
}

impl Order for RowMajor {
    type Alternate = ColMajor;
    const KIND: OrderKind = OrderKind::RowMajor;
}

impl Order for ColMajor {
    type Alternate = RowMajor;
    const KIND: OrderKind = OrderKind::ColMajor;
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum OrderKind {
    #[default]
    RowMajor,
    ColMajor,
}

/// # Invariants
///
/// - `self.major * self.minor <= usize::MAX`
/// - `self.major * self.minor * size_of::<T>() <= isize:::MAX as usize`
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

    pub(super) fn from_shape<S>(shape: S) -> Result<Self>
    where
        S: AsShape,
    {
        match O::KIND {
            OrderKind::RowMajor => Self::new(shape.nrows(), shape.ncols()),
            OrderKind::ColMajor => Self::new(shape.ncols(), shape.nrows()),
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

    pub(super) fn to_shape(self) -> Shape {
        match O::KIND {
            OrderKind::RowMajor => Shape::new(self.major, self.minor),
            OrderKind::ColMajor => Shape::new(self.minor, self.major),
        }
    }

    pub(super) fn to_alternate_order(self) -> Layout<T, O::Alternate> {
        Layout::new_unchecked(self.major, self.minor)
    }

    pub(super) fn cast<U>(self) -> Result<Layout<U, O>> {
        Layout::new(self.major, self.minor)
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

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(super) struct LayoutIndex {
    pub(super) major: usize,
    pub(super) minor: usize,
}

impl LayoutIndex {
    fn new(major: usize, minor: usize) -> Self {
        Self { major, minor }
    }

    pub(super) fn from_index<I, O>(index: I) -> Self
    where
        I: AsIndex,
        O: Order,
    {
        match O::KIND {
            OrderKind::RowMajor => Self::new(index.row(), index.col()),
            OrderKind::ColMajor => Self::new(index.col(), index.row()),
        }
    }

    pub(super) fn to_index<O>(self) -> Index
    where
        O: Order,
    {
        match O::KIND {
            OrderKind::RowMajor => Index::new(self.major, self.minor),
            OrderKind::ColMajor => Index::new(self.minor, self.major),
        }
    }

    pub(super) fn from_flattened(index: usize, stride: Stride) -> Self {
        let major = index / stride.major();
        let minor = (index % stride.major()) / stride.minor();
        Self::new(major, minor)
    }

    pub(super) fn to_flattened(self, stride: Stride) -> usize {
        self.major * stride.major() + self.minor * stride.minor()
    }

    pub(super) fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }
}

impl Index {
    pub(super) fn from_flattened<O>(index: usize, stride: Stride) -> Self
    where
        O: Order,
    {
        LayoutIndex::from_flattened(index, stride).to_index::<O>()
    }

    pub(super) fn to_flattened<O>(self, stride: Stride) -> usize
    where
        O: Order,
    {
        LayoutIndex::from_index::<_, O>(self).to_flattened(stride)
    }
}

mod internal {
    use super::{ColMajor, RowMajor};

    pub trait Sealed {}

    impl Sealed for RowMajor {}
    impl Sealed for ColMajor {}
}
