use self::internal::Sealed;
use crate::error::{Error, Result};
use crate::index::{AsIndex, Index};
use crate::shape::{AsShape, Shape};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

/// # Invariants
///
/// - `self.major() * self.minor() <= usize::MAX`
/// - `self.major() * self.minor() * size_of::<T>() <= isize:::MAX as usize`
#[derive(Debug)]
pub struct Layout<T, O>
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
    #[inline]
    pub fn new(major: usize, minor: usize) -> Result<Self> {
        Self::new_with_size(major, minor).map(|(layout, _)| layout)
    }

    pub fn new_with_size(major: usize, minor: usize) -> Result<(Self, usize)> {
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

    #[inline]
    pub fn from_shape<S>(shape: S) -> Result<Self>
    where
        S: AsShape,
    {
        O::shape_to_layout(shape)
    }

    #[inline]
    pub fn from_shape_with_size<S>(shape: S) -> Result<(Self, usize)>
    where
        S: AsShape,
    {
        O::shape_to_layout_with_size(shape)
    }

    #[inline]
    pub fn to_shape(self) -> Shape
    {
        O::layout_to_shape(self)
    }

    #[inline]
    pub fn major(&self) -> usize {
        self.major
    }

    #[inline]
    pub fn minor(&self) -> usize {
        self.minor
    }

    #[inline]
    pub fn stride(&self) -> Stride {
        Stride(self.minor)
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.major * self.minor
    }

    #[inline]
    pub fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }

    #[inline]
    pub fn switch_order(self) -> Layout<T, O::Alternate> {
        Layout::new_unchecked(self.major, self.minor)
    }

    #[inline]
    pub fn cast<U>(self) -> Result<Layout<U, O>> {
        Layout::new(self.major, self.minor)
    }
}

impl<T, O> Copy for Layout<T, O> where O: Order {}

impl<T, O> Clone for Layout<T, O>
where
    O: Order,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, O> Default for Layout<T, O>
where
    O: Order,
{
    #[inline]
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
    #[inline]
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
    #[inline]
    fn eq(&self, other: &Layout<R, O>) -> bool {
        self.major == other.major && self.minor == other.minor
    }
}

impl<T, O> Eq for Layout<T, O> where O: Order {}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct Stride(usize);

impl Stride {
    #[inline]
    pub fn major(&self) -> usize {
        self.0
    }

    #[inline]
    pub fn minor(&self) -> usize {
        1
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct LayoutIndex {
    pub major: usize,
    pub minor: usize,
}

impl LayoutIndex {
    #[inline]
    pub fn new(major: usize, minor: usize) -> Self {
        Self { major, minor }
    }

    #[inline]
    pub fn from_flattened(index: usize, stride: Stride) -> Self {
        let major = index / stride.major();
        let minor = (index % stride.major()) / stride.minor();
        Self::new(major, minor)
    }

    #[inline]
    pub fn to_flattened(self, stride: Stride) -> usize {
        self.major * stride.major() + self.minor * stride.minor()
    }

    #[inline]
    pub fn swap(&mut self) -> &mut Self {
        (self.major, self.minor) = (self.minor, self.major);
        self
    }
}

pub trait Order: Sealed + Sized{
    type Alternate: Order;

    fn shape_to_layout<S, T>(shape: S) -> Result<Layout<T, Self>>
    where
        S: AsShape;

    fn shape_to_layout_with_size<S, T>(shape: S) -> Result<(Layout<T, Self>, usize)>
    where
        S: AsShape;

    fn layout_to_shape<T>(layout: Layout<T, Self>) -> Shape;

    fn index_to_layout_index<I>(index: I) -> LayoutIndex
    where
        I: AsIndex;

    fn layout_index_to_index(index: LayoutIndex) -> Index;
}

#[derive(Debug)]
pub struct RowMajor;

impl Order for RowMajor {
    type Alternate = ColMajor;

    #[inline]
    fn shape_to_layout<S, T>(shape: S) -> Result<Layout<T, Self>>
    where
        S: AsShape,
    {
        Layout::new(shape.nrows(), shape.ncols())
    }

    #[inline]
    fn shape_to_layout_with_size<S, T>(shape: S) -> Result<(Layout<T, Self>, usize)>
    where
        S: AsShape,
    {
        Layout::new_with_size(shape.nrows(), shape.ncols())
    }

    #[inline]
    fn layout_to_shape<T>(layout: Layout<T, Self>) -> Shape {
        Shape::new(layout.major(), layout.minor())
    }

    #[inline]
    fn index_to_layout_index<I>(index: I) -> LayoutIndex
    where
        I: AsIndex,
    {
        LayoutIndex::new(index.row(), index.col())
    }

    #[inline]
    fn layout_index_to_index(index: LayoutIndex) -> Index {
        Index::new(index.major, index.minor)
    }
}

#[derive(Debug)]
pub struct ColMajor;

impl Order for ColMajor {
    type Alternate = RowMajor;

    #[inline]
    fn shape_to_layout<S, T>(shape: S) -> Result<Layout<T, Self>>
    where
        S: AsShape,
    {
        Layout::new(shape.ncols(), shape.nrows())
    }

    #[inline]
    fn shape_to_layout_with_size<S, T>(shape: S) -> Result<(Layout<T, Self>, usize)>
    where
        S: AsShape,
    {
        Layout::new_with_size(shape.ncols(), shape.nrows())
    }

    #[inline]
    fn layout_to_shape<T>(layout: Layout<T, Self>) -> Shape {
        Shape::new(layout.minor(), layout.major())
    }

    #[inline]
    fn index_to_layout_index<I>(index: I) -> LayoutIndex
    where
        I: AsIndex,
    {
        LayoutIndex::new(index.col(), index.row())
    }

    #[inline]
    fn layout_index_to_index(index: LayoutIndex) -> Index {
        Index::new(index.minor, index.major)
    }
}

mod internal {
    use super::{ColMajor, RowMajor};

    pub trait Sealed {}

    impl Sealed for RowMajor {}
    impl Sealed for ColMajor {}
}
