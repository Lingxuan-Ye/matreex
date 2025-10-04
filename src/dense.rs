pub use self::layout::{ColMajor, RowMajor};

use self::layout::{Layout, Order, OrderKind, Stride};
use crate::error::Result;
use crate::index::Index;
use crate::shape::Shape;
use alloc::vec::Vec;
use core::cmp;
use core::hash::{Hash, Hasher};
use core::ptr;

#[cfg(feature = "parallel")]
pub mod parallel;

mod arithmetic;
mod construct;
mod fmt;
mod index;
mod iter;
mod layout;
mod resize;

pub struct Matrix<T, O = RowMajor>
where
    O: Order,
{
    layout: Layout<T, O>,
    data: Vec<T>,
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn shape(&self) -> Shape {
        self.layout.to_shape()
    }

    pub fn nrows(&self) -> usize {
        self.shape().nrows()
    }

    pub fn ncols(&self) -> usize {
        self.shape().ncols()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    fn major(&self) -> usize {
        self.layout.major()
    }

    fn minor(&self) -> usize {
        self.layout.minor()
    }

    fn stride(&self) -> Stride {
        self.layout.stride()
    }
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn transpose(&mut self) -> &mut Self {
        if size_of::<T>() == 0 {
            self.layout.swap();
            return self;
        }

        let old_stride = self.stride();
        self.layout.swap();
        let new_stride = self.stride();
        let size = self.size();
        unsafe { self.data.set_len(0) };
        let mut new_data = Vec::<T>::with_capacity(size);
        let old_base = self.data.as_ptr();
        let new_base = new_data.as_mut_ptr();

        for index in 0..size {
            unsafe {
                let src = old_base.add(index);
                let index = Index::from_flattened::<O>(index, old_stride)
                    .swap()
                    .to_flattened::<O>(new_stride);
                let dst = new_base.add(index);
                ptr::copy_nonoverlapping(src, dst, 1);
            }
        }

        unsafe { new_data.set_len(size) };
        self.data = new_data;

        self
    }

    pub fn into_transposed_no_rearrange(self) -> Matrix<T, O::Alternate> {
        self.into_alternate_order_no_rearrange()
    }

    pub fn into_alternate_order(mut self) -> Matrix<T, O::Alternate> {
        self.transpose();
        self.into_alternate_order_no_rearrange()
    }

    fn into_alternate_order_no_rearrange(self) -> Matrix<T, O::Alternate> {
        let layout = self.layout.to_alternate_order();
        let data = self.data;
        Matrix { layout, data }
    }

    pub fn into_row_major(mut self) -> Matrix<T, RowMajor> {
        if O::KIND != OrderKind::RowMajor {
            self.transpose();
        }
        let layout = self.layout.to_row_major();
        let data = self.data;
        Matrix { layout, data }
    }

    pub fn into_col_major(mut self) -> Matrix<T, ColMajor> {
        if O::KIND != OrderKind::ColMajor {
            self.transpose();
        }
        let layout = self.layout.to_col_major();
        let data = self.data;
        Matrix { layout, data }
    }

    pub fn shrink_to_fit(&mut self) -> &mut Self {
        self.data.shrink_to_fit();
        self
    }

    pub fn shrink_to(&mut self, min_capacity: usize) -> &mut Self {
        self.data.shrink_to(min_capacity);
        self
    }

    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.data.contains(value)
    }

    pub fn overwrite(&mut self, src: &Self) -> &mut Self
    where
        T: Clone,
    {
        let src_stride = src.stride();
        let dst_stride = self.stride();
        let major = cmp::min(self.major(), src.major());
        let minor = cmp::min(self.minor(), src.minor());

        for i in 0..major {
            let src_lower = i * src_stride.major();
            let src_upper = src_lower + minor * src_stride.minor();
            let dst_lower = i * dst_stride.major();
            let dst_upper = dst_lower + minor * dst_stride.minor();
            unsafe {
                let src = src.data.get_unchecked(src_lower..src_upper);
                let dst = self.data.get_unchecked_mut(dst_lower..dst_upper);
                dst.clone_from_slice(src);
            }
        }

        self
    }

    pub fn overwrite_cross_order(&mut self, src: &Matrix<T, O::Alternate>) -> &mut Self
    where
        T: Clone,
    {
        let src_stride = src.stride();
        let dst_stride = self.stride();
        let major = cmp::min(self.major(), src.minor());
        let minor = cmp::min(self.minor(), src.major());

        for i in 0..major {
            let dst_lower = i * dst_stride.major();
            let dst_upper = dst_lower + minor * dst_stride.minor();
            unsafe {
                let src = src.data.iter().skip(i).step_by(src_stride.major());
                let dst = self.data.get_unchecked_mut(dst_lower..dst_upper).iter_mut();
                dst.zip(src).for_each(|(dst, src)| *dst = src.clone());
            }
        }

        self
    }

    pub fn apply<F>(&mut self, f: F) -> &mut Self
    where
        F: FnMut(&mut T),
    {
        self.data.iter_mut().for_each(f);
        self
    }

    pub fn map<F, U>(self, f: F) -> Result<Matrix<U, O>>
    where
        F: FnMut(T) -> U,
    {
        let layout = self.layout.cast()?;
        let data = self.data.into_iter().map(f).collect();
        Ok(Matrix { layout, data })
    }

    pub fn map_ref<'a, F, U>(&'a self, f: F) -> Result<Matrix<U, O>>
    where
        F: FnMut(&'a T) -> U,
    {
        let layout = self.layout.cast()?;
        let data = self.data.iter().map(f).collect();
        Ok(Matrix { layout, data })
    }

    pub fn clear(&mut self) -> &mut Self {
        self.layout = Layout::default();
        self.data.clear();
        self
    }
}

impl<T, O> Clone for Matrix<T, O>
where
    T: Clone,
    O: Order,
{
    fn clone(&self) -> Self {
        let layout = self.layout;
        let data = self.data.clone();
        Self { layout, data }
    }
}

impl<T, O> Hash for Matrix<T, O>
where
    T: Hash,
    O: Order,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.layout.hash(state);
        self.data.hash(state);
    }
}

impl<L, R, LO, RO> PartialEq<Matrix<R, RO>> for Matrix<L, LO>
where
    L: PartialEq<R>,
    LO: Order,
    RO: Order,
{
    fn eq(&self, other: &Matrix<R, RO>) -> bool {
        if self.shape() != other.shape() {
            false
        } else if LO::KIND == RO::KIND {
            self.data == other.data
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = other.stride();
            self.data.iter().enumerate().all(|(index, left)| {
                let index =
                    Index::from_flattened::<LO>(index, lhs_stride).to_flattened::<RO>(rhs_stride);
                let right = unsafe { other.data.get_unchecked(index) };
                left == right
            })
        }
    }
}

impl<T, O> Eq for Matrix<T, O>
where
    T: PartialEq,
    O: Order,
{
}
