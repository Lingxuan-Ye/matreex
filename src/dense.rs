pub use self::layout::{ColMajor, RowMajor};

use self::layout::{Layout, LayoutIndex, Order, Stride};
use crate::error::Result;
use crate::shape::Shape;
use alloc::vec::Vec;
use core::cmp;
use core::ptr;

#[cfg(feature = "parallel")]
pub mod parallel;

mod construct;
mod fmt;
mod iter;
mod layout;
mod resize;

#[derive(Clone, Hash, PartialEq, Eq)]
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

        for old_index in 0..size {
            unsafe {
                let src = old_base.add(old_index);
                let new_index = LayoutIndex::from_flattened(old_index, old_stride)
                    .swap()
                    .to_flattened(new_stride);
                let dst = new_base.add(new_index);
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

    pub fn map<U, F>(self, f: F) -> Result<Matrix<U, O>>
    where
        F: FnMut(T) -> U,
    {
        let layout = self.layout.cast()?;
        let data = self.data.into_iter().map(f).collect();
        Ok(Matrix { layout, data })
    }

    pub fn map_ref<'a, U, F>(&'a self, f: F) -> Result<Matrix<U, O>>
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
