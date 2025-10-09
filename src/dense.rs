//! Dense matrix implementation.

use self::layout::{Layout, Order, Stride};
use crate::error::Result;
use crate::index::Index;
use crate::shape::Shape;
use alloc::vec::Vec;
use core::cmp;
use core::hash::{Hash, Hasher};
use core::ptr;

pub mod layout;

mod arithmetic;
mod construct;
mod convert;
mod eq;
mod fmt;
mod index;
mod iter;
mod resize;
mod swap;

#[cfg(feature = "parallel")]
mod parallel;

#[cfg(feature = "serde")]
mod serde;

pub struct Matrix<T, O>
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
        unsafe {
            self.data.set_len(0);
        }
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

        unsafe {
            new_data.set_len(size);
        }
        self.data = new_data;

        self
    }

    pub fn with_order<P>(mut self) -> Matrix<T, P>
    where
        P: Order,
    {
        if O::KIND != P::KIND {
            self.transpose();
        }
        self.reinterpret::<P>()
    }

    pub fn reinterpret<P>(self) -> Matrix<T, P>
    where
        P: Order,
    {
        let layout = self.layout.with_order::<P>();
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

    pub fn overwrite<P>(&mut self, src: &Matrix<T, P>) -> &mut Self
    where
        T: Clone,
        P: Order,
    {
        let src_stride = src.stride();
        let dst_stride = self.stride();

        if O::KIND == P::KIND {
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
        } else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::layout::{ColMajor, RowMajor};
    use crate::error::Error;
    use crate::{dispatch_binary, dispatch_unary, matrix};

    #[test]
    fn test_shape() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert_eq!(matrix.shape(), Shape::new(2, 3));

            let matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<O>();
            assert_eq!(matrix.shape(), Shape::new(3, 2));
        }}
    }

    #[test]
    fn test_nrows() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert_eq!(matrix.nrows(), 2);

            let matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<O>();
            assert_eq!(matrix.nrows(), 3);
        }}
    }

    #[test]
    fn test_ncols() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert_eq!(matrix.ncols(), 3);

            let matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<O>();
            assert_eq!(matrix.ncols(), 2);
        }}
    }

    #[test]
    fn test_size() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::new();
            assert_eq!(matrix.size(), 0);

            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert_eq!(matrix.size(), 6);
        }}
    }

    #[test]
    fn test_is_empty() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::new();
            assert!(matrix.is_empty());

            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert!(!matrix.is_empty());
        }}
    }

    #[test]
    fn test_capacity() {
        dispatch_unary! {{
            let matrix = Matrix::<i32, O>::with_capacity(10);
            assert!(matrix.capacity() >= 10);
        }}
    }

    #[test]
    fn test_transpose() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.transpose();
            let expected = matrix![[1, 4], [2, 5], [3, 6]];
            assert_eq!(matrix, expected);

            let mut matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<O>();
            matrix.transpose();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(matrix, expected);

            // Assert no panic from unflattening indices occurs.
            let mut matrix = matrix![[0; 0]; 2];
            matrix.transpose();
            let expected = matrix![[0; 2]; 0];
            assert_eq!(matrix, expected);

            // Assert no panic from unflattening indices occurs.
            let mut matrix = matrix![[0; 3]; 0];
            matrix.transpose();
            let expected = matrix![[0; 0]; 3];
            assert_eq!(matrix, expected);
        }}
    }

    #[test]
    fn test_with_order() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]];
            let output = matrix.with_order::<O>();
            let expected = matrix![[1, 2, 3], [4, 5, 6]];
            assert_eq!(output, expected);
        }}
    }

    #[test]
    fn test_reinterpret() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        let output = matrix.reinterpret::<RowMajor>();
        let expected = matrix![[1, 2, 3], [4, 5, 6]];
        assert_eq!(output, expected);

        let matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<RowMajor>();
        let output = matrix.reinterpret::<RowMajor>();
        let expected = matrix![[1, 4], [2, 5], [3, 6]];
        assert_eq!(output, expected);

        let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<RowMajor>();
        let output = matrix.reinterpret::<ColMajor>();
        let expected = matrix![[1, 4], [2, 5], [3, 6]];
        assert_eq!(output, expected);

        let matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<RowMajor>();
        let output = matrix.reinterpret::<ColMajor>();
        let expected = matrix![[1, 2, 3], [4, 5, 6]];
        assert_eq!(output, expected);

        let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<ColMajor>();
        let output = matrix.reinterpret::<RowMajor>();
        let expected = matrix![[1, 4], [2, 5], [3, 6]];
        assert_eq!(output, expected);

        let matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<ColMajor>();
        let output = matrix.reinterpret::<RowMajor>();
        let expected = matrix![[1, 2, 3], [4, 5, 6]];
        assert_eq!(output, expected);

        let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<ColMajor>();
        let output = matrix.reinterpret::<ColMajor>();
        let expected = matrix![[1, 2, 3], [4, 5, 6]];
        assert_eq!(output, expected);

        let matrix = matrix![[1, 4], [2, 5], [3, 6]].with_order::<ColMajor>();
        let output = matrix.reinterpret::<ColMajor>();
        let expected = matrix![[1, 4], [2, 5], [3, 6]];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_shrink_to_fit() {
        dispatch_unary! {{
            let mut matrix = Matrix::<i32, O>::with_capacity(10);
            assert!(matrix.capacity() >= 10);

            let shape = Shape::new(2, 3);
            matrix.resize(shape, 0).unwrap();
            assert!(matrix.capacity() >= 10);

            matrix.shrink_to_fit();
            assert!(matrix.capacity() >= 6);
        }}
    }

    #[test]
    fn test_shrink_to() {
        dispatch_unary! {{
            let mut matrix = Matrix::<i32, O>::with_capacity(10);
            assert!(matrix.capacity() >= 10);

            let shape = Shape::new(2, 3);
            matrix.resize(shape, 0).unwrap();
            assert!(matrix.capacity() >= 10);

            matrix.shrink_to(8);
            assert!(matrix.capacity() >= 8);

            matrix.shrink_to(4);
            assert!(matrix.capacity() >= 6);
        }}
    }

    #[test]
    fn test_contains() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            assert!(!matrix.contains(&0));
            assert!(matrix.contains(&1));
            assert!(matrix.contains(&2));
            assert!(matrix.contains(&3));
            assert!(matrix.contains(&4));
            assert!(matrix.contains(&5));
            assert!(matrix.contains(&6));
        }}
    }

    #[test]
    fn test_overwrite() {
        dispatch_binary! {{
            let mut dst = matrix![[0, 0, 0], [0, 0, 0]].with_order::<O>();
            let src = matrix![[1, 2]].with_order::<P>();
            dst.overwrite(&src);
            let expected = matrix![[1, 2, 0], [0, 0, 0]];
            assert_eq!(dst, expected);

            let mut dst = matrix![[0, 0, 0], [0, 0, 0]].with_order::<O>();
            let src = matrix![[1, 2], [3, 4]].with_order::<P>();
            dst.overwrite(&src);
            let expected = matrix![[1, 2, 0], [3, 4, 0]];
            assert_eq!(dst, expected);

            let mut dst = matrix![[0, 0, 0], [0, 0, 0]].with_order::<O>();
            let src = matrix![[1, 2], [3, 4], [5, 6]].with_order::<P>();
            dst.overwrite(&src);
            let expected = matrix![[1, 2, 0], [3, 4, 0]];
            assert_eq!(dst, expected);

            let mut dst = matrix![[0, 0, 0], [0, 0, 0]].with_order::<O>();
            let src = matrix![[1, 2, 3]].with_order::<P>();
            dst.overwrite(&src);
            let expected = matrix![[1, 2, 3], [0, 0, 0]];
            assert_eq!(dst, expected);

            let mut dst = matrix![[0, 0, 0], [0, 0, 0]].with_order::<O>();
            let src = matrix![[1, 2, 3, 4]].with_order::<P>();
            dst.overwrite(&src);
            let expected = matrix![[1, 2, 3], [0, 0, 0]];
            assert_eq!(dst, expected);
        }}
    }

    #[test]
    fn test_apply() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.apply(|element| *element += 2);
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(matrix, expected);
        }}
    }

    #[test]
    fn test_map() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = matrix.map(|element| element as f64).unwrap();
            let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
            assert_eq!(output, expected);

            let matrix = matrix![[(); usize::MAX]; 1].with_order::<O>();
            let error = matrix.map(|_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_map_ref() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = matrix.map_ref(|element| *element as f64).unwrap();
            let expected = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
            assert_eq!(output, expected);

            // Map to matrix of references.
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = matrix.map_ref(|element| element).unwrap();
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            assert_eq!(output, expected);

            let matrix = matrix![[(); usize::MAX]; 1].with_order::<O>();
            let error = matrix.map_ref(|_| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_clear() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            matrix.clear();
            assert_eq!(matrix.nrows(), 0);
            assert_eq!(matrix.ncols(), 0);
            assert!(matrix.is_empty());
            assert!(matrix.capacity() >= 6);
        }}
    }
}
