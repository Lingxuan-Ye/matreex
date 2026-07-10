use super::super::Matrix;
use super::super::order::Order;
use crate::error::{Error, Result};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::num::NonZero;
use core::ptr;
use core::ptr::NonNull;

#[derive(Debug)]
pub(super) struct IterVectorsMut<'a, T> {
    ptr: NonNull<T>,
    end_or_len: *mut T,
    axis_stride: usize,
    vector_len: usize,
    vector_stride: usize,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> IterVectorsMut<'a, T> {
    pub(super) const fn over_major_axis<O>(matrix: &'a mut Matrix<T, O>) -> Self
    where
        O: Order,
    {
        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let axis_len = matrix.major();
        let axis_stride = matrix_stride.major();
        let vector_len = matrix.minor();
        let vector_stride = matrix_stride.minor();
        unsafe { Self::new(base, axis_len, axis_stride, vector_len, vector_stride) }
    }

    pub(super) const fn over_minor_axis<O>(matrix: &'a mut Matrix<T, O>) -> Self
    where
        O: Order,
    {
        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let axis_len = matrix.minor();
        let axis_stride = matrix_stride.minor();
        let vector_len = matrix.major();
        let vector_stride = matrix_stride.major();
        unsafe { Self::new(base, axis_len, axis_stride, vector_len, vector_stride) }
    }

    const unsafe fn new(
        base: NonNull<T>,
        axis_len: usize,
        axis_stride: usize,
        vector_len: usize,
        vector_stride: usize,
    ) -> Self {
        let end_or_len = if size_of::<T>() == 0 || vector_len == 0 {
            ptr::without_provenance_mut(axis_len)
        } else if axis_len == 0 {
            ptr::null_mut()
        } else {
            let offset = unsafe { axis_len.unchecked_sub(1).unchecked_mul(axis_stride) };
            unsafe { base.add(offset).as_ptr() }
        };
        Self {
            ptr: base,
            end_or_len,
            axis_stride,
            vector_len,
            vector_stride,
            marker: PhantomData,
        }
    }
}

unsafe impl<T> Send for IterVectorsMut<'_, T> where T: Send {}
unsafe impl<T> Sync for IterVectorsMut<'_, T> where T: Sync {}

impl<'a, T> Iterator for IterVectorsMut<'a, T> {
    type Item = IterNthVectorMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            let len = self.end_or_len.addr().checked_sub(1)?;
            self.end_or_len = ptr::without_provenance_mut(len);
            let ptr = NonNull::dangling();
            let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
            return Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) });
        }

        if self.end_or_len.is_null() {
            return None;
        }
        let ptr = self.ptr;
        if self.ptr.as_ptr() == self.end_or_len {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = self.axis_stride;
            self.ptr = unsafe { self.ptr.add(offset) };
        }
        let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
        Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            let mut len = self.end_or_len.addr();
            if n >= len {
                self.end_or_len = ptr::without_provenance_mut(0);
                return None;
            }
            len = unsafe { len.unchecked_sub(n).unchecked_sub(1) };
            self.end_or_len = ptr::without_provenance_mut(len);
            let ptr = NonNull::dangling();
            let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
            return Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) });
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = unsafe { size_of::<T>().unchecked_mul(self.axis_stride) };
        let offset = n.checked_mul(stride)?;
        let addr = start.checked_add(offset)?;
        if addr > end {
            self.end_or_len = ptr::null_mut();
            return None;
        }
        let addr = unsafe { NonZero::new_unchecked(addr) };
        let ptr = self.ptr.with_addr(addr);
        if addr.get() == end {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = stride;
            self.ptr = unsafe { ptr.byte_add(offset) };
        }
        let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
        Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) })
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            let mut len = self.end_or_len.addr();
            if len == 0 {
                return init;
            }
            let mut acc = init;
            let ptr = NonNull::dangling();
            loop {
                let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
                let item = unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) };
                acc = f(acc, item);
                if len == 1 {
                    break;
                }
                len = unsafe { len.unchecked_sub(1) };
            }
            return acc;
        }

        let mut len = self.len();
        if len == 0 {
            return init;
        }
        let mut acc = init;
        let mut ptr = self.ptr;
        let offset = self.axis_stride;
        loop {
            let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
            let item = unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) };
            acc = f(acc, item);
            if len == 1 {
                break;
            }
            len = unsafe { len.unchecked_sub(1) };
            ptr = unsafe { ptr.add(offset) };
        }
        acc
    }
}

impl<T> ExactSizeIterator for IterVectorsMut<'_, T> {
    fn len(&self) -> usize {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            return self.end_or_len.addr();
        }

        if self.end_or_len.is_null() {
            return 0;
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = self.axis_stride;
        unsafe {
            (end.unchecked_sub(start) / size_of::<T>().unchecked_mul(stride)).unchecked_add(1)
        }
    }
}

impl<T> DoubleEndedIterator for IterVectorsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            let len = self.end_or_len.addr().checked_sub(1)?;
            self.end_or_len = ptr::without_provenance_mut(len);
            let ptr = NonNull::dangling();
            let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
            return Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) });
        }

        let ptr = NonNull::new(self.end_or_len)?;
        if self.ptr.as_ptr() == self.end_or_len {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = self.axis_stride;
            self.end_or_len = unsafe { self.end_or_len.sub(offset) };
        }
        let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
        Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) })
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            let mut len = self.end_or_len.addr();
            if n >= len {
                self.end_or_len = ptr::without_provenance_mut(0);
                return None;
            }
            len = unsafe { len.unchecked_sub(n).unchecked_sub(1) };
            self.end_or_len = ptr::without_provenance_mut(len);
            let ptr = NonNull::dangling();
            let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
            return Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) });
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = unsafe { size_of::<T>().unchecked_mul(self.axis_stride) };
        let offset = n.checked_mul(stride)?;
        let addr = end.checked_sub(offset)?;
        if addr < start {
            self.end_or_len = ptr::null_mut();
            return None;
        }
        let addr = unsafe { NonZero::new_unchecked(addr) };
        let ptr = self.ptr.with_addr(addr);
        if addr.get() == start {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = stride;
            self.end_or_len = unsafe { ptr.byte_sub(offset).as_ptr() };
        }
        let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
        Some(unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) })
    }

    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            let mut len = self.end_or_len.addr();
            if len == 0 {
                return init;
            }
            let mut acc = init;
            let ptr = NonNull::dangling();
            loop {
                let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
                let item = unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) };
                acc = f(acc, item);
                if len == 1 {
                    break;
                }
                len = unsafe { len.unchecked_sub(1) };
            }
            return acc;
        }

        let mut len = self.len();
        if len == 0 {
            return init;
        }
        let mut acc = init;
        let mut ptr = unsafe { NonNull::new_unchecked(self.end_or_len) };
        let offset = self.axis_stride;
        loop {
            let vector_stride = unsafe { NonZero::new_unchecked(self.vector_stride) };
            let item = unsafe { IterNthVectorMut::new(ptr, self.vector_len, vector_stride) };
            acc = f(acc, item);
            if len == 1 {
                break;
            }
            len = unsafe { len.unchecked_sub(1) };
            ptr = unsafe { ptr.sub(offset) };
        }
        acc
    }
}

impl<T> FusedIterator for IterVectorsMut<'_, T> {}

#[derive(Debug)]
pub(super) struct IterNthVectorMut<'a, T> {
    ptr: NonNull<T>,
    end_or_len: *mut T,
    stride: NonZero<usize>,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> IterNthVectorMut<'a, T> {
    /// This is an alternative to [`Matrix::iter_nth_major_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(super) const fn over_major_axis_vector<O>(
        matrix: &'a mut Matrix<T, O>,
        n: usize,
    ) -> Result<Self>
    where
        O: Order,
    {
        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let axis_len = matrix.major();
        let axis_stride = matrix_stride.major();
        let vector_len = matrix.minor();
        let vector_stride = matrix_stride.minor();

        if n >= axis_len {
            return Err(Error::IndexOutOfBounds);
        }

        let ptr = if size_of::<T>() == 0 {
            NonNull::dangling()
        } else {
            let offset = unsafe { n.unchecked_mul(axis_stride) };
            // SAFETY: When `T` is not zero-sized, `base` is dangling if and only if
            // `matrix` is empty. In this case, since `axis_len > n >= 0`, `axis_stride`
            // and the resulting `offset` must be `0`. Therefore, the dangling pointer
            // is only ever advanced by `0`, which is safe.
            unsafe { base.add(offset) }
        };
        // SAFETY: `vector_stride` is always `1` for major axis vectors.
        let vector_stride = unsafe { NonZero::new_unchecked(vector_stride) };
        // When `T` is not zero-sized, `ptr` is dangling if and only if `matrix` is
        // empty. In this case, since `axis_len > n >= 0`, `vector_len` must be `0`.
        // Therefore, the iterator is empty and the dangling pointer will never be
        // dereferenced.
        Ok(unsafe { Self::new(ptr, vector_len, vector_stride) })
    }

    /// This is an alternative to [`Matrix::iter_nth_minor_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(super) const fn over_minor_axis_vector<O>(
        matrix: &'a mut Matrix<T, O>,
        n: usize,
    ) -> Result<Self>
    where
        O: Order,
    {
        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let axis_len = matrix.minor();
        let axis_stride = matrix_stride.minor();
        let vector_len = matrix.major();
        let vector_stride = matrix_stride.major();

        if n >= axis_len {
            return Err(Error::IndexOutOfBounds);
        }

        let ptr = if size_of::<T>() == 0 || vector_len == 0 {
            NonNull::dangling()
        } else {
            let offset = unsafe { n.unchecked_mul(axis_stride) };
            // SAFETY: `base` is not dangling since `axis_len > n >= 0` and `vector_len > 0`.
            unsafe { base.add(offset) }
        };
        // SAFETY: `vector_stride` is equal to `axis_len`. Since `axis_len > n >= 0`,
        // `vector_stride` cannot be `0`.
        let vector_stride = unsafe { NonZero::new_unchecked(vector_stride) };
        // When `T` is not zero-sized, `ptr` is dangling if and only if `matrix` is
        // empty. In this case, since `axis_len > n >= 0`, `vector_len` must be `0`.
        // Therefore, the iterator is empty and the dangling pointer will never be
        // dereferenced.
        Ok(unsafe { Self::new(ptr, vector_len, vector_stride) })
    }

    const unsafe fn new(ptr: NonNull<T>, len: usize, stride: NonZero<usize>) -> Self {
        let end_or_len = if size_of::<T>() == 0 {
            ptr::without_provenance_mut(len)
        } else if len == 0 {
            ptr::null_mut()
        } else {
            let offset = unsafe { len.unchecked_sub(1).unchecked_mul(stride.get()) };
            unsafe { ptr.add(offset).as_ptr() }
        };
        Self {
            ptr,
            end_or_len,
            stride,
            marker: PhantomData,
        }
    }
}

unsafe impl<T> Send for IterNthVectorMut<'_, T> where T: Send {}
unsafe impl<T> Sync for IterNthVectorMut<'_, T> where T: Sync {}

impl<'a, T> Iterator for IterNthVectorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if size_of::<T>() == 0 {
            let len = self.end_or_len.addr().checked_sub(1)?;
            self.end_or_len = ptr::without_provenance_mut(len);
            return Some(unsafe { NonNull::dangling().as_mut() });
        }

        if self.end_or_len.is_null() {
            return None;
        }
        let mut ptr = self.ptr;
        if self.ptr.as_ptr() == self.end_or_len {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = self.stride.get();
            self.ptr = unsafe { self.ptr.add(offset) };
        }
        Some(unsafe { ptr.as_mut() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if size_of::<T>() == 0 {
            let mut len = self.end_or_len.addr();
            if n >= len {
                self.end_or_len = ptr::without_provenance_mut(0);
                return None;
            }
            len = unsafe { len.unchecked_sub(n).unchecked_sub(1) };
            self.end_or_len = ptr::without_provenance_mut(len);
            return Some(unsafe { NonNull::dangling().as_mut() });
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = unsafe { size_of::<T>().unchecked_mul(self.stride.get()) };
        let offset = n.checked_mul(stride)?;
        let addr = start.checked_add(offset)?;
        if addr > end {
            self.end_or_len = ptr::null_mut();
            return None;
        }
        let addr = unsafe { NonZero::new_unchecked(addr) };
        let mut ptr = self.ptr.with_addr(addr);
        if addr.get() == end {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = stride;
            self.ptr = unsafe { ptr.byte_add(offset) };
        }
        Some(unsafe { ptr.as_mut() })
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        if size_of::<T>() == 0 {
            let mut len = self.len();
            if len == 0 {
                return init;
            }
            let mut acc = init;
            let mut ptr = NonNull::dangling();
            loop {
                let item = unsafe { ptr.as_mut() };
                acc = f(acc, item);
                if len == 1 {
                    break;
                }
                len = unsafe { len.unchecked_sub(1) };
            }
            return acc;
        }

        let mut len = self.len();
        if len == 0 {
            return init;
        }
        let mut acc = init;
        let mut ptr = self.ptr;
        let offset = self.stride.get();
        loop {
            let item = unsafe { ptr.as_mut() };
            acc = f(acc, item);
            if len == 1 {
                break;
            }
            len = unsafe { len.unchecked_sub(1) };
            ptr = unsafe { ptr.add(offset) };
        }
        acc
    }
}

impl<T> ExactSizeIterator for IterNthVectorMut<'_, T> {
    fn len(&self) -> usize {
        if size_of::<T>() == 0 {
            return self.end_or_len.addr();
        }

        if self.end_or_len.is_null() {
            return 0;
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = self.stride.get();
        unsafe {
            (end.unchecked_sub(start) / size_of::<T>().unchecked_mul(stride)).unchecked_add(1)
        }
    }
}

impl<T> DoubleEndedIterator for IterNthVectorMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if size_of::<T>() == 0 {
            let len = self.end_or_len.addr().checked_sub(1)?;
            self.end_or_len = ptr::without_provenance_mut(len);
            return Some(unsafe { NonNull::dangling().as_mut() });
        }

        let mut ptr = NonNull::new(self.end_or_len)?;
        if self.ptr.as_ptr() == self.end_or_len {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = self.stride.get();
            self.end_or_len = unsafe { self.end_or_len.sub(offset) };
        }
        Some(unsafe { ptr.as_mut() })
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if size_of::<T>() == 0 {
            let mut len = self.end_or_len.addr();
            if n >= len {
                self.end_or_len = ptr::without_provenance_mut(0);
                return None;
            }
            len = unsafe { len.unchecked_sub(n).unchecked_sub(1) };
            self.end_or_len = ptr::without_provenance_mut(len);
            return Some(unsafe { NonNull::dangling().as_mut() });
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = unsafe { size_of::<T>().unchecked_mul(self.stride.get()) };
        let offset = n.checked_mul(stride)?;
        let addr = end.checked_sub(offset)?;
        if addr < start {
            self.end_or_len = ptr::null_mut();
            return None;
        }
        let addr = unsafe { NonZero::new_unchecked(addr) };
        let mut ptr = self.ptr.with_addr(addr);
        if addr.get() == start {
            self.end_or_len = ptr::null_mut();
        } else {
            let offset = stride;
            self.end_or_len = unsafe { ptr.byte_sub(offset).as_ptr() };
        }
        Some(unsafe { ptr.as_mut() })
    }

    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        if size_of::<T>() == 0 {
            let mut len = self.len();
            if len == 0 {
                return init;
            }
            let mut acc = init;
            let mut ptr = NonNull::dangling();
            loop {
                let item = unsafe { ptr.as_mut() };
                acc = f(acc, item);
                if len == 1 {
                    break;
                }
                len = unsafe { len.unchecked_sub(1) };
            }
            return acc;
        }

        let mut len = self.len();
        if len == 0 {
            return init;
        }
        let mut acc = init;
        let mut ptr = unsafe { NonNull::new_unchecked(self.end_or_len) };
        let offset = self.stride.get();
        loop {
            let item = unsafe { ptr.as_mut() };
            acc = f(acc, item);
            if len == 1 {
                break;
            }
            len = unsafe { len.unchecked_sub(1) };
            ptr = unsafe { ptr.sub(offset) };
        }
        acc
    }
}

impl<T> FusedIterator for IterNthVectorMut<'_, T> {}

#[cfg(test)]
mod tests {
    // In order not to bloat the test cases, only row-major matrices are tested.
    // In addition, there is a degree of abstraction leakage in this module that
    // mixes layout-level details with row/column semantics; that is, the major
    // axis is always treated as the row axis and the minor axis as the column
    // axis.

    use super::*;
    use crate::Matrix;
    use crate::index::Index;
    use crate::shape::Shape;

    #[test]
    fn test_iter_vectors_mut_next() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            assert_eq!(iter.len(), shape.nrows);

            let mut row = 0;
            while let Some(vector) = iter.next() {
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                assert_eq!(iter.len(), shape.nrows - 1 - row);
                row += 1;
            }

            assert_eq!(iter.len(), 0);

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            assert_eq!(iter.len(), shape.ncols);

            let mut col = 0;
            while let Some(vector) = iter.next() {
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                assert_eq!(iter.len(), shape.ncols - 1 - col);
                col += 1;
            }

            assert_eq!(iter.len(), 0);

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            assert_eq!(iter.len(), shape.nrows);

            let mut row = 0;
            while let Some(vector) = iter.next() {
                assert_eq!(vector.len(), shape.ncols);
                assert_eq!(iter.len(), shape.nrows - 1 - row);
                row += 1;
            }

            assert_eq!(iter.len(), 0);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            assert_eq!(iter.len(), shape.ncols);

            let mut col = 0;
            while let Some(vector) = iter.next() {
                assert_eq!(vector.len(), shape.nrows);
                assert_eq!(iter.len(), shape.ncols - 1 - col);
                col += 1;
            }

            assert_eq!(iter.len(), 0);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            assert_eq!(iter.len(), shape.nrows);

            let mut row = 0;
            while let Some(vector) = iter.next() {
                assert_eq!(vector.len(), shape.ncols);
                assert_eq!(iter.len(), shape.nrows - 1 - row);
                row += 1;
            }

            assert_eq!(iter.len(), 0);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            assert_eq!(iter.len(), shape.ncols);

            let mut col = 0;
            while let Some(vector) = iter.next() {
                assert_eq!(vector.len(), shape.nrows);
                assert_eq!(iter.len(), shape.ncols - 1 - col);
                col += 1;
            }

            assert_eq!(iter.len(), 0);
        }

        Ok(())
    }

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn test_iter_vectors_mut_nth() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            for row in 0..shape.nrows {
                let vector = iter.nth(0).unwrap();
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            assert!(iter.nth(0).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            for col in 0..shape.ncols {
                let vector = iter.nth(0).unwrap();
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            assert!(iter.nth(0).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth(row).unwrap();
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth(shape.nrows).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth(col).unwrap();
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth(shape.ncols).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth(row).unwrap();
                assert_eq!(vector.len(), shape.ncols);
            }

            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth(shape.nrows).is_none());
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth(col).unwrap();
                assert_eq!(vector.len(), shape.nrows);
            }

            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth(shape.ncols).is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth(row).unwrap();
                assert_eq!(vector.len(), shape.ncols);
            }

            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth(shape.nrows).is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth(col).unwrap();
                assert_eq!(vector.len(), shape.nrows);
            }

            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth(shape.ncols).is_none());
        }

        Ok(())
    }

    #[test]
    fn test_iter_vectors_mut_fold() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            iter.fold(0, |row, vector| {
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                row + 1
            });

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            iter.fold(0, |col, vector| {
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                col + 1
            });

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let iter_len = iter.fold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.ncols);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.nrows);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let iter_len = iter.fold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.nrows);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.ncols);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let iter_len = iter.fold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.ncols);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.nrows);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let iter_len = iter.fold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.nrows);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.ncols);
        }

        Ok(())
    }

    #[test]
    fn test_iter_vectors_mut_next_back() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            assert_eq!(iter.len(), shape.nrows);

            let mut row = shape.nrows - 1;
            while let Some(vector) = iter.next_back() {
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                assert_eq!(iter.len(), row);
                row = row.wrapping_sub(1);
            }

            assert_eq!(iter.len(), 0);

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            assert_eq!(iter.len(), shape.ncols);

            let mut col = shape.ncols - 1;
            while let Some(vector) = iter.next_back() {
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                assert_eq!(iter.len(), col);
                col = col.wrapping_sub(1);
            }

            assert_eq!(iter.len(), 0);

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            assert_eq!(iter.len(), shape.nrows);

            let mut row = shape.nrows - 1;
            while let Some(vector) = iter.next_back() {
                assert_eq!(vector.len(), shape.ncols);
                assert_eq!(iter.len(), row);
                row = row.wrapping_sub(1);
            }

            assert_eq!(iter.len(), 0);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            assert_eq!(iter.len(), shape.ncols);

            let mut col = shape.ncols - 1;
            while let Some(vector) = iter.next_back() {
                assert_eq!(vector.len(), shape.nrows);
                assert_eq!(iter.len(), col);
                col = col.wrapping_sub(1);
            }

            assert_eq!(iter.len(), 0);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            assert_eq!(iter.len(), shape.nrows);

            let mut row = shape.nrows - 1;
            while let Some(vector) = iter.next_back() {
                assert_eq!(vector.len(), shape.ncols);
                assert_eq!(iter.len(), row);
                row = row.wrapping_sub(1);
            }

            assert_eq!(iter.len(), 0);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            assert_eq!(iter.len(), shape.ncols);

            let mut col = shape.ncols - 1;
            while let Some(vector) = iter.next_back() {
                assert_eq!(vector.len(), shape.nrows);
                assert_eq!(iter.len(), col);
                col = col.wrapping_sub(1);
            }

            assert_eq!(iter.len(), 0);
        }

        Ok(())
    }

    #[test]
    fn test_iter_vectors_mut_nth_back() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);

            for row in (0..shape.nrows).rev() {
                let vector = iter.nth_back(0).unwrap();
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            assert!(iter.nth_back(0).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);

            for col in (0..shape.ncols).rev() {
                let vector = iter.nth_back(0).unwrap();
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            assert!(iter.nth_back(0).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth_back(shape.nrows - 1 - row).unwrap();
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth_back(shape.nrows).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth_back(shape.ncols - 1 - col).unwrap();
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
            }

            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth_back(shape.ncols).is_none());

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth_back(row).unwrap();
                assert_eq!(vector.len(), shape.ncols);
            }

            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth_back(shape.nrows).is_none());
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth_back(col).unwrap();
                assert_eq!(vector.len(), shape.nrows);
            }

            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth_back(shape.ncols).is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth_back(row).unwrap();
                assert_eq!(vector.len(), shape.ncols);
            }

            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth_back(shape.nrows).is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth_back(col).unwrap();
                assert_eq!(vector.len(), shape.nrows);
            }

            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth_back(shape.ncols).is_none());
        }

        Ok(())
    }

    #[test]
    fn test_iter_vectors_mut_rfold() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            iter.rfold(shape.nrows - 1, |row, vector| {
                for (col, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                row.saturating_sub(1)
            });

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            iter.rfold(shape.ncols - 1, |col, vector| {
                for (row, element) in vector.enumerate() {
                    *element = Index::new(row, col);
                }
                col.saturating_sub(1)
            });

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let iter_len = iter.rfold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.ncols);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.nrows);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let iter_len = iter.rfold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.nrows);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.ncols);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let iter_len = iter.fold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.ncols);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.nrows);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let iter_len = iter.fold(0, |iter_len, vector| {
                assert_eq!(vector.len(), shape.nrows);
                iter_len + 1
            });
            assert_eq!(iter_len, shape.ncols);
        }

        Ok(())
    }

    #[test]
    fn test_iter_nth_vector_mut_next() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;

                assert_eq!(iter.len(), shape.ncols);

                let mut col = 0;
                while let Some(element) = iter.next() {
                    *element = Index::new(row, col);
                    assert_eq!(iter.len(), shape.ncols - 1 - col);
                    col += 1;
                }

                assert_eq!(iter.len(), 0);
            }

            let error =
                IterNthVectorMut::over_major_axis_vector(&mut matrix, shape.nrows).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;

                assert_eq!(iter.len(), shape.nrows);

                let mut row = 0;
                while let Some(element) = iter.next() {
                    *element = Index::new(row, col);
                    assert_eq!(iter.len(), shape.nrows - 1 - row);
                    row += 1;
                }

                assert_eq!(iter.len(), 0);
            }

            let error =
                IterNthVectorMut::over_minor_axis_vector(&mut matrix, shape.ncols).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert_eq!(iter.len(), shape.ncols);
                assert!(iter.next().is_none());
            }

            let error =
                IterNthVectorMut::over_major_axis_vector(&mut matrix, shape.nrows).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert_eq!(iter.len(), shape.nrows);
                assert!(iter.next().is_none());
            }

            let error =
                IterNthVectorMut::over_minor_axis_vector(&mut matrix, shape.ncols).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;

                assert_eq!(iter.len(), shape.ncols);

                let mut col = 0;
                while iter.next().is_some() {
                    assert_eq!(iter.len(), shape.ncols - 1 - col);
                    col += 1;
                }

                assert_eq!(iter.len(), 0);
            }

            let error =
                IterNthVectorMut::over_major_axis_vector(&mut matrix, shape.nrows).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;

                assert_eq!(iter.len(), shape.nrows);

                let mut row = 0;
                while iter.next().is_some() {
                    assert_eq!(iter.len(), shape.nrows - 1 - row);
                    row += 1;
                }

                assert_eq!(iter.len(), 0);
            }

            let error =
                IterNthVectorMut::over_minor_axis_vector(&mut matrix, shape.ncols).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        Ok(())
    }

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn test_iter_nth_vector_mut_nth() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;

                for col in 0..shape.ncols {
                    let element = iter.nth(0).unwrap();
                    *element = Index::new(row, col);
                }

                assert!(iter.nth(0).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;

                for row in 0..shape.nrows {
                    let element = iter.nth(0).unwrap();
                    *element = Index::new(row, col);
                }

                assert!(iter.nth(0).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                for col in 0..shape.ncols {
                    let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                    let element = iter.nth(col).unwrap();
                    *element = Index::new(row, col);
                }

                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert!(iter.nth(shape.ncols).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                for row in 0..shape.nrows {
                    let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                    let element = iter.nth(row).unwrap();
                    *element = Index::new(row, col);
                }

                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert!(iter.nth(shape.nrows).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert!(iter.nth(0).is_none());
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert!(iter.nth(0).is_none());
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                for col in 0..shape.ncols {
                    let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                    assert!(iter.nth(col).is_some());
                }

                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert!(iter.nth(shape.ncols).is_none());
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                for row in 0..shape.nrows {
                    let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                    assert!(iter.nth(row).is_some());
                }

                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert!(iter.nth(shape.nrows).is_none());
            }
        }

        Ok(())
    }

    #[test]
    fn test_iter_nth_vector_mut_fold() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                iter.fold(0, |col, element| {
                    *element = Index::new(row, col);
                    col + 1
                });
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                iter.fold(0, |row, element| {
                    *element = Index::new(row, col);
                    row + 1
                });
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                let iter_len = iter.fold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.ncols);
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                let iter_len = iter.fold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.nrows);
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                let iter_len = iter.fold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.ncols);
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                let iter_len = iter.fold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.nrows);
            }
        }

        Ok(())
    }

    #[test]
    fn test_iter_nth_vector_mut_next_back() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;

                assert_eq!(iter.len(), shape.ncols);

                let mut col = shape.ncols - 1;
                while let Some(element) = iter.next_back() {
                    *element = Index::new(row, col);
                    assert_eq!(iter.len(), col);
                    col = col.wrapping_sub(1);
                }

                assert_eq!(iter.len(), 0);
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;

                assert_eq!(iter.len(), shape.nrows);

                let mut row = shape.nrows - 1;
                while let Some(element) = iter.next_back() {
                    *element = Index::new(row, col);
                    assert_eq!(iter.len(), row);
                    row = row.wrapping_sub(1);
                }

                assert_eq!(iter.len(), 0);
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert_eq!(iter.len(), shape.ncols);
                assert!(iter.next_back().is_none());
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert_eq!(iter.len(), shape.nrows);
                assert!(iter.next_back().is_none());
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;

                assert_eq!(iter.len(), shape.ncols);

                let mut col = shape.ncols - 1;
                while iter.next_back().is_some() {
                    assert_eq!(iter.len(), col);
                    col = col.wrapping_sub(1);
                }

                assert_eq!(iter.len(), 0);
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;

                assert_eq!(iter.len(), shape.nrows);

                let mut row = shape.nrows - 1;
                while iter.next_back().is_some() {
                    assert_eq!(iter.len(), row);
                    row = row.wrapping_sub(1);
                }

                assert_eq!(iter.len(), 0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_iter_nth_vector_mut_nth_back() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;

                for col in (0..shape.ncols).rev() {
                    let element = iter.nth_back(0).unwrap();
                    *element = Index::new(row, col);
                }

                assert!(iter.nth_back(0).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;

                for row in (0..shape.nrows).rev() {
                    let element = iter.nth_back(0).unwrap();
                    *element = Index::new(row, col);
                }

                assert!(iter.nth_back(0).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                for col in 0..shape.ncols {
                    let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                    let element = iter.nth_back(shape.ncols - 1 - col).unwrap();
                    *element = Index::new(row, col);
                }

                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert!(iter.nth_back(shape.ncols).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                for row in 0..shape.nrows {
                    let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                    let element = iter.nth_back(shape.nrows - 1 - row).unwrap();
                    *element = Index::new(row, col);
                }

                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert!(iter.nth_back(shape.nrows).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert!(iter.nth_back(0).is_none());
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert!(iter.nth_back(0).is_none());
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                for col in 0..shape.ncols {
                    let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                    assert!(iter.nth_back(col).is_some());
                }

                let mut iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                assert!(iter.nth_back(shape.ncols).is_none());
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                for row in 0..shape.nrows {
                    let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                    assert!(iter.nth_back(row).is_some());
                }

                let mut iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                assert!(iter.nth_back(shape.nrows).is_none());
            }
        }

        Ok(())
    }

    #[test]
    fn test_iter_nth_vector_mut_rfold() -> Result<()> {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                iter.rfold(shape.ncols - 1, |col, element| {
                    *element = Index::new(row, col);
                    col.saturating_sub(1)
                });
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;

                iter.rfold(shape.nrows - 1, |row, element| {
                    *element = Index::new(row, col);
                    row.saturating_sub(1)
                });
            }

            let expected = Matrix::from_fn(shape, |index| index)?;
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for row in 0..shape.nrows {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                let iter_len = iter.rfold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.ncols);
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<Index>::from_default(shape)?;

            for col in 0..shape.ncols {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                let iter_len = iter.rfold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.nrows);
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for row in 0..shape.nrows {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, row)?;
                let iter_len = iter.rfold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.ncols);
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape)?;

            for col in 0..shape.ncols {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, col)?;
                let iter_len = iter.rfold(0, |iter_len, _| iter_len + 1);
                assert_eq!(iter_len, shape.nrows);
            }
        }

        Ok(())
    }
}
