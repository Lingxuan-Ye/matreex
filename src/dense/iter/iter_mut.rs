use super::super::Matrix;
use super::super::layout::Order;
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
    pub(super) fn over_major_axis<O>(matrix: &'a mut Matrix<T, O>) -> Self
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

    pub(super) fn over_minor_axis<O>(matrix: &'a mut Matrix<T, O>) -> Self
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

    unsafe fn new(
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
            unsafe { base.as_ptr().add(offset) }
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
            let mut len = self.end_or_len.addr();
            if len == 0 {
                return None;
            }
            len = unsafe { len.unchecked_sub(1) };
            self.end_or_len = ptr::without_provenance_mut(len);
            let ptr = NonNull::dangling();
            // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
            // the upper bound of `len`, which cannot be `0` here.
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
        // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
        // `0` only if `self.end_or_len` is null, which is unreachable here.
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
            // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
            // the upper bound of `len`, which cannot be `0` here.
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
        // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
        // `0` only if `self.end_or_len` is null, which is unreachable here.
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
                // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
                // `0` only if `self.end_or_len` is null, which is unreachable here.
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
            // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
            // `0` only if `self.end_or_len` is null, which is unreachable here.
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
            let len = self.end_or_len.addr();
            return len;
        }

        if self.end_or_len.is_null() {
            return 0;
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = self.axis_stride;
        unsafe {
            ((end.unchecked_sub(start)) / (size_of::<T>().unchecked_mul(stride))).unchecked_add(1)
        }
    }
}

impl<T> DoubleEndedIterator for IterVectorsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if size_of::<T>() == 0 || self.vector_len == 0 {
            let mut len = self.end_or_len.addr();
            if len == 0 {
                return None;
            }
            len = unsafe { len.unchecked_sub(1) };
            self.end_or_len = ptr::without_provenance_mut(len);
            let ptr = NonNull::dangling();
            // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
            // the upper bound of `len`, which cannot be `0` here.
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
        // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
        // `0` only if `self.end_or_len` is null, which is unreachable here.
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
            // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
            // the upper bound of `len`, which cannot be `0` here.
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
            self.ptr = unsafe { ptr.byte_sub(offset) };
        }
        // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
        // `0` only if `self.end_or_len` is null, which is unreachable here.
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
                // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
                // `0` only if `self.end_or_len` is null, which is unreachable here.
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
        // SAFETY: `self.end_or_len` is not null since `len > 0`.
        let mut ptr = unsafe { NonNull::new_unchecked(self.end_or_len) };
        let offset = self.axis_stride;
        loop {
            // SAFETY: `self.vector_stride` is either `axis_len` or `1`, and `axis_len` is
            // `0` only if `self.end_or_len` is null, which is unreachable here.
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
    pub(super) fn over_major_axis_vector<O>(matrix: &'a mut Matrix<T, O>, n: usize) -> Result<Self>
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
        // When `T` is not zero-sized, `base` is dangling if and only if `matrix` is
        // empty. In this case, since `axis_len > n >= 0`, `vector_len` must be `0`.
        // Therefore, the iterator is empty and the dangling pointer will never be
        // dereferenced.
        Ok(unsafe { Self::new(ptr, vector_len, vector_stride) })
    }

    /// This is an alternative to [`Matrix::iter_nth_minor_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(super) fn over_minor_axis_vector<O>(matrix: &'a mut Matrix<T, O>, n: usize) -> Result<Self>
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

        if vector_len == 0 {
            let ptr = NonNull::dangling();
            // SAFETY: `vector_stride` is equal to `axis_len`. Since `axis_len > n >= 0`,
            // `vector_stride` cannot be `0`.
            let vector_stride = unsafe { NonZero::new_unchecked(vector_stride) };
            // Safety: The iterator is empty and the dangling pointer will never be
            // dereferenced.
            return Ok(unsafe { Self::new(ptr, vector_len, vector_stride) });
        }

        let ptr = if size_of::<T>() == 0 {
            NonNull::dangling()
        } else {
            let offset = unsafe { n.unchecked_mul(axis_stride) };
            // SAFETY: `base` is not dangling since `axis_len > n >= 0` and `vector_len > 0`.
            unsafe { base.add(offset) }
        };
        // SAFETY: `vector_stride` is equal to `axis_len`. Since `axis_len > n >= 0`,
        // `vector_stride` cannot be `0`.
        let vector_stride = unsafe { NonZero::new_unchecked(vector_stride) };
        Ok(unsafe { Self::new(ptr, vector_len, vector_stride) })
    }

    unsafe fn new(ptr: NonNull<T>, len: usize, stride: NonZero<usize>) -> Self {
        let end_or_len = if size_of::<T>() == 0 {
            ptr::without_provenance_mut(len)
        } else if len == 0 {
            ptr::null_mut()
        } else {
            let offset = unsafe { len.unchecked_sub(1).unchecked_mul(stride.get()) };
            unsafe { ptr.as_ptr().add(offset) }
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
            let mut len = self.end_or_len.addr();
            if len == 0 {
                return None;
            }
            len = unsafe { len.unchecked_sub(1) };
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
            let len = self.end_or_len.addr();
            return len;
        }

        if self.end_or_len.is_null() {
            return 0;
        }

        let start = self.ptr.addr().get();
        let end = self.end_or_len.addr();
        let stride = self.stride.get();
        unsafe {
            ((end.unchecked_sub(start)) / (size_of::<T>().unchecked_mul(stride))).unchecked_add(1)
        }
    }
}

impl<T> DoubleEndedIterator for IterNthVectorMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if size_of::<T>() == 0 {
            let mut len = self.end_or_len.addr();
            if len == 0 {
                return None;
            }
            len = unsafe { len.unchecked_sub(1) };
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
            self.ptr = unsafe { ptr.byte_sub(offset) };
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
        // SAFETY: `self.end_or_len` is not null since `len > 0`.
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
    use super::*;
    use crate::Matrix;
    use crate::index::Index;
    use crate::shape::Shape;

    // In order not to bloat the test cases, only row-major matrices are tested.
    // In addition, there is a degree of abstraction leakage in this module that
    // mixes layout-level details with row/column semantics; that is, the major
    // axis is always treated as the row axis and the minor axis as the column
    // axis.

    #[test]
    fn test_iter_vectors_mut_next() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert_eq!(iter.len(), 2);

            let vector_0 = iter.next().unwrap();
            for (minor, element) in vector_0.enumerate() {
                *element = Index::new(0, minor);
            }
            assert_eq!(iter.len(), 1);

            let vector_1 = iter.next().unwrap();
            for (minor, element) in vector_1.enumerate() {
                *element = Index::new(1, minor);
            }
            assert_eq!(iter.len(), 0);

            assert!(iter.next().is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert_eq!(iter.len(), 3);

            let vector_0 = iter.next().unwrap();
            for (major, element) in vector_0.enumerate() {
                *element = Index::new(major, 0);
            }
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next().unwrap();
            for (major, element) in vector_1.enumerate() {
                *element = Index::new(major, 1);
            }
            assert_eq!(iter.len(), 1);

            let vector_2 = iter.next().unwrap();
            for (major, element) in vector_2.enumerate() {
                *element = Index::new(major, 2);
            }
            assert_eq!(iter.len(), 0);

            assert!(iter.next().is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert_eq!(iter.len(), 2);

            let vector_0 = iter.next().unwrap();
            let mut minor = 0;
            for _ in vector_0 {
                minor += 1;
            }
            assert_eq!(minor, 3);
            assert_eq!(iter.len(), 1);

            let vector_1 = iter.next().unwrap();
            let mut minor = 0;
            for _ in vector_1 {
                minor += 1;
            }
            assert_eq!(minor, 3);
            assert_eq!(iter.len(), 0);

            assert!(iter.next().is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert_eq!(iter.len(), 3);

            let vector_0 = iter.next().unwrap();
            let mut major = 0;
            for _ in vector_0 {
                major += 1;
            }
            assert_eq!(major, 2);
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next().unwrap();
            let mut major = 0;
            for _ in vector_1 {
                major += 1;
            }
            assert_eq!(major, 2);
            assert_eq!(iter.len(), 1);

            let vector_2 = iter.next().unwrap();
            let mut major = 0;
            for _ in vector_2 {
                major += 1;
            }
            assert_eq!(major, 2);
            assert_eq!(iter.len(), 0);

            assert!(iter.next().is_none());
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert_eq!(iter.len(), 2);

            let vector_0 = iter.next().unwrap();
            let mut minor = 0;
            for _ in vector_0 {
                minor += 1;
            }
            assert_eq!(minor, 0);
            assert_eq!(iter.len(), 1);

            let vector_1 = iter.next().unwrap();
            let mut minor = 0;
            for _ in vector_1 {
                minor += 1;
            }
            assert_eq!(minor, 0);
            assert_eq!(iter.len(), 0);

            assert!(iter.next().is_none());
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert_eq!(iter.len(), 3);

            let vector_0 = iter.next().unwrap();
            let mut major = 0;
            for _ in vector_0 {
                major += 1;
            }
            assert_eq!(major, 0);
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next().unwrap();
            let mut major = 0;
            for _ in vector_1 {
                major += 1;
            }
            assert_eq!(major, 0);
            assert_eq!(iter.len(), 1);

            let vector_2 = iter.next().unwrap();
            let mut major = 0;
            for _ in vector_2 {
                major += 1;
            }
            assert_eq!(major, 0);
            assert_eq!(iter.len(), 0);

            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn test_iter_vectors_mut_nth() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth(major).unwrap();
                for (minor, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
            }

            let major = matrix.major();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth(major).is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth(minor).unwrap();
                for (major, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
            }

            let minor = matrix.minor();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth(minor).is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth(major).unwrap();
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 3);
            }

            let major = matrix.major();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth(major).is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth(minor).unwrap();
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 2);
            }

            let minor = matrix.minor();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth(minor).is_none());
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth(major).unwrap();
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 0);
            }

            let major = matrix.major();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth(major).is_none());
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth(minor).unwrap();
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 0);
            }

            let minor = matrix.minor();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth(minor).is_none());
        }
    }

    #[test]
    fn test_iter_vectors_mut_fold() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let mut major = 0;
            iter.fold((), |_, vector| {
                for (minor, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
                major += 1;
            });

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let mut minor = 0;
            iter.fold((), |_, vector| {
                for (major, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
                minor += 1;
            });

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let mut major = 0;
            iter.fold((), |_, vector| {
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 3);
                major += 1;
            });
            assert_eq!(major, 2);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let mut minor = 0;
            iter.fold((), |_, vector| {
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 2);
                minor += 1;
            });
            assert_eq!(minor, 3);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let mut major = 0;
            iter.fold((), |_, vector| {
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 0);
                major += 1;
            });
            assert_eq!(major, 2);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let mut minor = 0;
            iter.fold((), |_, vector| {
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 0);
                minor += 1;
            });
            assert_eq!(minor, 3);
        }
    }

    #[test]
    fn test_iter_vectors_mut_next_back() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next_back().unwrap();
            for (minor, element) in vector_1.enumerate() {
                *element = Index::new(1, minor);
            }
            assert_eq!(iter.len(), 1);

            let vector_0 = iter.next_back().unwrap();
            for (minor, element) in vector_0.enumerate() {
                *element = Index::new(0, minor);
            }
            assert_eq!(iter.len(), 0);

            assert!(iter.next_back().is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert_eq!(iter.len(), 3);

            let vector_2 = iter.next_back().unwrap();
            for (major, element) in vector_2.enumerate() {
                *element = Index::new(major, 2);
            }
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next_back().unwrap();
            for (major, element) in vector_1.enumerate() {
                *element = Index::new(major, 1);
            }
            assert_eq!(iter.len(), 1);

            let vector_0 = iter.next_back().unwrap();
            for (major, element) in vector_0.enumerate() {
                *element = Index::new(major, 0);
            }
            assert_eq!(iter.len(), 0);

            assert!(iter.next_back().is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next_back().unwrap();
            let mut minor = 0;
            for _ in vector_1 {
                minor += 1;
            }
            assert_eq!(minor, 3);
            assert_eq!(iter.len(), 1);

            let vector_1 = iter.next_back().unwrap();
            let mut minor = 0;
            for _ in vector_1 {
                minor += 1;
            }
            assert_eq!(minor, 3);
            assert_eq!(iter.len(), 0);

            assert!(iter.next_back().is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert_eq!(iter.len(), 3);

            let vector_2 = iter.next_back().unwrap();
            let mut major = 0;
            for _ in vector_2 {
                major += 1;
            }
            assert_eq!(major, 2);
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next_back().unwrap();
            let mut major = 0;
            for _ in vector_1 {
                major += 1;
            }
            assert_eq!(major, 2);
            assert_eq!(iter.len(), 1);

            let vector_0 = iter.next_back().unwrap();
            let mut major = 0;
            for _ in vector_0 {
                major += 1;
            }
            assert_eq!(major, 2);
            assert_eq!(iter.len(), 0);

            assert!(iter.next_back().is_none());
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next_back().unwrap();
            let mut minor = 0;
            for _ in vector_1 {
                minor += 1;
            }
            assert_eq!(minor, 0);
            assert_eq!(iter.len(), 1);

            let vector_0 = iter.next_back().unwrap();
            let mut minor = 0;
            for _ in vector_0 {
                minor += 1;
            }
            assert_eq!(minor, 0);
            assert_eq!(iter.len(), 0);

            assert!(iter.next_back().is_none());
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert_eq!(iter.len(), 3);

            let vector_2 = iter.next_back().unwrap();
            let mut major = 0;
            for _ in vector_2 {
                major += 1;
            }
            assert_eq!(major, 0);
            assert_eq!(iter.len(), 2);

            let vector_1 = iter.next_back().unwrap();
            let mut major = 0;
            for _ in vector_1 {
                major += 1;
            }
            assert_eq!(major, 0);
            assert_eq!(iter.len(), 1);

            let vector_0 = iter.next_back().unwrap();
            let mut major = 0;
            for _ in vector_0 {
                major += 1;
            }
            assert_eq!(major, 0);
            assert_eq!(iter.len(), 0);

            assert!(iter.next_back().is_none());
        }
    }

    #[test]
    fn test_iter_vectors_mut_nth_back() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;

            for major in 0..layout.major() {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth_back(layout.major() - 1 - major).unwrap();
                for (minor, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
            }

            let major = matrix.major();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth_back(major).is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;

            for minor in 0..layout.minor() {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth_back(layout.minor() - 1 - minor).unwrap();
                for (major, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
            }

            let minor = layout.minor();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth_back(minor).is_none());

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth_back(major).unwrap();
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 3);
            }

            let major = matrix.major();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth_back(major).is_none());
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth_back(minor).unwrap();
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 2);
            }

            let minor = matrix.minor();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth_back(minor).is_none());
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
                let vector = iter.nth_back(major).unwrap();
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 0);
            }

            let major = matrix.major();
            let mut iter = IterVectorsMut::over_major_axis(&mut matrix);
            assert!(iter.nth_back(major).is_none());
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
                let vector = iter.nth_back(minor).unwrap();
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 0);
            }

            let minor = matrix.minor();
            let mut iter = IterVectorsMut::over_minor_axis(&mut matrix);
            assert!(iter.nth_back(minor).is_none());
        }
    }

    #[test]
    fn test_iter_vectors_mut_rfold() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let mut major = layout.major() - 1;
            iter.rfold((), |_, vector| {
                for (minor, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
                major = major.saturating_sub(1);
            });

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let mut minor = layout.minor() - 1;
            iter.rfold((), |_, vector| {
                for (major, element) in vector.enumerate() {
                    *element = Index::new(major, minor);
                }
                minor = minor.saturating_sub(1);
            });

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let mut major = 0;
            iter.rfold((), |_, vector| {
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 3);
                major += 1;
            });
            assert_eq!(major, 2);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let mut minor = 0;
            iter.rfold((), |_, vector| {
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 2);
                minor += 1;
            });
            assert_eq!(minor, 3);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_major_axis(&mut matrix);

            let mut major = 0;
            iter.rfold((), |_, vector| {
                let mut minor = 0;
                for _ in vector {
                    minor += 1;
                }
                assert_eq!(minor, 0);
                major += 1;
            });
            assert_eq!(major, 2);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();
            let iter = IterVectorsMut::over_minor_axis(&mut matrix);

            let mut minor = 0;
            iter.rfold((), |_, vector| {
                let mut major = 0;
                for _ in vector {
                    major += 1;
                }
                assert_eq!(major, 0);
                minor += 1;
            });
            assert_eq!(minor, 3);
        }
    }

    #[test]
    fn test_iter_nth_vector_mut_next() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert_eq!(iter.len(), 3);

                let element_0 = iter.next().unwrap();
                *element_0 = Index::new(major, 0);
                assert_eq!(iter.len(), 2);

                let element_1 = iter.next().unwrap();
                *element_1 = Index::new(major, 1);
                assert_eq!(iter.len(), 1);

                let element_2 = iter.next().unwrap();
                *element_2 = Index::new(major, 2);
                assert_eq!(iter.len(), 0);

                assert!(iter.next().is_none());
            }

            let major = matrix.major();
            let error = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert_eq!(iter.len(), 2);

                let element_0 = iter.next().unwrap();
                *element_0 = Index::new(0, minor);
                assert_eq!(iter.len(), 1);

                let element_1 = iter.next().unwrap();
                *element_1 = Index::new(1, minor);
                assert_eq!(iter.len(), 0);

                assert!(iter.next().is_none());
            }

            let minor = matrix.minor();
            let error = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert_eq!(iter.len(), 3);

                assert!(iter.next().is_some());
                assert_eq!(iter.len(), 2);

                assert!(iter.next().is_some());
                assert_eq!(iter.len(), 1);

                assert!(iter.next().is_some());
                assert_eq!(iter.len(), 0);

                assert!(iter.next().is_none());
            }

            let major = matrix.major();
            let error = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert_eq!(iter.len(), 2);

                assert!(iter.next().is_some());
                assert_eq!(iter.len(), 1);

                assert!(iter.next().is_some());
                assert_eq!(iter.len(), 0);

                assert!(iter.next().is_none());
            }

            let minor = matrix.minor();
            let error = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert_eq!(iter.len(), 0);

                assert!(iter.next().is_none());
            }

            let major = matrix.major();
            let error = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert_eq!(iter.len(), 0);

                assert!(iter.next().is_none());
            }

            let minor = matrix.minor();
            let error = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }
    }

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn test_iter_nth_vector_mut_nth() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                for minor in 0..matrix.minor() {
                    let mut iter =
                        IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                    let element = iter.nth(minor).unwrap();
                    *element = Index::new(major, minor);
                }

                let minor = matrix.minor();
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert!(iter.nth(minor).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                for major in 0..matrix.major() {
                    let mut iter =
                        IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                    let element = iter.nth(major).unwrap();
                    *element = Index::new(major, minor);
                }

                let major = matrix.major();
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert!(iter.nth(major).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                for minor in 0..matrix.minor() {
                    let mut iter =
                        IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                    assert!(iter.nth(minor).is_some());
                }

                let minor = matrix.minor();
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert!(iter.nth(minor).is_none());
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                for major in 0..matrix.major() {
                    let mut iter =
                        IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                    assert!(iter.nth(major).is_some());
                }

                let major = matrix.major();
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert!(iter.nth(major).is_none());
            }
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert!(iter.nth(0).is_none());
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert!(iter.nth(0).is_none());
            }
        }
    }

    #[test]
    fn test_iter_nth_vector_mut_fold() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();

                let mut minor = 0;
                iter.fold((), |_, element| {
                    *element = Index::new(major, minor);
                    minor += 1;
                });
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();

                let mut major = 0;
                iter.fold((), |_, element| {
                    *element = Index::new(major, minor);
                    major += 1;
                });
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();

                let mut minor = 0;
                iter.fold((), |_, _| {
                    minor += 1;
                });
                assert_eq!(minor, 3);
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();

                let mut major = 0;
                iter.fold((), |_, _| {
                    major += 1;
                });
                assert_eq!(major, 2);
            }
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();

                let mut minor = 0;
                iter.fold((), |_, _| {
                    minor += 1;
                });
                assert_eq!(minor, 0);
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();

                let mut major = 0;
                iter.fold((), |_, _| {
                    major += 1;
                });
                assert_eq!(major, 0);
            }
        }
    }

    #[test]
    fn test_iter_nth_vector_mut_next_back() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert_eq!(iter.len(), 3);

                let element_0 = iter.next_back().unwrap();
                *element_0 = Index::new(major, 2);
                assert_eq!(iter.len(), 2);

                let element_1 = iter.next_back().unwrap();
                *element_1 = Index::new(major, 1);
                assert_eq!(iter.len(), 1);

                let element_2 = iter.next_back().unwrap();
                *element_2 = Index::new(major, 0);
                assert_eq!(iter.len(), 0);

                assert!(iter.next_back().is_none());
            }

            let major = matrix.major();
            let error = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert_eq!(iter.len(), 2);

                let element_0 = iter.next_back().unwrap();
                *element_0 = Index::new(1, minor);
                assert_eq!(iter.len(), 1);

                let element_1 = iter.next_back().unwrap();
                *element_1 = Index::new(0, minor);
                assert_eq!(iter.len(), 0);

                assert!(iter.next_back().is_none());
            }

            let minor = matrix.minor();
            let error = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert_eq!(iter.len(), 3);

                assert!(iter.next_back().is_some());
                assert_eq!(iter.len(), 2);

                assert!(iter.next_back().is_some());
                assert_eq!(iter.len(), 1);

                assert!(iter.next_back().is_some());
                assert_eq!(iter.len(), 0);

                assert!(iter.next_back().is_none());
            }

            let major = matrix.major();
            let error = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert_eq!(iter.len(), 2);

                assert!(iter.next_back().is_some());
                assert_eq!(iter.len(), 1);

                assert!(iter.next_back().is_some());
                assert_eq!(iter.len(), 0);

                assert!(iter.next_back().is_none());
            }

            let minor = matrix.minor();
            let error = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert_eq!(iter.len(), 0);

                assert!(iter.next_back().is_none());
            }

            let major = matrix.major();
            let error = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert_eq!(iter.len(), 0);

                assert!(iter.next_back().is_none());
            }

            let minor = matrix.minor();
            let error = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap_err();
            assert_eq!(error, Error::IndexOutOfBounds);
        }
    }

    #[test]
    fn test_iter_nth_vector_mut_nth_back() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;

            for major in 0..layout.major() {
                for minor in 0..layout.minor() {
                    let mut iter =
                        IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                    let element = iter.nth_back(layout.minor() - 1 - minor).unwrap();
                    *element = Index::new(major, minor);
                }

                let minor = layout.minor();
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert!(iter.nth_back(minor).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;

            for minor in 0..layout.minor() {
                for major in 0..layout.major() {
                    let mut iter =
                        IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                    let element = iter.nth_back(layout.major() - 1 - major).unwrap();
                    *element = Index::new(major, minor);
                }

                let major = layout.major();
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert!(iter.nth_back(major).is_none());
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                for minor in 0..matrix.minor() {
                    let mut iter =
                        IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                    assert!(iter.nth_back(minor).is_some());
                }

                let minor = matrix.minor();
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert!(iter.nth_back(minor).is_none());
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                for major in 0..matrix.major() {
                    let mut iter =
                        IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                    assert!(iter.nth_back(major).is_some());
                }

                let major = matrix.major();
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert!(iter.nth_back(major).is_none());
            }
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let mut iter =
                    IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();
                assert!(iter.nth_back(0).is_none());
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let mut iter =
                    IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();
                assert!(iter.nth_back(0).is_none());
            }
        }
    }

    #[test]
    fn test_iter_nth_vector_mut_rfold() {
        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;

            for major in 0..layout.major() {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();

                let mut minor = layout.minor() - 1;
                iter.rfold((), |_, element| {
                    *element = Index::new(major, minor);
                    minor = minor.saturating_sub(1);
                });
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::from_default(shape).unwrap();
            let layout = matrix.layout;

            for minor in 0..layout.minor() {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();

                let mut major = layout.major() - 1;
                iter.rfold((), |_, element| {
                    *element = Index::new(major, minor);
                    major = major.saturating_sub(1);
                });
            }

            let expected = Matrix::from_fn(shape, |index| index).unwrap();
            assert_eq!(matrix, expected);
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();

                let mut minor = 0;
                iter.rfold((), |_, _| {
                    minor += 1;
                });
                assert_eq!(minor, 3);
            }
        }

        {
            let shape = Shape::new(2, 3);
            let mut matrix = Matrix::<()>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();

                let mut major = 0;
                iter.rfold((), |_, _| {
                    major += 1;
                });
                assert_eq!(major, 2);
            }
        }

        {
            let shape = Shape::new(2, 0);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for major in 0..matrix.major() {
                let iter = IterNthVectorMut::over_major_axis_vector(&mut matrix, major).unwrap();

                let mut minor = 0;
                iter.rfold((), |_, _| {
                    minor += 1;
                });
                assert_eq!(minor, 0);
            }
        }

        {
            let shape = Shape::new(0, 3);
            let mut matrix = Matrix::<i32>::from_default(shape).unwrap();

            for minor in 0..matrix.minor() {
                let iter = IterNthVectorMut::over_minor_axis_vector(&mut matrix, minor).unwrap();

                let mut major = 0;
                iter.rfold((), |_, _| {
                    major += 1;
                });
                assert_eq!(major, 0);
            }
        }
    }
}
