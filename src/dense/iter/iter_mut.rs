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
