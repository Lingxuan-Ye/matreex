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
    lower: NonNull<T>,
    upper: NonNull<T>,
    layout: Layout,
    marker: PhantomData<&'a mut T>,
}

#[derive(Debug)]
enum Layout {
    Empty {
        iter_length: usize,
    },

    NonEmpty {
        axis_stride: NonZero<usize>,
        vector_length: NonZero<usize>,
        vector_stride: NonZero<usize>,
    },
}

unsafe impl<T> Send for IterVectorsMut<'_, T> where T: Send {}
unsafe impl<T> Sync for IterVectorsMut<'_, T> where T: Sync {}

impl<'a, T> IterVectorsMut<'a, T> {
    pub(super) fn over_major_axis<O>(matrix: &'a mut Matrix<T, O>) -> Self
    where
        O: Order,
    {
        if matrix.is_empty() {
            return Self::empty(matrix.major());
        }

        let matrix_stride = matrix.stride();

        unsafe {
            let base = NonNull::new_unchecked(matrix.data.as_mut_ptr());
            let axis_length = NonZero::new_unchecked(matrix.major());
            let axis_stride = NonZero::new_unchecked(matrix_stride.major());
            let vector_length = NonZero::new_unchecked(matrix.minor());
            let vector_stride = NonZero::new_unchecked(matrix_stride.minor());

            Self::non_empty(base, axis_length, axis_stride, vector_length, vector_stride)
        }
    }

    pub(super) fn over_minor_axis<O>(matrix: &'a mut Matrix<T, O>) -> Self
    where
        O: Order,
    {
        if matrix.is_empty() {
            return Self::empty(matrix.minor());
        }

        let matrix_stride = matrix.stride();

        unsafe {
            let base = NonNull::new_unchecked(matrix.data.as_mut_ptr());
            let axis_length = NonZero::new_unchecked(matrix.minor());
            let axis_stride = NonZero::new_unchecked(matrix_stride.minor());
            let vector_length = NonZero::new_unchecked(matrix.major());
            let vector_stride = NonZero::new_unchecked(matrix_stride.major());

            Self::non_empty(base, axis_length, axis_stride, vector_length, vector_stride)
        }
    }

    /// # Safety
    ///
    /// This returns a detached iterator whose lifetime is not tied to any
    /// matrix. However, it is safe.
    fn empty(iter_length: usize) -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            layout: Layout::Empty { iter_length },
            marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// To iterate over the vectors of a matrix, `base` must point to the
    /// underlying buffer of that matrix, and the remaining arguments must
    /// be one of the following sets of values in their [`NonZero`] form:
    ///
    /// - For iterating over the major axis:
    ///   - `axis_length`: `matrix.major()`
    ///   - `axis_stride`: `matrix.stride().major()`
    ///   - `vector_length`: `matrix.minor()`
    ///   - `vector_stride`: `matrix.stride().minor()` (i.e., `1`)
    ///
    /// - For iterating over the minor axis:
    ///   - `axis_length`: `matrix.minor()`
    ///   - `axis_stride`: `matrix.stride().minor()` (i.e., `1`)
    ///   - `vector_length`: `matrix.major()`
    ///   - `vector_stride`: `matrix.stride().major()`
    ///
    /// This returns a detached iterator whose lifetime is not tied to any
    /// matrix. The returned iterator is valid only if the matrix remains
    /// in scope.
    unsafe fn non_empty(
        base: NonNull<T>,
        axis_length: NonZero<usize>,
        axis_stride: NonZero<usize>,
        vector_length: NonZero<usize>,
        vector_stride: NonZero<usize>,
    ) -> Self {
        let lower = base;
        let offset = (axis_length.get() - 1) * axis_stride.get();
        let upper = if size_of::<T>() == 0 {
            let addr = lower.addr().get() + offset;
            let ptr = ptr::without_provenance_mut(addr);
            unsafe { NonNull::new_unchecked(ptr) }
        } else {
            unsafe { lower.add(offset) }
        };
        let layout = Layout::NonEmpty {
            axis_stride,
            vector_length,
            vector_stride,
        };

        Self {
            lower,
            upper,
            layout,
            marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for IterVectorsMut<'a, T> {
    type Item = IterNthVectorMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.layout {
            Layout::Empty { mut iter_length } => {
                if iter_length == 0 {
                    return None;
                }

                iter_length -= 1;
                self.layout = Layout::Empty { iter_length };

                Some(IterNthVectorMut::empty())
            }

            Layout::NonEmpty {
                axis_stride,
                vector_length,
                vector_stride,
            } => {
                let item = unsafe {
                    IterNthVectorMut::non_empty(self.lower, vector_length, vector_stride)
                };

                if self.lower == self.upper {
                    let iter_length = 0;
                    self.layout = Layout::Empty { iter_length };
                    return Some(item);
                }

                let stride = axis_stride.get();
                self.lower = if size_of::<T>() == 0 {
                    let addr = self.lower.addr().get() + stride;
                    let ptr = ptr::without_provenance_mut(addr);
                    unsafe { NonNull::new_unchecked(ptr) }
                } else {
                    unsafe { self.lower.add(stride) }
                };

                Some(item)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterVectorsMut<'_, T> {
    fn len(&self) -> usize {
        match self.layout {
            Layout::Empty { iter_length } => iter_length,
            Layout::NonEmpty { axis_stride, .. } => {
                let upper = self.upper.addr().get();
                let lower = self.lower.addr().get();
                let element_size = size_of::<T>();
                let stride = axis_stride.get();
                1 + (upper - lower) / (if element_size == 0 { 1 } else { element_size } * stride)
            }
        }
    }
}

impl<T> DoubleEndedIterator for IterVectorsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.layout {
            Layout::Empty { mut iter_length } => {
                if iter_length == 0 {
                    return None;
                }

                iter_length -= 1;
                self.layout = Layout::Empty { iter_length };

                Some(IterNthVectorMut::empty())
            }

            Layout::NonEmpty {
                axis_stride,
                vector_length,
                vector_stride,
            } => {
                let item = unsafe {
                    IterNthVectorMut::non_empty(self.upper, vector_length, vector_stride)
                };

                if self.lower == self.upper {
                    let iter_length = 0;
                    self.layout = Layout::Empty { iter_length };
                    return Some(item);
                }

                let stride = axis_stride.get();
                self.upper = if size_of::<T>() == 0 {
                    let addr = self.upper.addr().get() - stride;
                    let ptr = ptr::without_provenance_mut(addr);
                    unsafe { NonNull::new_unchecked(ptr) }
                } else {
                    unsafe { self.upper.sub(stride) }
                };

                Some(item)
            }
        }
    }
}

impl<T> FusedIterator for IterVectorsMut<'_, T> {}

#[derive(Debug)]
pub(super) struct IterNthVectorMut<'a, T> {
    lower: NonNull<T>,
    upper: NonNull<T>,
    stride: Option<NonZero<usize>>,
    marker: PhantomData<&'a mut T>,
}

unsafe impl<T> Send for IterNthVectorMut<'_, T> where T: Send {}
unsafe impl<T> Sync for IterNthVectorMut<'_, T> where T: Sync {}

impl<'a, T> IterNthVectorMut<'a, T> {
    /// This is an alternative to [`Matrix::iter_nth_major_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(super) fn over_major_axis_vector<O>(matrix: &'a mut Matrix<T, O>, n: usize) -> Result<Self>
    where
        O: Order,
    {
        if n >= matrix.major() {
            return Err(Error::IndexOutOfBounds);
        }

        if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let lower = if size_of::<T>() == 0 {
            base
        } else {
            let offset = n * matrix_stride.major();
            unsafe { base.add(offset) }
        };
        let length = unsafe { NonZero::new_unchecked(matrix.minor()) };
        let stride = unsafe { NonZero::new_unchecked(matrix_stride.minor()) };

        unsafe { Ok(Self::non_empty(lower, length, stride)) }
    }

    /// This is an alternative to [`Matrix::iter_nth_minor_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(super) fn over_minor_axis_vector<O>(matrix: &'a mut Matrix<T, O>, n: usize) -> Result<Self>
    where
        O: Order,
    {
        if n >= matrix.minor() {
            return Err(Error::IndexOutOfBounds);
        }

        if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let lower = if size_of::<T>() == 0 {
            base
        } else {
            let offset = n * matrix_stride.minor();
            unsafe { base.add(offset) }
        };
        let length = unsafe { NonZero::new_unchecked(matrix.major()) };
        let stride = unsafe { NonZero::new_unchecked(matrix_stride.major()) };

        unsafe { Ok(Self::non_empty(lower, length, stride)) }
    }

    /// # Safety
    ///
    /// This returns a detached iterator whose lifetime is not tied to any
    /// matrix. However, it is safe.
    fn empty() -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            stride: None,
            marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// To iterate over the nth vector of a matrix, `lower` must point to
    /// the start of that vector, and the remaining arguments must be one
    /// of the following sets of values in their [`NonZero`] form:
    ///
    /// - For iterating over a major axis vector:
    ///   - `length`: `matrix.minor()`
    ///   - `stride`: `matrix.stride().minor()` (i.e., `1`)
    ///
    /// - For iterating over a minor axis vector:
    ///   - `length`: `matrix.major()`
    ///   - `stride`: `matrix.stride().major()`
    ///
    /// This returns a detached iterator whose lifetime is not tied to any
    /// matrix. The returned iterator is valid only if the matrix remains
    /// in scope.
    unsafe fn non_empty(lower: NonNull<T>, length: NonZero<usize>, stride: NonZero<usize>) -> Self {
        let offset = (length.get() - 1) * stride.get();
        let upper = if size_of::<T>() == 0 {
            let addr = lower.addr().get() + offset;
            let ptr = ptr::without_provenance_mut(addr);
            unsafe { NonNull::new_unchecked(ptr) }
        } else {
            unsafe { lower.add(offset) }
        };

        Self {
            lower,
            upper,
            stride: Some(stride),
            marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for IterNthVectorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let stride = self.stride?.get();

        let item = if size_of::<T>() == 0 {
            // `self.lower` may not be properly aligned.
            unsafe { NonNull::dangling().as_mut() }
        } else {
            unsafe { self.lower.as_mut() }
        };

        if self.lower == self.upper {
            self.stride = None;
            return Some(item);
        }

        self.lower = if size_of::<T>() == 0 {
            let addr = self.lower.addr().get() + stride;
            let ptr = ptr::without_provenance_mut(addr);
            unsafe { NonNull::new_unchecked(ptr) }
        } else {
            unsafe { self.lower.add(stride) }
        };

        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterNthVectorMut<'_, T> {
    fn len(&self) -> usize {
        match self.stride {
            None => 0,
            Some(stride) => {
                let upper = self.upper.addr().get();
                let lower = self.lower.addr().get();
                let element_size = size_of::<T>();
                let stride = stride.get();
                1 + (upper - lower) / (if element_size == 0 { 1 } else { element_size } * stride)
            }
        }
    }
}

impl<T> DoubleEndedIterator for IterNthVectorMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let stride = self.stride?.get();

        let item = if size_of::<T>() == 0 {
            // `self.upper` may not be properly aligned.
            unsafe { NonNull::dangling().as_mut() }
        } else {
            unsafe { self.upper.as_mut() }
        };

        if self.lower == self.upper {
            self.stride = None;
            return Some(item);
        }

        self.upper = if size_of::<T>() == 0 {
            let addr = self.upper.addr().get() - stride;
            let ptr = ptr::without_provenance_mut(addr);
            unsafe { NonNull::new_unchecked(ptr) }
        } else {
            unsafe { self.upper.sub(stride) }
        };

        Some(item)
    }
}

impl<T> FusedIterator for IterNthVectorMut<'_, T> {}
