use crate::Matrix;
use crate::error::{Error, Result};
use core::marker::PhantomData;
use core::num::NonZero;
use core::ptr::{NonNull, without_provenance_mut};

/// # Design Details
///
/// To maintain consistency with [`IterNthVectorMut`], `upper`
/// must point **to** (not one vector past) the exact vector that
/// [`DoubleEndedIterator::next_back`] would return. In this case,
/// comparing pointers to determine whether an iterator is empty
/// is unsound, because pointers may offset outside the allocated
/// object in the final iteration. Therefore, an additional state
/// is needed to track whether the iterator is empty. This state
/// can be stored in either `lower`, `upper`, or `layout` as an
/// `Option` discriminant to avoid spatial overhead. Here `layout`
/// is chosen for simplicity.
#[derive(Debug)]
pub(crate) struct IterVectorsMut<'a, T> {
    lower: NonNull<T>,
    upper: NonNull<T>,
    layout: Option<Layout>,
    marker: PhantomData<&'a mut T>,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    axis_stride: NonZero<usize>,
    vector_stride: NonZero<usize>,
    vector_length: NonZero<usize>,
}

unsafe impl<T> Send for IterVectorsMut<'_, T> where T: Send {}
unsafe impl<T> Sync for IterVectorsMut<'_, T> where T: Sync {}

impl<'a, T> IterVectorsMut<'a, T> {
    pub(crate) fn over_major_axis(matrix: &'a mut Matrix<T>) -> Self {
        if matrix.is_empty() {
            return Self::empty();
        }

        let matrix_stride = matrix.stride();

        unsafe {
            let base = NonNull::new_unchecked(matrix.data.as_mut_ptr());
            let axis_stride = NonZero::new_unchecked(matrix_stride.major());
            let axis_length = NonZero::new_unchecked(matrix.major());
            let vector_stride = NonZero::new_unchecked(matrix_stride.minor());
            let vector_length = NonZero::new_unchecked(matrix.minor());

            Self::assemble(base, axis_stride, axis_length, vector_stride, vector_length)
        }
    }

    pub(crate) fn over_minor_axis(matrix: &'a mut Matrix<T>) -> Self {
        if matrix.is_empty() {
            return Self::empty();
        }

        let matrix_stride = matrix.stride();

        unsafe {
            let base = NonNull::new_unchecked(matrix.data.as_mut_ptr());
            let axis_stride = NonZero::new_unchecked(matrix_stride.minor());
            let axis_length = NonZero::new_unchecked(matrix.minor());
            let vector_stride = NonZero::new_unchecked(matrix_stride.major());
            let vector_length = NonZero::new_unchecked(matrix.major());

            Self::assemble(base, axis_stride, axis_length, vector_stride, vector_length)
        }
    }

    /// # Safety
    ///
    /// This returns a detached iterator whose lifetime is not bound to
    /// any matrix. However, it is safe.
    fn empty() -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            layout: None,
            marker: PhantomData,
        }
    }

    /// This is a helper function that abstracts some repetitive code,
    /// while exposing certain `unsafe` operations that were previously
    /// well-encapsulated. As its signature implies, this function only
    /// accepts non-empty matrices.
    ///
    /// # Safety
    ///
    /// To iterate over the vectors of a matrix, `base` must point to the
    /// underlying buffer of that matrix, and the remaining arguments must
    /// be one of the following sets of values in their [`NonZero`] form:
    ///
    /// - For iterating over the major axis:
    ///   - `axis_stride`: `matrix.stride().major()`
    ///   - `axis_length`: `matrix.major()`
    ///   - `vector_stride`: `matrix.stride().minor()` (i.e., `1`)
    ///   - `vector_length`: `matrix.minor()`
    ///
    /// - For iterating over the minor axis:
    ///   - `axis_stride`: `matrix.stride().minor()` (i.e., `1`)
    ///   - `axis_length`: `matrix.minor()`
    ///   - `vector_stride`: `matrix.stride().major()`
    ///   - `vector_length`: `matrix.major()`
    ///
    /// This returns a detached iterator whose lifetime is not bound to
    /// any matrix. The returned iterator is valid only if the matrix
    /// remains in scope.
    unsafe fn assemble(
        base: NonNull<T>,
        axis_stride: NonZero<usize>,
        axis_length: NonZero<usize>,
        vector_stride: NonZero<usize>,
        vector_length: NonZero<usize>,
    ) -> Self {
        let lower = base;
        let offset = axis_stride.get() * (axis_length.get() - 1);
        let upper = if size_of::<T>() == 0 {
            let addr = lower.addr().get() + offset;
            let ptr = without_provenance_mut(addr);
            unsafe { NonNull::new_unchecked(ptr) }
        } else {
            unsafe { lower.add(offset) }
        };
        let layout = Some(Layout {
            axis_stride,
            vector_stride,
            vector_length,
        });

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
        let layout = self.layout?;

        let result = unsafe {
            IterNthVectorMut::assemble(self.lower, layout.vector_stride, layout.vector_length)
        };

        if self.lower == self.upper {
            self.layout = None;
        } else {
            let stride = layout.axis_stride.get();
            self.lower = if size_of::<T>() == 0 {
                let addr = self.lower.addr().get() + stride;
                let ptr = without_provenance_mut(addr);
                unsafe { NonNull::new_unchecked(ptr) }
            } else {
                unsafe { self.lower.add(stride) }
            };
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterVectorsMut<'_, T> {
    fn len(&self) -> usize {
        match self.layout {
            None => 0,
            Some(layout) => {
                let upper = self.upper.addr().get();
                let lower = self.lower.addr().get();
                let stride = layout.axis_stride.get();
                let elem_size = size_of::<T>();
                1 + (upper - lower) / (stride * if elem_size == 0 { 1 } else { elem_size })
            }
        }
    }
}

impl<T> DoubleEndedIterator for IterVectorsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let layout = self.layout?;

        let result = unsafe {
            IterNthVectorMut::assemble(self.upper, layout.vector_stride, layout.vector_length)
        };

        if self.lower == self.upper {
            self.layout = None;
        } else {
            let stride = layout.axis_stride.get();
            self.upper = if size_of::<T>() == 0 {
                let addr = self.upper.addr().get() - stride;
                let ptr = without_provenance_mut(addr);
                unsafe { NonNull::new_unchecked(ptr) }
            } else {
                unsafe { self.upper.sub(stride) }
            };
        }

        Some(result)
    }
}

/// # Design Details
///
/// To prevent pointers from exceeding their provenance, `upper` must
/// point **to** (not `stride` elements past) the exact element that
/// [`DoubleEndedIterator::next_back`] would return. In this case,
/// comparing pointers to determine whether an iterator is empty is
/// unsound, because pointers may offset outside the allocated object
/// in the final iteration. Therefore, an additional state is needed
/// to track whether the iterator is empty. This state can be stored
/// in either `lower`, `upper`, or `stride` as an `Option` discriminant
/// to avoid spatial overhead. Here `stride` is chosen for simplicity.
#[derive(Debug)]
pub(crate) struct IterNthVectorMut<'a, T> {
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
    pub(crate) fn over_major_axis_vector(matrix: &'a mut Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.major() {
            return Err(Error::IndexOutOfBounds);
        }

        if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let lower = if size_of::<T>() == 0 {
            // would work, trust me
            base
        } else {
            let offset = n * matrix_stride.major();
            unsafe { base.add(offset) }
        };
        let stride = unsafe { NonZero::new_unchecked(matrix_stride.minor()) };
        let length = unsafe { NonZero::new_unchecked(matrix.minor()) };

        unsafe { Ok(Self::assemble(lower, stride, length)) }
    }

    /// This is an alternative to [`Matrix::iter_nth_minor_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(crate) fn over_minor_axis_vector(matrix: &'a mut Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.minor() {
            return Err(Error::IndexOutOfBounds);
        }

        if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = unsafe { NonNull::new_unchecked(matrix.data.as_mut_ptr()) };
        let matrix_stride = matrix.stride();
        let lower = if size_of::<T>() == 0 {
            // would work, trust me
            base
        } else {
            let offset = n * matrix_stride.minor();
            unsafe { base.add(offset) }
        };
        let stride = unsafe { NonZero::new_unchecked(matrix_stride.major()) };
        let length = unsafe { NonZero::new_unchecked(matrix.major()) };

        unsafe { Ok(Self::assemble(lower, stride, length)) }
    }

    /// # Safety
    ///
    /// This returns a detached iterator whose lifetime is not bound to
    /// any matrix. However, it is safe.
    fn empty() -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            stride: None,
            marker: PhantomData,
        }
    }

    /// This is a helper function that abstracts some repetitive code,
    /// while exposing certain `unsafe` operations that were previously
    /// well-encapsulated. As its signature implies, this function only
    /// accepts non-empty vectors.
    ///
    /// # Safety
    ///
    /// To iterate over the nth vector of a matrix, `lower` must point to
    /// the start of that vector, and the remaining arguments must be one
    /// of the following sets of values in their [`NonZero`] form:
    ///
    /// - For iterating over a major axis vector:
    ///   - `stride`: `matrix.stride().minor()` (i.e., `1`)
    ///   - `length`: `matrix.minor()`
    ///
    /// - For iterating over a minor axis vector:
    ///   - `stride`: `matrix.stride().major()`
    ///   - `length`: `matrix.major()`
    ///
    /// This returns a detached iterator whose lifetime is not bound to
    /// any matrix. The returned iterator is valid only if the matrix
    /// remains in scope.
    unsafe fn assemble(lower: NonNull<T>, stride: NonZero<usize>, length: NonZero<usize>) -> Self {
        let offset = stride.get() * (length.get() - 1);
        let upper = if size_of::<T>() == 0 {
            let addr = lower.addr().get() + offset;
            let ptr = without_provenance_mut(addr);
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

        let result = if size_of::<T>() == 0 {
            unsafe { NonNull::dangling().as_mut() }
        } else {
            unsafe { self.lower.as_mut() }
        };

        if self.lower == self.upper {
            self.stride = None;
        } else {
            self.lower = if size_of::<T>() == 0 {
                let addr = self.lower.addr().get() + stride;
                let ptr = without_provenance_mut(addr);
                unsafe { NonNull::new_unchecked(ptr) }
            } else {
                unsafe { self.lower.add(stride) }
            };
        }

        Some(result)
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
                let stride = stride.get();
                let elem_size = size_of::<T>();
                1 + (upper - lower) / (stride * if elem_size == 0 { 1 } else { elem_size })
            }
        }
    }
}

impl<T> DoubleEndedIterator for IterNthVectorMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let stride = self.stride?.get();

        let result = if size_of::<T>() == 0 {
            unsafe { NonNull::dangling().as_mut() }
        } else {
            unsafe { self.upper.as_mut() }
        };

        if self.lower == self.upper {
            self.stride = None;
        } else {
            self.upper = if size_of::<T>() == 0 {
                let addr = self.upper.addr().get() - stride;
                let ptr = without_provenance_mut(addr);
                unsafe { NonNull::new_unchecked(ptr) }
            } else {
                unsafe { self.upper.sub(stride) }
            };
        }

        Some(result)
    }
}
