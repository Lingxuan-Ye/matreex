use crate::Matrix;
use crate::error::{Error, Result};
use std::num::NonZero;
use std::ptr::{NonNull, without_provenance_mut};

// To prevent pointers from exceeding their provenance, `upper`
// must point **to** (not one vector past) the exact vector that
// [`DoubleEndedIterator::next_back`] would return. In this case,
// comparing pointers to determine whether an iterator is empty
// is unsound, because pointers may offset outside the allocated
// object in the final iteration. Therefore, an additional state
// is needed to track whether the iterator is empty. This state
// can be stored in either `lower`, `upper`, or `layout` as an
// `Option` discriminant to avoid spatial overhead. Here `layout`
// is chosen for simplicity.

/// # Safety
///
/// This does not track the validity of its underlying pointers.
#[derive(Debug)]
pub(super) struct IterVectorsInner<T> {
    lower: NonNull<T>,
    upper: NonNull<T>,
    layout: Option<Layout>,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    axis_stride: NonZero<usize>,
    vector_stride: NonZero<usize>,
    vector_length: NonZero<usize>,
}

impl<T> IterVectorsInner<T> {
    pub(crate) fn over_major_axis(matrix: &mut Matrix<T>) -> Self {
        if matrix.is_empty() {
            return Self::empty();
        }

        unsafe {
            let axis_stride = NonZero::new_unchecked(matrix.major_stride());
            let axis_length = NonZero::new_unchecked(matrix.major());
            let vector_stride = NonZero::new_unchecked(matrix.minor_stride());
            let vector_length = NonZero::new_unchecked(matrix.minor());

            Self::assemble(
                &mut matrix.data,
                axis_stride,
                axis_length,
                vector_stride,
                vector_length,
            )
        }
    }

    pub(crate) fn over_minor_axis(matrix: &mut Matrix<T>) -> Self {
        if matrix.is_empty() {
            return Self::empty();
        }

        unsafe {
            let axis_stride = NonZero::new_unchecked(matrix.minor_stride());
            let axis_length = NonZero::new_unchecked(matrix.minor());
            let vector_stride = NonZero::new_unchecked(matrix.major_stride());
            let vector_length = NonZero::new_unchecked(matrix.major());

            Self::assemble(
                &mut matrix.data,
                axis_stride,
                axis_length,
                vector_stride,
                vector_length,
            )
        }
    }

    /// This is a helper function that abstracts some repetitive code,
    /// while breaking the safety boundary and exposing `unsafe` operations
    /// that were originally well-encapsulated.
    ///
    /// # Safety
    ///
    /// To iterate over the vectors of a [`Matrix<T>`] (referred to as
    /// `matrix`), `data` must be a mutable reference to `matrix.data`,
    /// and the remaining arguments must be one of the following sets
    /// of values in their [`NonZero`] form:
    ///
    /// - For iterating over the major axis:
    ///   - `axis_stride`: `matrix.major_stride()`
    ///   - `axis_length`: `matrix.major()`
    ///   - `vector_stride`: `matrix.minor_stride()` (i.e., `1`)
    ///   - `vector_length`: `matrix.minor()`
    ///
    /// - For iterating over the minor axis:
    ///   - `axis_stride`: `matrix.minor_stride()` (i.e., `1`)
    ///   - `axis_length`: `matrix.minor()`
    ///   - `vector_stride`: `matrix.major_stride()`
    ///   - `vector_length`: `matrix.major()`
    unsafe fn assemble(
        data: &mut Vec<T>,
        axis_stride: NonZero<usize>,
        axis_length: NonZero<usize>,
        vector_stride: NonZero<usize>,
        vector_length: NonZero<usize>,
    ) -> Self {
        let base = data.as_mut_ptr();
        let lower = unsafe { NonNull::new_unchecked(base) };

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
        }
    }

    fn empty() -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            layout: None,
        }
    }
}

impl<T> Iterator for IterVectorsInner<T> {
    type Item = IterNthVectorInner<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let layout = self.layout?;

        let result = unsafe {
            IterNthVectorInner::assemble(self.lower, layout.vector_stride, layout.vector_length)
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
        let len = match self.layout {
            None => 0,
            Some(strides) => {
                let stride = strides.axis_stride.get();
                let elem_size = size_of::<T>();
                1 + (self.upper.addr().get() - self.lower.addr().get())
                    / (stride * if elem_size == 0 { 1 } else { elem_size })
            }
        };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterVectorsInner<T> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

impl<T> DoubleEndedIterator for IterVectorsInner<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let layout = self.layout?;

        let result = unsafe {
            IterNthVectorInner::assemble(self.upper, layout.vector_stride, layout.vector_length)
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

// To prevent pointers from exceeding their provenance, `upper` must
// point **to** (not `stride` elements past) the exact element that
// [`DoubleEndedIterator::next_back`] would return. In this case,
// comparing pointers to determine whether an iterator is empty is
// unsound, because pointers may offset outside the allocated object
// in the final iteration. Therefore, an additional state is needed
// to track whether the iterator is empty. This state can be stored
// in either `lower`, `upper`, or `stride` as an `Option` discriminant
// to avoid spatial overhead. Here `stride` is chosen for simplicity.

/// # Safety
///
/// This does not track the validity of its underlying pointers.
#[derive(Debug)]
pub(super) struct IterNthVectorInner<T> {
    lower: NonNull<T>,
    upper: NonNull<T>,
    stride: Option<NonZero<usize>>,
}

impl<T> IterNthVectorInner<T> {
    pub(super) fn over_major_axis(matrix: &mut Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.major() {
            return Err(Error::IndexOutOfBounds);
        } else if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = matrix.data.as_mut_ptr();
        let lower = if size_of::<T>() == 0 {
            // would work, trust me
            unsafe { NonNull::new_unchecked(base) }
        } else {
            let offset = n * matrix.major_stride();
            unsafe { NonNull::new_unchecked(base.add(offset)) }
        };
        let stride = unsafe { NonZero::new_unchecked(matrix.minor_stride()) };
        let length = unsafe { NonZero::new_unchecked(matrix.minor()) };

        unsafe { Ok(Self::assemble(lower, stride, length)) }
    }

    pub(super) fn over_minor_axis(matrix: &mut Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.minor() {
            return Err(Error::IndexOutOfBounds);
        } else if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = matrix.data.as_mut_ptr();
        let lower = if size_of::<T>() == 0 {
            // would work, trust me
            unsafe { NonNull::new_unchecked(base) }
        } else {
            let offset = n * matrix.minor_stride();
            unsafe { NonNull::new_unchecked(base.add(offset)) }
        };
        let stride = unsafe { NonZero::new_unchecked(matrix.major_stride()) };
        let length = unsafe { NonZero::new_unchecked(matrix.major()) };

        unsafe { Ok(Self::assemble(lower, stride, length)) }
    }

    /// This is a helper function that abstracts some repetitive code,
    /// while breaking the safety boundary and exposing `unsafe` operations
    /// that were originally well-encapsulated.
    ///
    /// # Safety
    ///
    /// To iterate over the nth vector of a [`Matrix<T>`] (referred to as
    /// `matrix`), `lower` must point to the head of that vector, and the
    /// remaining arguments must be one of the following sets of values in
    /// their [`NonZero`] form:
    ///
    /// - For iterating over the major axis vector:
    ///   - `stride`: `matrix.minor_stride()` (i.e., `1`)
    ///   - `length`: `matrix.minor()`
    ///
    /// - For iterating over the minor axis vector:
    ///   - `stride`: `matrix.major_stride()`
    ///   - `length`: `matrix.major()`
    unsafe fn assemble(lower: NonNull<T>, stride: NonZero<usize>, length: NonZero<usize>) -> Self {
        let offset = (length.get() - 1) * stride.get();
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
        }
    }

    fn empty() -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            stride: None,
        }
    }
}

impl<T> Iterator for IterNthVectorInner<T> {
    type Item = NonNull<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let stride = self.stride?.get();

        let result = if size_of::<T>() == 0 {
            NonNull::dangling()
        } else {
            self.lower
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
        let len = match self.stride {
            None => 0,
            Some(stride) => {
                let stride = stride.get();
                let elem_size = size_of::<T>();
                1 + (self.upper.addr().get() - self.lower.addr().get())
                    / (stride * if elem_size == 0 { 1 } else { elem_size })
            }
        };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterNthVectorInner<T> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

impl<T> DoubleEndedIterator for IterNthVectorInner<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let stride = self.stride?.get();

        let result = if size_of::<T>() == 0 {
            NonNull::dangling()
        } else {
            self.upper
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
