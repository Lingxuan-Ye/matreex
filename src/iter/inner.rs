use crate::error::{Error, Result};
use crate::shape::AxisShape;
use std::num::NonZero;
use std::ptr::{NonNull, without_provenance_mut};

/// # Safety
///
/// This iterator does not track the validity of its underlying
/// pointers. Iterating over a dangling [`Matrix<T>`] is *[undefined
/// behavior]* even if the resulting pointer is not used. In other
/// words, once the matrix goes out of scope, the iterator must not
/// be used unless it is empty.
///
/// Based on the above, all constructors except [`empty`] are marked
/// as `unsafe`.
///
/// # Design Details
///
/// To prevent pointers from exceeding their provenance, `upper`
/// must point **to** (not one vector past) the exact vector that
/// [`DoubleEndedIterator::next_back`] would return. In this case,
/// comparing pointers to determine whether an iterator is empty
/// is unsound, because pointers may offset outside the allocated
/// object in the final iteration. Therefore, an additional state
/// is needed to track whether the iterator is empty. This state
/// can be stored in either `lower`, `upper`, or `layout` as an
/// `Option` discriminant to avoid spatial overhead. Here `layout`
/// is chosen for simplicity.
///
/// [`Matrix<T>`]: crate::Matrix
/// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
/// [`empty`]: IterVectorsInner::empty
#[derive(Debug)]
pub(crate) struct IterVectorsInner<T> {
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
    /// # Safety
    ///
    /// To iterate over the vectors of a matrix, `buffer` must point to the
    /// underlying buffer of that matrix, and `shape` must be `matrix.shape`.
    ///
    /// The returned iterator is valid only if the matrix remains in scope.
    pub(crate) unsafe fn over_major_axis(buffer: NonNull<T>, shape: AxisShape) -> Self {
        if shape.size() == 0 {
            return Self::empty();
        }

        unsafe {
            let axis_stride = NonZero::new_unchecked(shape.major_stride());
            let axis_length = NonZero::new_unchecked(shape.major());
            let vector_stride = NonZero::new_unchecked(shape.minor_stride());
            let vector_length = NonZero::new_unchecked(shape.minor());

            Self::assemble(
                buffer,
                axis_stride,
                axis_length,
                vector_stride,
                vector_length,
            )
        }
    }

    /// # Safety
    ///
    /// To iterate over the vectors of a matrix, `buffer` must point to the
    /// underlying buffer of that matrix, and `shape` must be `matrix.shape`.
    ///
    /// The returned iterator is valid only if the matrix remains in scope.
    pub(crate) unsafe fn over_minor_axis(buffer: NonNull<T>, shape: AxisShape) -> Self {
        if shape.size() == 0 {
            return Self::empty();
        }

        unsafe {
            let axis_stride = NonZero::new_unchecked(shape.minor_stride());
            let axis_length = NonZero::new_unchecked(shape.minor());
            let vector_stride = NonZero::new_unchecked(shape.major_stride());
            let vector_length = NonZero::new_unchecked(shape.major());

            Self::assemble(
                buffer,
                axis_stride,
                axis_length,
                vector_stride,
                vector_length,
            )
        }
    }

    /// This is a helper function that abstracts some repetitive code,
    /// while exposing certain `unsafe` operations that were perviously
    /// well-encapsulated.
    ///
    /// # Safety
    ///
    /// To iterate over the vectors of a matrix, `buffer` must point to the
    /// underlying buffer of that matrix, and the remaining arguments must
    /// be one of the following sets of values in their [`NonZero`] form:
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
    ///
    /// The returned iterator is valid only if the matrix remains in scope.
    unsafe fn assemble(
        buffer: NonNull<T>,
        axis_stride: NonZero<usize>,
        axis_length: NonZero<usize>,
        vector_stride: NonZero<usize>,
        vector_length: NonZero<usize>,
    ) -> Self {
        let lower = buffer;
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

/// # Safety
///
/// This iterator does not track the validity of its underlying
/// pointers. Iterating over a dangling [`Matrix<T>`] vector is
/// *[undefined behavior]* even if the resulting pointer is not
/// used. In other words, once the matrix goes out of scope, the
/// iterator must not be used unless it is empty.
///
/// Based on the above, all constructors except [`empty`] are marked
/// as `unsafe`.
///
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
///
/// [`Matrix<T>`]: crate::Matrix
/// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
/// [`empty`]: IterNthVectorInner::empty
#[derive(Debug)]
pub(crate) struct IterNthVectorInner<T> {
    lower: NonNull<T>,
    upper: NonNull<T>,
    stride: Option<NonZero<usize>>,
}

impl<T> IterNthVectorInner<T> {
    /// # Safety
    ///
    /// To iterate over the nth vector of a matrix, `buffer` must point to the
    /// underlying buffer of that matrix, and `shape` must be `matrix.shape`.
    ///
    /// The returned iterator is valid only if the matrix remains in scope.
    pub(crate) unsafe fn over_major_axis(
        buffer: NonNull<T>,
        shape: AxisShape,
        n: usize,
    ) -> Result<Self> {
        if n >= shape.major() {
            return Err(Error::IndexOutOfBounds);
        } else if shape.size() == 0 {
            return Ok(Self::empty());
        }

        let lower = if size_of::<T>() == 0 {
            // would work, trust me
            buffer
        } else {
            let offset = n * shape.major_stride();
            unsafe { buffer.add(offset) }
        };
        let stride = unsafe { NonZero::new_unchecked(shape.minor_stride()) };
        let length = unsafe { NonZero::new_unchecked(shape.minor()) };

        unsafe { Ok(Self::assemble(lower, stride, length)) }
    }

    /// # Safety
    ///
    /// To iterate over the nth vector of a matrix, `buffer` must point to the
    /// underlying buffer of that matrix, and `shape` must be `matrix.shape`.
    ///
    /// The returned iterator is valid only if the matrix remains in scope.
    pub(crate) unsafe fn over_minor_axis(
        buffer: NonNull<T>,
        shape: AxisShape,
        n: usize,
    ) -> Result<Self> {
        if n >= shape.minor() {
            return Err(Error::IndexOutOfBounds);
        } else if shape.size() == 0 {
            return Ok(Self::empty());
        }

        let lower = if size_of::<T>() == 0 {
            // would work, trust me
            buffer
        } else {
            let offset = n * shape.minor_stride();
            unsafe { buffer.add(offset) }
        };
        let stride = unsafe { NonZero::new_unchecked(shape.major_stride()) };
        let length = unsafe { NonZero::new_unchecked(shape.major()) };

        unsafe { Ok(Self::assemble(lower, stride, length)) }
    }

    /// This is a helper function that abstracts some repetitive code,
    /// while exposing certain `unsafe` operations that were previously
    /// well-encapsulated.
    ///
    /// # Safety
    ///
    /// To iterate over the nth vector of a matrix, `lower` must point
    /// to the head of that vector, and the remaining arguments must be
    /// one of the following sets of values in their [`NonZero`] form:
    ///
    /// - For iterating over a major axis vector:
    ///   - `stride`: `matrix.minor_stride()` (i.e., `1`)
    ///   - `length`: `matrix.minor()`
    ///
    /// - For iterating over a minor axis vector:
    ///   - `stride`: `matrix.major_stride()`
    ///   - `length`: `matrix.major()`
    ///
    /// The returned iterator is valid only if the matrix remains in scope.
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
