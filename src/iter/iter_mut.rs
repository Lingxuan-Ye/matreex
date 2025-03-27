use crate::Matrix;
use crate::error::{Error, Result};
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr::NonNull;

// To prevent pointers from exceeding their provenance, `upper`
// must point **to** (not one vector past) the exact vector that
// [`DoubleEndedIterator::next_back`] would return. In this case,
// comparing pointers to determine whether an iterator is empty is
// unsound, because pointers may offset outside the allocated object
// in the final iteration. Therefore, an additional state is needed
// to track whether the iterator is empty. This state can be stored
// in either `lower`, `upper`, or `vector_layout` as an `Option`
// discriminant to avoid spatial overhead. Here `vector_layout` is
// chosen for simplicity.
#[derive(Debug)]
pub(crate) struct IterVectorsMut<'a, T> {
    lower: NonNull<T>,
    upper: NonNull<T>,
    vector_layout: Option<VectorLayout>,
    _marker: PhantomData<&'a mut T>,
}

#[derive(Clone, Copy, Debug)]
struct VectorLayout {
    inter_stride: NonZero<usize>,
    intra_stride: NonZero<usize>,
    length: NonZero<usize>,
}

unsafe impl<T: Send> Send for IterVectorsMut<'_, T> {}
unsafe impl<T: Sync> Sync for IterVectorsMut<'_, T> {}

impl<T> IterVectorsMut<'_, T> {
    pub(crate) fn on_major_axis(matrix: &mut Matrix<T>) -> Self {
        if matrix.is_empty() {
            return Self::empty();
        }

        // Guarantees:
        // - `matrix.major()` is greater than zero
        // - `matrix.minor()` is greater than zero
        // - `matrix.major_stride()` is greater than zero
        // - `matrix.data.as_mut_ptr()` is not dangling if `T` is not a ZST

        let base = matrix.data.as_mut_ptr();
        let lower = unsafe { NonNull::new_unchecked(base) };

        let stride = matrix.major_stride();
        let offset = (matrix.major() - 1) * stride;
        let upper = if size_of::<T>() == 0 {
            let addr = lower.as_ptr() as usize + offset;
            unsafe { NonNull::new_unchecked(addr as *mut T) }
        } else {
            unsafe { lower.add(offset) }
        };

        let vector_layout = unsafe {
            Some(VectorLayout {
                inter_stride: NonZero::new_unchecked(stride),
                intra_stride: NonZero::new_unchecked(1),
                length: NonZero::new_unchecked(matrix.minor()),
            })
        };

        Self {
            lower,
            upper,
            vector_layout,
            _marker: PhantomData,
        }
    }

    pub(crate) fn on_minor_axis(matrix: &mut Matrix<T>) -> Self {
        if matrix.is_empty() {
            return Self::empty();
        }

        // Guarantees:
        // - `matrix.major()` is greater than zero
        // - `matrix.minor()` is greater than zero
        // - `matrix.major_stride()` is greater than zero
        // - `matrix.data.as_mut_ptr()` is not dangling if `T` is not a ZST

        let base = matrix.data.as_mut_ptr();
        let lower = unsafe { NonNull::new_unchecked(base) };

        let stride = 1;
        let offset = (matrix.minor() - 1) * stride;
        let upper = if size_of::<T>() == 0 {
            let addr = lower.as_ptr() as usize + offset;
            unsafe { NonNull::new_unchecked(addr as *mut T) }
        } else {
            unsafe { lower.add(offset) }
        };

        let vector_layout = unsafe {
            Some(VectorLayout {
                inter_stride: NonZero::new_unchecked(stride),
                intra_stride: NonZero::new_unchecked(matrix.major_stride()),
                length: NonZero::new_unchecked(matrix.major()),
            })
        };

        Self {
            lower,
            upper,
            vector_layout,
            _marker: PhantomData,
        }
    }

    fn empty() -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            vector_layout: None,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for IterVectorsMut<'a, T> {
    type Item = IterNthVectorMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let vector_layout = self.vector_layout?;
        let stride = vector_layout.inter_stride.get();
        let result = unsafe {
            IterNthVectorMut::new_unchecked(
                self.lower,
                vector_layout.intra_stride,
                vector_layout.length,
            )
        };

        if self.lower == self.upper {
            self.vector_layout = None;
        } else {
            self.lower = if size_of::<T>() == 0 {
                let addr = self.lower.as_ptr() as usize + stride;
                unsafe { NonNull::new_unchecked(addr as *mut T) }
            } else {
                unsafe { self.lower.add(stride) }
            };
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self.vector_layout {
            None => 0,
            Some(strides) => {
                let stride = strides.inter_stride.get();
                let elem_size = size_of::<T>();
                1 + (self.upper.as_ptr() as usize - self.lower.as_ptr() as usize)
                    / (stride * if elem_size == 0 { 1 } else { elem_size })
            }
        };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterVectorsMut<'_, T> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

impl<T> DoubleEndedIterator for IterVectorsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let vector_layout = self.vector_layout?;
        let stride = vector_layout.inter_stride.get();
        let result = unsafe {
            IterNthVectorMut::new_unchecked(
                self.upper,
                vector_layout.intra_stride,
                vector_layout.length,
            )
        };

        if self.lower == self.upper {
            self.vector_layout = None;
        } else {
            self.upper = if size_of::<T>() == 0 {
                let addr = self.upper.as_ptr() as usize - stride;
                unsafe { NonNull::new_unchecked(addr as *mut T) }
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
#[derive(Debug)]
pub(crate) struct IterNthVectorMut<'a, T> {
    lower: NonNull<T>,
    upper: NonNull<T>,
    stride: Option<NonZero<usize>>,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<T: Send> Send for IterNthVectorMut<'_, T> {}
unsafe impl<T: Sync> Sync for IterNthVectorMut<'_, T> {}

impl<T> IterNthVectorMut<'_, T> {
    /// This is an alternative to [`Matrix::iter_nth_major_axis_vector`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(crate) fn on_major_axis(matrix: &mut Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.major() {
            return Err(Error::IndexOutOfBounds);
        } else if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = matrix.data.as_mut_ptr();
        let lower = if size_of::<T>() == 0 {
            // it would work, trust me
            unsafe { NonNull::new_unchecked(base) }
        } else {
            let offset = n * matrix.major_stride();
            unsafe { NonNull::new_unchecked(base.add(offset)) }
        };
        let stride = unsafe { NonZero::new_unchecked(1) };
        let length = unsafe { NonZero::new_unchecked(matrix.minor()) };

        unsafe { Ok(Self::new_unchecked(lower, stride, length)) }
    }

    /// This is an alternative to [`Matrix::iter_nth_minor_axis_vector`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(crate) fn on_minor_axis(matrix: &mut Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.minor() {
            return Err(Error::IndexOutOfBounds);
        } else if matrix.is_empty() {
            return Ok(Self::empty());
        }

        let base = matrix.data.as_mut_ptr();
        let lower = if size_of::<T>() == 0 {
            // it would work, trust me
            unsafe { NonNull::new_unchecked(base) }
        } else {
            let offset = n;
            unsafe { NonNull::new_unchecked(base.add(offset)) }
        };
        let stride = unsafe { NonZero::new_unchecked(matrix.major_stride()) };
        let length = unsafe { NonZero::new_unchecked(matrix.major()) };

        unsafe { Ok(Self::new_unchecked(lower, stride, length)) }
    }

    /// # Safety
    ///
    /// - `lower` must be properly aligned. The following points assume this.
    /// - If `T` is a zero-sized type, it is always safe.
    /// - If `T` is not a zero-sized type, starting from `lower`, every
    ///   `stride * size_of::<T>()`-th pointer must have exclusive write
    ///   access to its memory, up to the `(length - 1)`-th pointer.
    unsafe fn new_unchecked(
        lower: NonNull<T>,
        stride: NonZero<usize>,
        length: NonZero<usize>,
    ) -> Self {
        let offset = (length.get() - 1) * stride.get();
        let upper = if size_of::<T>() == 0 {
            let addr = lower.as_ptr() as usize + offset;
            unsafe { NonNull::new_unchecked(addr as *mut T) }
        } else {
            unsafe { lower.add(offset) }
        };

        Self {
            lower,
            upper,
            stride: Some(stride),
            _marker: PhantomData,
        }
    }

    fn empty() -> Self {
        Self {
            lower: NonNull::dangling(),
            upper: NonNull::dangling(),
            stride: None,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for IterNthVectorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let stride = self.stride?.get();
        let result = unsafe { self.lower.as_mut() };

        if self.lower == self.upper {
            self.stride = None;
        } else {
            self.lower = if size_of::<T>() == 0 {
                let addr = self.lower.as_ptr() as usize + stride;
                unsafe { NonNull::new_unchecked(addr as *mut T) }
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
                1 + (self.upper.as_ptr() as usize - self.lower.as_ptr() as usize)
                    / (stride * if elem_size == 0 { 1 } else { elem_size })
            }
        };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterNthVectorMut<'_, T> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

impl<T> DoubleEndedIterator for IterNthVectorMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let stride = self.stride?.get();
        let result = unsafe { self.upper.as_mut() };

        if self.lower == self.upper {
            self.stride = None;
        } else {
            self.upper = if size_of::<T>() == 0 {
                let addr = self.upper.as_ptr() as usize - stride;
                unsafe { NonNull::new_unchecked(addr as *mut T) }
            } else {
                unsafe { self.upper.sub(stride) }
            };
        }

        Some(result)
    }
}
