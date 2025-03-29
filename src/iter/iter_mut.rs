use super::inner::{IterNthVectorInner, IterVectorsInner};
use crate::Matrix;
use crate::error::Result;
use std::marker::PhantomData;

#[derive(Debug)]
pub(crate) struct IterVectorsMut<'a, T> {
    inner: IterVectorsInner<T>,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<T: Send> Send for IterVectorsMut<'_, T> {}
unsafe impl<T: Sync> Sync for IterVectorsMut<'_, T> {}

impl<T> IterVectorsMut<'_, T> {
    pub(crate) fn over_major_axis(matrix: &mut Matrix<T>) -> Self {
        let inner = IterVectorsInner::over_major_axis(matrix);
        let _marker = PhantomData;
        Self { inner, _marker }
    }

    pub(crate) fn over_minor_axis(matrix: &mut Matrix<T>) -> Self {
        let inner = IterVectorsInner::over_minor_axis(matrix);
        let _marker = PhantomData;
        Self { inner, _marker }
    }
}

impl<'a, T> Iterator for IterVectorsMut<'a, T> {
    type Item = IterNthVectorMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let iter = self.inner.next()?;
        Some(IterNthVectorMut::from_inner(iter))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for IterVectorsMut<'_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> DoubleEndedIterator for IterVectorsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let iter = self.inner.next_back()?;
        Some(IterNthVectorMut::from_inner(iter))
    }
}

#[derive(Debug)]
pub(crate) struct IterNthVectorMut<'a, T> {
    inner: IterNthVectorInner<T>,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<T: Send> Send for IterNthVectorMut<'_, T> {}
unsafe impl<T: Sync> Sync for IterNthVectorMut<'_, T> {}

impl<T> IterNthVectorMut<'_, T> {
    /// This is an alternative to [`Matrix::iter_nth_major_axis_vector`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(crate) fn over_major_axis(matrix: &mut Matrix<T>, n: usize) -> Result<Self> {
        let inner = IterNthVectorInner::over_major_axis(matrix, n)?;
        let _marker = PhantomData;
        Ok(Self { inner, _marker })
    }

    /// This is an alternative to [`Matrix::iter_nth_minor_axis_vector`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(crate) fn over_minor_axis(matrix: &mut Matrix<T>, n: usize) -> Result<Self> {
        let inner = IterNthVectorInner::over_minor_axis(matrix, n)?;
        let _marker = PhantomData;
        Ok(Self { inner, _marker })
    }

    fn from_inner(inner: IterNthVectorInner<T>) -> Self {
        let _marker = PhantomData;
        Self { inner, _marker }
    }
}

impl<'a, T> Iterator for IterNthVectorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut ptr = self.inner.next()?;
        unsafe { Some(ptr.as_mut()) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for IterNthVectorMut<'_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> DoubleEndedIterator for IterNthVectorMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut ptr = self.inner.next_back()?;
        unsafe { Some(ptr.as_mut()) }
    }
}
