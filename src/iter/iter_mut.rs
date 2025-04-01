use super::inner::{IterNthVectorInner, IterVectorsInner};
use crate::Matrix;
use crate::error::Result;
use std::marker::PhantomData;

#[derive(Debug)]
pub(crate) struct IterVectorsMut<'a, T> {
    inner: IterVectorsInner<T>,
    marker: PhantomData<&'a mut T>,
}

unsafe impl<T: Send> Send for IterVectorsMut<'_, T> {}
unsafe impl<T: Sync> Sync for IterVectorsMut<'_, T> {}

impl<'a, T> IterVectorsMut<'a, T> {
    pub(crate) fn over_major_axis(matrix: &'a mut Matrix<T>) -> Self {
        let inner = IterVectorsInner::over_major_axis(matrix);
        let marker = PhantomData;
        Self { inner, marker }
    }

    pub(crate) fn over_minor_axis(matrix: &'a mut Matrix<T>) -> Self {
        let inner = IterVectorsInner::over_minor_axis(matrix);
        let marker = PhantomData;
        Self { inner, marker }
    }
}

impl<'a, T> Iterator for IterVectorsMut<'a, T> {
    type Item = IterNthVectorMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let inner = self.inner.next()?;
        let marker = PhantomData;
        Some(IterNthVectorMut { inner, marker })
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
        let inner = self.inner.next_back()?;
        let marker = PhantomData;
        Some(IterNthVectorMut { inner, marker })
    }
}

#[derive(Debug)]
pub(crate) struct IterNthVectorMut<'a, T> {
    inner: IterNthVectorInner<T>,
    marker: PhantomData<&'a mut T>,
}

unsafe impl<T: Send> Send for IterNthVectorMut<'_, T> {}
unsafe impl<T: Sync> Sync for IterNthVectorMut<'_, T> {}

impl<'a, T> IterNthVectorMut<'a, T> {
    /// This is an alternative to [`Matrix::iter_nth_major_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(crate) fn over_major_axis(matrix: &'a mut Matrix<T>, n: usize) -> Result<Self> {
        let inner = IterNthVectorInner::over_major_axis(matrix, n)?;
        let marker = PhantomData;
        Ok(Self { inner, marker })
    }

    /// This is an alternative to [`Matrix::iter_nth_minor_axis_vector_mut`],
    /// but slightly slower.
    #[allow(dead_code)]
    pub(crate) fn over_minor_axis(matrix: &'a mut Matrix<T>, n: usize) -> Result<Self> {
        let inner = IterNthVectorInner::over_minor_axis(matrix, n)?;
        let marker = PhantomData;
        Ok(Self { inner, marker })
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
