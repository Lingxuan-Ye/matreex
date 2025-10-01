use self::iter_mut::IterVectorsMut;
use super::Matrix;
use super::layout::{Order, OrderKind};
use crate::error::{Error, Result};
use crate::index::Index;
use core::iter::{Skip, StepBy, Take};
use core::slice::{Iter, IterMut};

mod iter_mut;

pub trait ExactSizeDoubleEndedIterator: ExactSizeIterator + DoubleEndedIterator {}

impl<I> ExactSizeDoubleEndedIterator for I where I: ExactSizeIterator + DoubleEndedIterator {}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn iter_rows(
        &self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &T>>
    {
        (0..self.nrows()).map(|n| match O::KIND {
            OrderKind::RowMajor => self.iter_nth_major_axis_vector_unchecked(n),
            OrderKind::ColMajor => self.iter_nth_minor_axis_vector_unchecked(n),
        })
    }

    pub fn iter_cols(
        &self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &T>>
    {
        (0..self.ncols()).map(|n| match O::KIND {
            OrderKind::RowMajor => self.iter_nth_minor_axis_vector_unchecked(n),
            OrderKind::ColMajor => self.iter_nth_major_axis_vector_unchecked(n),
        })
    }

    pub fn iter_rows_mut(
        &mut self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &mut T>>
    {
        match O::KIND {
            OrderKind::RowMajor => IterVectorsMut::over_major_axis(self),
            OrderKind::ColMajor => IterVectorsMut::over_minor_axis(self),
        }
    }

    pub fn iter_cols_mut(
        &mut self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &mut T>>
    {
        match O::KIND {
            OrderKind::RowMajor => IterVectorsMut::over_minor_axis(self),
            OrderKind::ColMajor => IterVectorsMut::over_major_axis(self),
        }
    }

    pub fn iter_nth_row(&self, n: usize) -> Result<impl ExactSizeDoubleEndedIterator<Item = &T>> {
        match O::KIND {
            OrderKind::RowMajor => self.iter_nth_major_axis_vector(n),
            OrderKind::ColMajor => self.iter_nth_minor_axis_vector(n),
        }
    }

    pub fn iter_nth_col(&self, n: usize) -> Result<impl ExactSizeDoubleEndedIterator<Item = &T>> {
        match O::KIND {
            OrderKind::RowMajor => self.iter_nth_minor_axis_vector(n),
            OrderKind::ColMajor => self.iter_nth_major_axis_vector(n),
        }
    }

    pub fn iter_nth_row_mut(
        &mut self,
        n: usize,
    ) -> Result<impl ExactSizeDoubleEndedIterator<Item = &mut T>> {
        match O::KIND {
            OrderKind::RowMajor => self.iter_nth_major_axis_vector_mut(n),
            OrderKind::ColMajor => self.iter_nth_minor_axis_vector_mut(n),
        }
    }

    pub fn iter_nth_col_mut(
        &mut self,
        n: usize,
    ) -> Result<impl ExactSizeDoubleEndedIterator<Item = &mut T>> {
        match O::KIND {
            OrderKind::RowMajor => self.iter_nth_minor_axis_vector_mut(n),
            OrderKind::ColMajor => self.iter_nth_major_axis_vector_mut(n),
        }
    }

    pub fn iter_elements(&self) -> impl ExactSizeDoubleEndedIterator<Item = &T> {
        self.data.iter()
    }

    pub fn iter_elements_mut(&mut self) -> impl ExactSizeDoubleEndedIterator<Item = &mut T> {
        self.data.iter_mut()
    }

    pub fn into_iter_elements(self) -> impl ExactSizeDoubleEndedIterator<Item = T> {
        self.data.into_iter()
    }

    pub fn iter_elements_with_index(
        &self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = (Index, &T)> {
        let stride = self.stride();
        self.data.iter().enumerate().map(move |(index, element)| {
            let index = Index::from_flattened::<O>(index, stride);
            (index, element)
        })
    }

    pub fn iter_elements_mut_with_index(
        &mut self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = (Index, &mut T)> {
        let stride = self.stride();
        self.data
            .iter_mut()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }

    pub fn into_iter_elements_with_index(
        self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = (Index, T)> {
        let stride = self.stride();
        self.data
            .into_iter()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    fn iter_nth_major_axis_vector(&self, n: usize) -> Result<Take<StepBy<Skip<Iter<'_, T>>>>> {
        if n >= self.major() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_major_axis_vector_unchecked(n))
        }
    }

    fn iter_nth_minor_axis_vector(&self, n: usize) -> Result<Take<StepBy<Skip<Iter<'_, T>>>>> {
        if n >= self.minor() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_minor_axis_vector_unchecked(n))
        }
    }

    fn iter_nth_major_axis_vector_mut(
        &mut self,
        n: usize,
    ) -> Result<Take<StepBy<Skip<IterMut<'_, T>>>>> {
        if n >= self.major() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_major_axis_vector_unchecked_mut(n))
        }
    }

    fn iter_nth_minor_axis_vector_mut(
        &mut self,
        n: usize,
    ) -> Result<Take<StepBy<Skip<IterMut<'_, T>>>>> {
        if n >= self.minor() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_minor_axis_vector_unchecked_mut(n))
        }
    }

    fn iter_nth_major_axis_vector_unchecked(&self, n: usize) -> Take<StepBy<Skip<Iter<'_, T>>>> {
        let stride = self.stride();
        let skip = n * stride.major();
        let step = stride.minor();
        let take = self.minor();
        self.data.iter().skip(skip).step_by(step).take(take)
    }

    fn iter_nth_minor_axis_vector_unchecked(&self, n: usize) -> Take<StepBy<Skip<Iter<'_, T>>>> {
        let stride = self.stride();
        let skip = n * stride.minor();
        let step = stride.major();
        let take = self.major();
        self.data.iter().skip(skip).step_by(step).take(take)
    }

    fn iter_nth_major_axis_vector_unchecked_mut(
        &mut self,
        n: usize,
    ) -> Take<StepBy<Skip<IterMut<'_, T>>>> {
        let stride = self.stride();
        let skip = n * stride.major();
        let step = stride.minor();
        let take = self.minor();
        self.data.iter_mut().skip(skip).step_by(step).take(take)
    }

    fn iter_nth_minor_axis_vector_unchecked_mut(
        &mut self,
        n: usize,
    ) -> Take<StepBy<Skip<IterMut<'_, T>>>> {
        let stride = self.stride();
        let skip = n * stride.minor();
        let step = stride.major();
        let take = self.major();
        self.data.iter_mut().skip(skip).step_by(step).take(take)
    }
}
