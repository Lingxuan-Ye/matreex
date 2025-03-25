use crate::Matrix;
use crate::error::{Error, Result};
use std::iter::{Skip, StepBy, Take};
use std::slice::Iter;

#[derive(Debug)]
pub(crate) struct IterNthVector;

impl IterNthVector {
    pub(crate) fn on_major_axis<T>(
        matrix: &Matrix<T>,
        n: usize,
    ) -> Result<Take<StepBy<Skip<Iter<'_, T>>>>> {
        if n >= matrix.major() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(Self::on_major_axis_unchecked(matrix, n))
        }
    }

    pub(crate) fn on_minor_axis<T>(
        matrix: &Matrix<T>,
        n: usize,
    ) -> Result<Take<StepBy<Skip<Iter<'_, T>>>>> {
        if n >= matrix.minor() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(Self::on_minor_axis_unchecked(matrix, n))
        }
    }

    pub(crate) fn on_major_axis_unchecked<T>(
        matrix: &Matrix<T>,
        n: usize,
    ) -> Take<StepBy<Skip<Iter<'_, T>>>> {
        let skip = n * matrix.major_stride();
        let step = matrix.minor_stride();
        let take = matrix.minor();
        matrix.data.iter().skip(skip).step_by(step).take(take)
    }

    pub(crate) fn on_minor_axis_unchecked<T>(
        matrix: &Matrix<T>,
        n: usize,
    ) -> Take<StepBy<Skip<Iter<'_, T>>>> {
        let skip = n;
        let step = matrix.major_stride();
        let take = matrix.major();
        matrix.data.iter().skip(skip).step_by(step).take(take)
    }
}
