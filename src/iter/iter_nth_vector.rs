use crate::Matrix;
use crate::error::{Error, Result};
use std::iter::{FusedIterator, Skip, StepBy};
use std::slice::Iter;

#[derive(Clone, Debug)]
pub(crate) enum IterNthVector<'a, T> {
    MajorAxis(Iter<'a, T>),
    MinorAxis(StepBy<Skip<Iter<'a, T>>>),
}

impl<'a, T> IterNthVector<'a, T> {
    pub(crate) fn iter_nth_major_axis_vector(matrix: &'a Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.major() {
            Err(Error::IndexOutOfBounds)
        } else {
            unsafe { Ok(Self::iter_nth_major_axis_vector_unchecked(matrix, n)) }
        }
    }

    pub(crate) fn iter_nth_minor_axis_vector(matrix: &'a Matrix<T>, n: usize) -> Result<Self> {
        if n >= matrix.minor() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(Self::iter_nth_minor_axis_vector_unchecked(matrix, n))
        }
    }

    /// # Safety
    ///
    /// Calling this method when `n >= matrix.major()` is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub(crate) unsafe fn iter_nth_major_axis_vector_unchecked(
        matrix: &'a Matrix<T>,
        n: usize,
    ) -> Self {
        let lower = n * matrix.major_stride();
        let upper = lower + matrix.major_stride();
        unsafe { Self::MajorAxis(matrix.data.get_unchecked(lower..upper).iter()) }
    }

    /// # Safety
    ///
    /// Calling this method when `n >= matrix.minor()` is erroneous but safe.
    pub(crate) fn iter_nth_minor_axis_vector_unchecked(matrix: &'a Matrix<T>, n: usize) -> Self {
        let step = matrix.major_stride();
        Self::MinorAxis(matrix.data.iter().skip(n).step_by(step))
    }
}

impl<'a, T> Iterator for IterNthVector<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::MajorAxis(iter) => iter.next(),
            Self::MinorAxis(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::MajorAxis(iter) => iter.size_hint(),
            Self::MinorAxis(iter) => iter.size_hint(),
        }
    }

    fn count(self) -> usize {
        match self {
            Self::MajorAxis(iter) => iter.count(),
            Self::MinorAxis(iter) => iter.count(),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            Self::MajorAxis(iter) => iter.nth(n),
            Self::MinorAxis(iter) => iter.nth(n),
        }
    }

    // unstable
    // fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
    //     match self {
    //         Self::Major(iter) => iter.advance_by(n),
    //         Self::Minor(iter) => iter.advance_by(n),
    //     }
    // }

    fn last(self) -> Option<Self::Item> {
        match self {
            Self::MajorAxis(iter) => iter.last(),
            Self::MinorAxis(iter) => iter.last(),
        }
    }

    // [`Try`] is unstable
    // fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    // where
    //     F: FnMut(B, Self::Item) -> R,
    //     R: Try<Output = B>,
    // {
    //     match self {
    //         Self::Major(iter) => iter.try_fold(init, f),
    //         Self::Minor(iter) => iter.try_fold(init, f),
    //     }
    // }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self {
            Self::MajorAxis(iter) => iter.fold(init, f),
            Self::MinorAxis(iter) => iter.fold(init, f),
        }
    }

    fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    {
        match self {
            Self::MajorAxis(iter) => iter.for_each(f),
            Self::MinorAxis(iter) => iter.for_each(f),
        }
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        match self {
            Self::MajorAxis(iter) => iter.all(f),
            Self::MinorAxis(iter) => iter.all(f),
        }
    }

    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        match self {
            Self::MajorAxis(iter) => iter.any(f),
            Self::MinorAxis(iter) => iter.any(f),
        }
    }

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        match self {
            Self::MajorAxis(iter) => iter.find(predicate),
            Self::MinorAxis(iter) => iter.find(predicate),
        }
    }

    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        F: FnMut(Self::Item) -> Option<B>,
    {
        match self {
            Self::MajorAxis(iter) => iter.find_map(f),
            Self::MinorAxis(iter) => iter.find_map(f),
        }
    }

    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        match self {
            Self::MajorAxis(iter) => iter.position(predicate),
            Self::MinorAxis(iter) => iter.position(predicate),
        }
    }

    fn rposition<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        match self {
            Self::MajorAxis(iter) => iter.rposition(predicate),
            Self::MinorAxis(iter) => iter.rposition(predicate),
        }
    }
}

impl<T> DoubleEndedIterator for IterNthVector<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::MajorAxis(iter) => iter.next_back(),
            Self::MinorAxis(iter) => iter.next_back(),
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            Self::MajorAxis(iter) => iter.nth_back(n),
            Self::MinorAxis(iter) => iter.nth_back(n),
        }
    }

    // unstable
    // fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>>  {
    //     match self {
    //         Self::Major(iter) => iter.advance_back_by(n),
    //         Self::Minor(iter) => iter.advance_back_by(n),
    //     }
    // }

    // [`Try`] is unstable
    // fn try_rfold<Acc, F, R>(&mut self, init: Acc, f: F) -> R
    // where
    //     F: FnMut(Acc, Self::Item) -> R,
    //     R: Try<Output = Acc>,
    // {
    //     match self {
    //         Self::Major(iter) => iter.try_rfold(init, f),
    //         Self::Minor(iter) => iter.try_rfold(init, f),
    //     }
    // }

    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self {
            Self::MajorAxis(iter) => iter.rfold(init, f),
            Self::MinorAxis(iter) => iter.rfold(init, f),
        }
    }
}

impl<T> ExactSizeIterator for IterNthVector<'_, T> {
    fn len(&self) -> usize {
        match self {
            Self::MajorAxis(iter) => iter.len(),
            Self::MinorAxis(iter) => iter.len(),
        }
    }

    // unstable
    // fn is_empty(&self) -> bool {
    //     match self {
    //         Self::Major(iter) => iter.is_empty(),
    //         Self::Minor(iter) => iter.is_empty(),
    //     }
    // }
}

impl<T> FusedIterator for IterNthVector<'_, T> {}
