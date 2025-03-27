//! Defines iterating operations.

use self::iter_mut::IterVectorsMut;
use crate::Matrix;
use crate::error::{Error, Result};
use crate::index::Index;
use crate::order::Order;
use std::iter::{Skip, StepBy, Take};
use std::slice::{Iter, IterMut};

mod iter_mut;

/// An iterator that knows its exact length and can yield elements
/// from both ends.
pub trait ExactSizeDoubleEndedIterator: ExactSizeIterator + DoubleEndedIterator {}

impl<I> ExactSizeDoubleEndedIterator for I where I: ExactSizeIterator + DoubleEndedIterator {}

impl<T> Matrix<T> {
    /// Returns an iterator over the rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let mut rows = matrix.iter_rows();
    ///
    /// let mut row_0 = rows.next().unwrap();
    /// assert_eq!(row_0.next(), Some(&1));
    /// assert_eq!(row_0.next(), Some(&2));
    /// assert_eq!(row_0.next(), Some(&3));
    /// assert_eq!(row_0.next(), None);
    ///
    /// let mut row_1 = rows.next().unwrap();
    /// assert_eq!(row_1.next(), Some(&4));
    /// assert_eq!(row_1.next(), Some(&5));
    /// assert_eq!(row_1.next(), Some(&6));
    /// assert_eq!(row_1.next(), None);
    ///
    /// assert!(rows.next().is_none());
    /// ```
    pub fn iter_rows(
        &self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &T>>
    {
        (0..self.nrows()).map(|n| match self.order {
            Order::RowMajor => self.iter_nth_major_axis_vector_unchecked(n),
            Order::ColMajor => self.iter_nth_minor_axis_vector_unchecked(n),
        })
    }

    /// Returns an iterator over the columns of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let mut cols = matrix.iter_cols();
    ///
    /// let mut col_0 = cols.next().unwrap();
    /// assert_eq!(col_0.next(), Some(&1));
    /// assert_eq!(col_0.next(), Some(&4));
    /// assert_eq!(col_0.next(), None);
    ///
    /// let mut col_1 = cols.next().unwrap();
    /// assert_eq!(col_1.next(), Some(&2));
    /// assert_eq!(col_1.next(), Some(&5));
    /// assert_eq!(col_1.next(), None);
    ///
    /// let mut col_2 = cols.next().unwrap();
    /// assert_eq!(col_2.next(), Some(&3));
    /// assert_eq!(col_2.next(), Some(&6));
    /// assert_eq!(col_2.next(), None);
    ///
    /// assert!(cols.next().is_none());
    /// ```
    pub fn iter_cols(
        &self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &T>>
    {
        (0..self.ncols()).map(|n| match self.order {
            Order::RowMajor => self.iter_nth_minor_axis_vector_unchecked(n),
            Order::ColMajor => self.iter_nth_major_axis_vector_unchecked(n),
        })
    }

    /// Returns an iterator that allows modifying each rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// for row in matrix.iter_rows_mut() {
    ///     for element in row {
    ///         *element += 2;
    ///     }
    /// }
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    pub fn iter_rows_mut(
        &mut self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &mut T>>
    {
        match self.order {
            Order::RowMajor => IterVectorsMut::on_major_axis(self),
            Order::ColMajor => IterVectorsMut::on_minor_axis(self),
        }
    }

    /// Returns an iterator that allows modifying each columns of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// for col in matrix.iter_cols_mut() {
    ///     for element in col {
    ///         *element += 2;
    ///     }
    /// }
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    pub fn iter_cols_mut(
        &mut self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = impl ExactSizeDoubleEndedIterator<Item = &mut T>>
    {
        match self.order {
            Order::RowMajor => IterVectorsMut::on_minor_axis(self),
            Order::ColMajor => IterVectorsMut::on_major_axis(self),
        }
    }

    /// Returns an iterator over the elements of the nth row in the matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if `n` is greater than or equal to
    ///   the number of rows in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let mut row_1 = matrix.iter_nth_row(1)?;
    /// assert_eq!(row_1.next(), Some(&4));
    /// assert_eq!(row_1.next(), Some(&5));
    /// assert_eq!(row_1.next(), Some(&6));
    /// assert_eq!(row_1.next(), None);
    /// # Ok(())
    /// # }
    /// ```
    pub fn iter_nth_row(&self, n: usize) -> Result<impl ExactSizeDoubleEndedIterator<Item = &T>> {
        match self.order {
            Order::RowMajor => self.iter_nth_major_axis_vector(n),
            Order::ColMajor => self.iter_nth_minor_axis_vector(n),
        }
    }

    /// Returns an iterator over the elements of the nth column in the matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if `n` is greater than or equal to
    ///   the number of columns in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let mut col_1 = matrix.iter_nth_col(1)?;
    /// assert_eq!(col_1.next(), Some(&2));
    /// assert_eq!(col_1.next(), Some(&5));
    /// assert_eq!(col_1.next(), None);
    /// # Ok(())
    /// # }
    /// ```
    pub fn iter_nth_col(&self, n: usize) -> Result<impl ExactSizeDoubleEndedIterator<Item = &T>> {
        match self.order {
            Order::RowMajor => self.iter_nth_minor_axis_vector(n),
            Order::ColMajor => self.iter_nth_major_axis_vector(n),
        }
    }

    /// Returns an iterator that allows modifying each element of the
    /// nth row in the matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if `n` is greater than or equal to
    ///   the number of rows in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// for element in matrix.iter_nth_row_mut(1)? {
    ///    *element += 2;
    /// }
    /// assert_eq!(matrix, matrix![[1, 2, 3], [6, 7, 8]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn iter_nth_row_mut(
        &mut self,
        n: usize,
    ) -> Result<impl ExactSizeDoubleEndedIterator<Item = &mut T>> {
        match self.order {
            Order::RowMajor => self.iter_nth_major_axis_vector_mut(n),
            Order::ColMajor => self.iter_nth_minor_axis_vector_mut(n),
        }
    }

    /// Returns an iterator that allows modifying each element of the
    /// nth column in the matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexOutOfBounds`] if `n` is greater than or equal to
    ///   the number of columns in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// for element in matrix.iter_nth_col_mut(1)? {
    ///    *element += 2;
    /// }
    /// assert_eq!(matrix, matrix![[1, 4, 3], [4, 7, 6]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn iter_nth_col_mut(
        &mut self,
        n: usize,
    ) -> Result<impl ExactSizeDoubleEndedIterator<Item = &mut T>> {
        match self.order {
            Order::RowMajor => self.iter_nth_minor_axis_vector_mut(n),
            Order::ColMajor => self.iter_nth_major_axis_vector_mut(n),
        }
    }

    /// Returns an iterator over the elements of the matrix.
    ///
    /// # Notes
    ///
    /// Elements are iterated in memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let sum = matrix.iter_elements().sum::<i32>();
    /// assert_eq!(sum, 21);
    /// ```
    #[inline]
    pub fn iter_elements(&self) -> impl ExactSizeDoubleEndedIterator<Item = &T> {
        self.data.iter()
    }

    /// Returns an iterator that allows modifying each element
    /// of the matrix.
    ///
    /// # Notes
    ///
    /// Elements are iterated in memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix.iter_elements_mut().for_each(|element| {
    ///     *element += 2;
    /// });
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    #[inline]
    pub fn iter_elements_mut(&mut self) -> impl ExactSizeDoubleEndedIterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Creates a consuming iterator, that is, one that moves each
    /// element out of the matrix.
    ///
    /// # Notes
    ///
    /// Elements are iterated in memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let sum = matrix.into_iter_elements().sum::<i32>();
    /// assert_eq!(sum, 21);
    /// ```
    #[inline]
    pub fn into_iter_elements(self) -> impl ExactSizeDoubleEndedIterator<Item = T> {
        self.data.into_iter()
    }

    /// Returns an iterator over the elements of the matrix along with
    /// their indices.
    ///
    /// # Notes
    ///
    /// Elements are iterated in memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .iter_elements_with_index()
    ///     .for_each(|(index, element)| {
    ///         assert_eq!(element, &matrix[index]);
    ///     });
    /// ```
    pub fn iter_elements_with_index(
        &self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = (Index, &T)> {
        self.data.iter().enumerate().map(|(index, element)| {
            // hope loop-invariant code motion applies here,
            // as well as to similar code
            let index = Index::from_flattened(index, self.order, self.shape);
            (index, element)
        })
    }

    /// Returns an iterator that allows modifying each element
    /// of the matrix along with its index.
    ///
    /// # Notes
    ///
    /// Elements are iterated in memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .iter_elements_mut_with_index()
    ///     .for_each(|(index, element)| {
    ///         *element += index.row as i32 + index.col as i32;
    ///     });
    /// assert_eq!(matrix, matrix![[1, 3, 5], [5, 7, 9]]);
    /// ```
    pub fn iter_elements_mut_with_index(
        &mut self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = (Index, &mut T)> {
        self.data.iter_mut().enumerate().map(|(index, element)| {
            let index = Index::from_flattened(index, self.order, self.shape);
            (index, element)
        })
    }

    /// Creates a consuming iterator, that is, one that moves each
    /// element out of the matrix along with its index.
    ///
    /// # Notes
    ///
    /// Elements are iterated in memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// matrix
    ///     .clone()
    ///     .into_iter_elements_with_index()
    ///     .for_each(|(index, element)| {
    ///         assert_eq!(element, matrix[index]);
    ///     });
    /// ```
    pub fn into_iter_elements_with_index(
        self,
    ) -> impl ExactSizeDoubleEndedIterator<Item = (Index, T)> {
        self.data
            .into_iter()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened(index, self.order, self.shape);
                (index, element)
            })
    }
}

impl<T> Matrix<T> {
    pub(crate) fn iter_nth_major_axis_vector(
        &self,
        n: usize,
    ) -> Result<Take<StepBy<Skip<Iter<'_, T>>>>> {
        if n >= self.major() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_major_axis_vector_unchecked(n))
        }
    }

    pub(crate) fn iter_nth_minor_axis_vector(
        &self,
        n: usize,
    ) -> Result<Take<StepBy<Skip<Iter<'_, T>>>>> {
        if n >= self.minor() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_minor_axis_vector_unchecked(n))
        }
    }

    pub(crate) fn iter_nth_major_axis_vector_mut(
        &mut self,
        n: usize,
    ) -> Result<Take<StepBy<Skip<IterMut<'_, T>>>>> {
        if n >= self.major() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_major_axis_vector_unchecked_mut(n))
        }
    }

    pub(crate) fn iter_nth_minor_axis_vector_mut(
        &mut self,
        n: usize,
    ) -> Result<Take<StepBy<Skip<IterMut<'_, T>>>>> {
        if n >= self.minor() {
            Err(Error::IndexOutOfBounds)
        } else {
            Ok(self.iter_nth_minor_axis_vector_unchecked_mut(n))
        }
    }

    pub(crate) fn iter_nth_major_axis_vector_unchecked(
        &self,
        n: usize,
    ) -> Take<StepBy<Skip<Iter<'_, T>>>> {
        let skip = n * self.major_stride();
        let step = 1;
        let take = self.minor();
        self.data.iter().skip(skip).step_by(step).take(take)
    }

    pub(crate) fn iter_nth_minor_axis_vector_unchecked(
        &self,
        n: usize,
    ) -> Take<StepBy<Skip<Iter<'_, T>>>> {
        let skip = n;
        let step = self.major_stride();
        let take = self.major();
        self.data.iter().skip(skip).step_by(step).take(take)
    }

    pub(crate) fn iter_nth_major_axis_vector_unchecked_mut(
        &mut self,
        n: usize,
    ) -> Take<StepBy<Skip<IterMut<'_, T>>>> {
        let skip = n * self.major_stride();
        let step = 1;
        let take = self.minor();
        self.data.iter_mut().skip(skip).step_by(step).take(take)
    }

    pub(crate) fn iter_nth_minor_axis_vector_unchecked_mut(
        &mut self,
        n: usize,
    ) -> Take<StepBy<Skip<IterMut<'_, T>>>> {
        let skip = n;
        let step = self.major_stride();
        let take = self.major();
        self.data.iter_mut().skip(skip).step_by(step).take(take)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_iter_rows() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut rows = matrix.iter_rows();
            let mut row_0 = rows.next().unwrap();
            assert_eq!(row_0.next(), Some(&1));
            assert_eq!(row_0.next(), Some(&2));
            assert_eq!(row_0.next(), Some(&3));
            assert_eq!(row_0.next(), None);
            let mut row_1 = rows.next().unwrap();
            assert_eq!(row_1.next(), Some(&4));
            assert_eq!(row_1.next(), Some(&5));
            assert_eq!(row_1.next(), Some(&6));
            assert_eq!(row_1.next(), None);
            assert!(rows.next().is_none());

            let mut rows = matrix.iter_rows();
            let mut row_1 = rows.next_back().unwrap();
            assert_eq!(row_1.next(), Some(&4));
            assert_eq!(row_1.next(), Some(&5));
            assert_eq!(row_1.next(), Some(&6));
            assert_eq!(row_1.next(), None);
            let mut row_0 = rows.next().unwrap();
            assert_eq!(row_0.next(), Some(&1));
            assert_eq!(row_0.next(), Some(&2));
            assert_eq!(row_0.next(), Some(&3));
            assert_eq!(row_0.next(), None);
            assert!(rows.next_back().is_none());
            assert!(rows.next().is_none());
        }

        matrix.switch_order();

        {
            let mut rows = matrix.iter_rows();
            let mut row_0 = rows.next().unwrap();
            assert_eq!(row_0.next(), Some(&1));
            assert_eq!(row_0.next(), Some(&2));
            assert_eq!(row_0.next(), Some(&3));
            assert_eq!(row_0.next(), None);
            let mut row_1 = rows.next().unwrap();
            assert_eq!(row_1.next(), Some(&4));
            assert_eq!(row_1.next(), Some(&5));
            assert_eq!(row_1.next(), Some(&6));
            assert_eq!(row_1.next(), None);
            assert!(rows.next().is_none());

            let mut rows = matrix.iter_rows();
            let mut row_1 = rows.next_back().unwrap();
            assert_eq!(row_1.next(), Some(&4));
            assert_eq!(row_1.next(), Some(&5));
            assert_eq!(row_1.next(), Some(&6));
            assert_eq!(row_1.next(), None);
            let mut row_0 = rows.next().unwrap();
            assert_eq!(row_0.next(), Some(&1));
            assert_eq!(row_0.next(), Some(&2));
            assert_eq!(row_0.next(), Some(&3));
            assert_eq!(row_0.next(), None);
            assert!(rows.next_back().is_none());
            assert!(rows.next().is_none());
        }
    }

    #[test]
    fn test_iter_cols() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut cols = matrix.iter_cols();
            let mut col_0 = cols.next().unwrap();
            assert_eq!(col_0.next(), Some(&1));
            assert_eq!(col_0.next(), Some(&4));
            assert_eq!(col_0.next(), None);
            let mut col_1 = cols.next().unwrap();
            assert_eq!(col_1.next(), Some(&2));
            assert_eq!(col_1.next(), Some(&5));
            assert_eq!(col_1.next(), None);
            let mut col_2 = cols.next().unwrap();
            assert_eq!(col_2.next(), Some(&3));
            assert_eq!(col_2.next(), Some(&6));
            assert_eq!(col_2.next(), None);
            assert!(cols.next().is_none());

            let mut cols = matrix.iter_cols();
            let mut col_2 = cols.next_back().unwrap();
            assert_eq!(col_2.next(), Some(&3));
            assert_eq!(col_2.next(), Some(&6));
            assert_eq!(col_2.next(), None);
            let mut col_0 = cols.next().unwrap();
            assert_eq!(col_0.next(), Some(&1));
            assert_eq!(col_0.next(), Some(&4));
            assert_eq!(col_0.next(), None);
            let mut col_1 = cols.next_back().unwrap();
            assert_eq!(col_1.next(), Some(&2));
            assert_eq!(col_1.next(), Some(&5));
            assert_eq!(col_1.next(), None);
            assert!(cols.next_back().is_none());
            assert!(cols.next().is_none());
        }

        matrix.switch_order();

        {
            let mut cols = matrix.iter_cols();
            let mut col_0 = cols.next().unwrap();
            assert_eq!(col_0.next(), Some(&1));
            assert_eq!(col_0.next(), Some(&4));
            assert_eq!(col_0.next(), None);
            let mut col_1 = cols.next().unwrap();
            assert_eq!(col_1.next(), Some(&2));
            assert_eq!(col_1.next(), Some(&5));
            assert_eq!(col_1.next(), None);
            let mut col_2 = cols.next().unwrap();
            assert_eq!(col_2.next(), Some(&3));
            assert_eq!(col_2.next(), Some(&6));
            assert_eq!(col_2.next(), None);
            assert!(cols.next().is_none());

            let mut cols = matrix.iter_cols();
            let mut col_2 = cols.next_back().unwrap();
            assert_eq!(col_2.next(), Some(&3));
            assert_eq!(col_2.next(), Some(&6));
            assert_eq!(col_2.next(), None);
            let mut col_0 = cols.next().unwrap();
            assert_eq!(col_0.next(), Some(&1));
            assert_eq!(col_0.next(), Some(&4));
            assert_eq!(col_0.next(), None);
            let mut col_1 = cols.next_back().unwrap();
            assert_eq!(col_1.next(), Some(&2));
            assert_eq!(col_1.next(), Some(&5));
            assert_eq!(col_1.next(), None);
            assert!(cols.next_back().is_none());
            assert!(cols.next().is_none());
        }
    }

    #[test]
    fn test_iter_rows_mut() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut matrix = matrix.clone();
            let mut count = 0;

            for row in matrix.iter_rows_mut() {
                for element in row {
                    count += 1;
                    *element += count;
                }
            }
            assert_eq!(matrix, matrix![[2, 4, 6], [8, 10, 12]]);

            for row in matrix.iter_rows_mut().rev() {
                for element in row.rev() {
                    *element -= count;
                    count -= 1;
                }
            }
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
        }

        matrix.switch_order();

        {
            let mut matrix = matrix.clone();
            let mut count = 0;

            for row in matrix.iter_rows_mut() {
                for element in row {
                    count += 1;
                    *element += count;
                }
            }
            matrix.switch_order();
            assert_eq!(matrix, matrix![[2, 4, 6], [8, 10, 12]]);

            matrix.switch_order();
            for row in matrix.iter_rows_mut().rev() {
                for element in row.rev() {
                    *element -= count;
                    count -= 1;
                }
            }
            matrix.switch_order();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
        }
    }

    #[test]
    fn test_iter_cols_mut() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut matrix = matrix.clone();
            let mut count = 0;

            for col in matrix.iter_cols_mut() {
                for element in col {
                    count += 1;
                    *element += count;
                }
            }
            assert_eq!(matrix, matrix![[2, 5, 8], [6, 9, 12]]);

            for col in matrix.iter_cols_mut().rev() {
                for element in col.rev() {
                    *element -= count;
                    count -= 1;
                }
            }
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
        }

        matrix.switch_order();

        {
            let mut matrix = matrix.clone();
            let mut count = 0;

            for col in matrix.iter_cols_mut() {
                for element in col {
                    count += 1;
                    *element += count;
                }
            }
            matrix.switch_order();
            assert_eq!(matrix, matrix![[2, 5, 8], [6, 9, 12]]);

            matrix.switch_order();
            for col in matrix.iter_cols_mut().rev() {
                for element in col.rev() {
                    *element -= count;
                    count -= 1;
                }
            }
            matrix.switch_order();
            assert_eq!(matrix, matrix![[1, 2, 3], [4, 5, 6]]);
        }
    }

    #[test]
    fn test_iter_nth_row() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut row_0 = matrix.iter_nth_row(0).unwrap();
            assert_eq!(row_0.next(), Some(&1));
            assert_eq!(row_0.next(), Some(&2));
            assert_eq!(row_0.next(), Some(&3));
            assert_eq!(row_0.next(), None);

            let mut row_1 = matrix.iter_nth_row(1).unwrap();
            assert_eq!(row_1.next(), Some(&4));
            assert_eq!(row_1.next(), Some(&5));
            assert_eq!(row_1.next(), Some(&6));
            assert_eq!(row_1.next(), None);

            assert!(matches!(
                matrix.iter_nth_row(2),
                Err(Error::IndexOutOfBounds)
            ));
        }

        matrix.switch_order();

        {
            let mut row_0 = matrix.iter_nth_row(0).unwrap();
            assert_eq!(row_0.next(), Some(&1));
            assert_eq!(row_0.next(), Some(&2));
            assert_eq!(row_0.next(), Some(&3));
            assert_eq!(row_0.next(), None);

            let mut row_1 = matrix.iter_nth_row(1).unwrap();
            assert_eq!(row_1.next(), Some(&4));
            assert_eq!(row_1.next(), Some(&5));
            assert_eq!(row_1.next(), Some(&6));
            assert_eq!(row_1.next(), None);

            assert!(matches!(
                matrix.iter_nth_row(2),
                Err(Error::IndexOutOfBounds)
            ));
        }
    }

    #[test]
    fn test_iter_nth_col() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut col_0 = matrix.iter_nth_col(0).unwrap();
            assert_eq!(col_0.next(), Some(&1));
            assert_eq!(col_0.next(), Some(&4));
            assert_eq!(col_0.next(), None);

            let mut col_1 = matrix.iter_nth_col(1).unwrap();
            assert_eq!(col_1.next(), Some(&2));
            assert_eq!(col_1.next(), Some(&5));
            assert_eq!(col_1.next(), None);

            let mut col_2 = matrix.iter_nth_col(2).unwrap();
            assert_eq!(col_2.next(), Some(&3));
            assert_eq!(col_2.next(), Some(&6));
            assert_eq!(col_2.next(), None);

            assert!(matches!(
                matrix.iter_nth_col(3),
                Err(Error::IndexOutOfBounds)
            ));
        }

        matrix.switch_order();

        {
            let mut col_0 = matrix.iter_nth_col(0).unwrap();
            assert_eq!(col_0.next(), Some(&1));
            assert_eq!(col_0.next(), Some(&4));
            assert_eq!(col_0.next(), None);

            let mut col_1 = matrix.iter_nth_col(1).unwrap();
            assert_eq!(col_1.next(), Some(&2));
            assert_eq!(col_1.next(), Some(&5));
            assert_eq!(col_1.next(), None);

            let mut col_2 = matrix.iter_nth_col(2).unwrap();
            assert_eq!(col_2.next(), Some(&3));
            assert_eq!(col_2.next(), Some(&6));
            assert_eq!(col_2.next(), None);

            assert!(matches!(
                matrix.iter_nth_col(3),
                Err(Error::IndexOutOfBounds)
            ));
        }
    }

    #[test]
    fn test_iter_nth_row_mut() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut matrix = matrix.clone();

            let row_0 = matrix.iter_nth_row_mut(0).unwrap();
            for element in row_0 {
                *element += 2;
            }

            let row_1 = matrix.iter_nth_row_mut(1).unwrap();
            for element in row_1.rev() {
                *element -= 2;
            }

            assert!(matches!(
                matrix.iter_nth_row_mut(2),
                Err(Error::IndexOutOfBounds)
            ));

            assert_eq!(matrix, matrix![[3, 4, 5], [2, 3, 4]]);
        }

        matrix.switch_order();

        {
            let mut matrix = matrix.clone();

            let row_0 = matrix.iter_nth_row_mut(0).unwrap();
            for element in row_0 {
                *element += 2;
            }

            let row_1 = matrix.iter_nth_row_mut(1).unwrap();
            for element in row_1.rev() {
                *element -= 2;
            }

            assert!(matches!(
                matrix.iter_nth_row_mut(2),
                Err(Error::IndexOutOfBounds)
            ));

            matrix.switch_order();
            assert_eq!(matrix, matrix![[3, 4, 5], [2, 3, 4]]);
        }
    }

    #[test]
    fn test_iter_nth_col_mut() {
        let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];

        {
            let mut matrix = matrix.clone();

            let col_0 = matrix.iter_nth_col_mut(0).unwrap();
            for element in col_0 {
                *element += 2;
            }

            let col_1 = matrix.iter_nth_col_mut(1).unwrap();
            for element in col_1.rev() {
                *element -= 2;
            }

            let col_2 = matrix.iter_nth_col_mut(2).unwrap();
            for element in col_2 {
                *element *= 2;
            }

            assert!(matches!(
                matrix.iter_nth_col_mut(3),
                Err(Error::IndexOutOfBounds)
            ));

            assert_eq!(matrix, matrix![[3, 0, 6], [6, 3, 12]]);
        }

        matrix.switch_order();

        {
            let mut matrix = matrix.clone();

            let col_0 = matrix.iter_nth_col_mut(0).unwrap();
            for element in col_0 {
                *element += 2;
            }

            let col_1 = matrix.iter_nth_col_mut(1).unwrap();
            for element in col_1.rev() {
                *element -= 2;
            }

            let col_2 = matrix.iter_nth_col_mut(2).unwrap();
            for element in col_2 {
                *element *= 2;
            }

            assert!(matches!(
                matrix.iter_nth_col_mut(3),
                Err(Error::IndexOutOfBounds)
            ));

            matrix.switch_order();
            assert_eq!(matrix, matrix![[3, 0, 6], [6, 3, 12]]);
        }
    }

    #[test]
    fn test_iter_elements() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = 21;

        // default order
        {
            let sum = matrix.iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            let sum = matrix.iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }
    }

    #[test]
    fn test_iter_elements_mut() {
        fn add_two(x: &mut i32) {
            *x += 2;
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        // default order
        {
            let mut matrix = matrix.clone();

            matrix.iter_elements_mut().for_each(add_two);
            assert_eq!(matrix, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.iter_elements_mut().for_each(add_two);
            matrix.switch_order();
            assert_eq!(matrix, expected);
        }
    }

    #[test]
    fn test_into_iter_elements() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = 21;

        // default order
        {
            let matrix = matrix.clone();

            let sum = matrix.into_iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            let sum = matrix.into_iter_elements().sum::<i32>();
            assert_eq!(sum, expected);
        }
    }

    #[test]
    fn test_iter_elements_with_index() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let test_index = |(index, element)| {
            assert_eq!(element, &matrix[index]);
        };

        // default order
        {
            matrix.iter_elements_with_index().for_each(test_index);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.iter_elements_with_index().for_each(test_index);
        }
    }

    #[test]
    fn test_iter_elements_mut_with_index() {
        fn add_index((index, element): (Index, &mut i32)) {
            *element += index.row as i32 + index.col as i32;
        }

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[1, 3, 5], [5, 7, 9]];

        // default order
        {
            let mut matrix = matrix.clone();

            matrix.iter_elements_mut_with_index().for_each(add_index);
            assert_eq!(matrix, expected);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.iter_elements_mut_with_index().for_each(add_index);
            matrix.switch_order();
            assert_eq!(matrix, expected);
        }
    }

    #[test]
    fn test_into_iter_elements_with_index() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let test_index = |(index, element)| {
            assert_eq!(element, matrix[index]);
        };

        // default order
        {
            let matrix = matrix.clone();

            matrix.into_iter_elements_with_index().for_each(test_index);
        }

        // alternative order
        {
            let mut matrix = matrix.clone();
            matrix.switch_order();

            matrix.into_iter_elements_with_index().for_each(test_index);
        }
    }
}
