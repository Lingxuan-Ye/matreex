use super::Matrix;
use super::layout::{ColMajor, Layout, Order, OrderKind, RowMajor};
use crate::error::{Error, Result};
use crate::index::Index;
use crate::shape::Shape;
use alloc::vec::Vec;
use core::ptr;

mod add;
mod div;
mod mul;
mod neg;
mod rem;
mod sub;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// Performs scalar operation on the matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let scalar = 2;
    /// let result = matrix.scalar_operation(&scalar, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn scalar_operation<'a, 'b, S, F, U>(
        &'a self,
        scalar: &'b S,
        mut op: F,
    ) -> Result<Matrix<U, O>>
    where
        F: FnMut(&'a T, &'b S) -> U,
    {
        let layout = self.layout.cast()?;
        let data = self
            .data
            .iter()
            .map(|element| op(element, scalar))
            .collect();
        Ok(Matrix { layout, data })
    }

    /// Performs scalar operation on the matrix, consuming `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let scalar = 2;
    /// let result = matrix.scalar_operation_consume_self(&scalar, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn scalar_operation_consume_self<'a, S, F, U>(
        self,
        scalar: &'a S,
        mut op: F,
    ) -> Result<Matrix<U, O>>
    where
        F: FnMut(T, &'a S) -> U,
    {
        let layout = self.layout.cast()?;
        let data = self
            .data
            .into_iter()
            .map(|element| op(element, scalar))
            .collect();
        Ok(Matrix { layout, data })
    }

    /// Performs scalar operation on the matrix, assigning the result
    /// to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let scalar = 2;
    /// matrix.scalar_operation_assign(&scalar, |x, y| *x += y);
    /// assert_eq!(matrix, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    pub fn scalar_operation_assign<'a, S, F>(&mut self, scalar: &'a S, mut op: F) -> &mut Self
    where
        F: FnMut(&mut T, &'a S),
    {
        self.data.iter_mut().for_each(|element| op(element, scalar));
        self
    }
}

impl<L, LO> Matrix<L, LO>
where
    LO: Order,
{
    /// Performs elementwise operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.shape() != rhs.shape()`.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_operation(&rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn elementwise_operation<'a, 'b, R, RO, F, U>(
        &'a self,
        rhs: &'b Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        RO: Order,
        F: FnMut(&'a L, &'b R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(lhs, rhs)| op(lhs, rhs))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .iter()
                .enumerate()
                .map(|(index, lhs)| {
                    let index = Index::from_flattened::<LO>(index, lhs_stride)
                        .to_flattened::<RO>(rhs_stride);
                    let rhs = unsafe { rhs.data.get_unchecked(index) };
                    op(lhs, rhs)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    /// Performs elementwise operation on two matrices, consuming `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.shape() != rhs.shape()`.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_operation_consume_self(&rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn elementwise_operation_consume_self<'a, R, RO, F, U>(
        self,
        rhs: &'a Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        RO: Order,
        F: FnMut(L, &'a R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .into_iter()
                .zip(&rhs.data)
                .map(|(lhs, rhs)| op(lhs, rhs))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, lhs)| {
                    let index = Index::from_flattened::<LO>(index, lhs_stride)
                        .to_flattened::<RO>(rhs_stride);
                    let rhs = unsafe { rhs.data.get_unchecked(index) };
                    op(lhs, rhs)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    /// Performs elementwise operation on two matrices, consuming `rhs`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.shape() != rhs.shape()`.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_operation_consume_rhs(rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn elementwise_operation_consume_rhs<'a, R, RO, F, U>(
        &'a self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        RO: Order,
        F: FnMut(&'a L, R) -> U,
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .iter()
                .zip(rhs.data)
                .map(|(lhs, rhs)| op(lhs, rhs))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            let mut rhs = rhs;
            unsafe {
                rhs.data.set_len(0);
            }
            let rhs_base = rhs.data.as_ptr();
            self.data
                .iter()
                .enumerate()
                .map(|(index, lhs)| {
                    let index = Index::from_flattened::<LO>(index, lhs_stride)
                        .to_flattened::<RO>(rhs_stride);
                    let rhs = unsafe { ptr::read(rhs_base.add(index)) };
                    op(lhs, rhs)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    /// Performs elementwise operation on two matrices, consuming both `self`
    /// and `rhs`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.shape() != rhs.shape()`.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_operation_consume_both(rhs, |x, y| x + y);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    pub fn elementwise_operation_consume_both<R, RO, F, U>(
        self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        RO: Order,
        F: FnMut(L, R) -> U,
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        let layout = self.layout.cast::<U>()?;
        let data = if LO::KIND == RO::KIND {
            self.data
                .into_iter()
                .zip(rhs.data)
                .map(|(lhs, rhs)| op(lhs, rhs))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            let mut rhs = rhs;
            unsafe {
                rhs.data.set_len(0);
            }
            let rhs_base = rhs.data.as_ptr();
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, lhs)| {
                    let index = Index::from_flattened::<LO>(index, lhs_stride)
                        .to_flattened::<RO>(rhs_stride);
                    let rhs = unsafe { ptr::read(rhs_base.add(index)) };
                    op(lhs, rhs)
                })
                .collect()
        };

        Ok(Matrix { layout, data })
    }

    /// Performs elementwise operation on two matrices, assigning the result
    /// to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.shape() != rhs.shape()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let _ = lhs.elementwise_operation_assign(&rhs, |x, y| *x += y);
    /// assert_eq!(lhs, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    pub fn elementwise_operation_assign<'a, R, RO, F>(
        &mut self,
        rhs: &'a Matrix<R, RO>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        RO: Order,
        F: FnMut(&mut L, &'a R),
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        if LO::KIND == RO::KIND {
            self.data
                .iter_mut()
                .zip(&rhs.data)
                .for_each(|(lhs, rhs)| op(lhs, rhs));
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data.iter_mut().enumerate().for_each(|(index, lhs)| {
                let index =
                    Index::from_flattened::<LO>(index, lhs_stride).to_flattened::<RO>(rhs_stride);
                let rhs = unsafe { rhs.data.get_unchecked(index) };
                op(lhs, rhs)
            });
        }

        Ok(self)
    }

    /// Performs elementwise operation on two matrices, consuming `rhs`
    /// and assigning the result to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.shape() != rhs.shape()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let _ = lhs.elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y);
    /// assert_eq!(lhs, matrix![[3, 4, 5], [6, 7, 8]]);
    /// ```
    pub fn elementwise_operation_assign_consume_rhs<R, RO, F>(
        &mut self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        RO: Order,
        F: FnMut(&mut L, R),
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        if LO::KIND == RO::KIND {
            self.data
                .iter_mut()
                .zip(rhs.data)
                .for_each(|(lhs, rhs)| op(lhs, rhs));
        } else {
            let mut rhs = rhs;
            unsafe {
                rhs.data.set_len(0);
            }
            let rhs_base = rhs.data.as_ptr();
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data.iter_mut().enumerate().for_each(|(index, lhs)| {
                let index =
                    Index::from_flattened::<LO>(index, lhs_stride).to_flattened::<RO>(rhs_stride);
                let rhs = unsafe { ptr::read(rhs_base.add(index)) };
                op(lhs, rhs)
            });
        }

        Ok(self)
    }

    /// Ensures that two matrices are conformable for elementwise operations.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.shape() != rhs.shape()`.
    fn ensure_elementwise_operation_conformable<R, RO>(&self, rhs: &Matrix<R, RO>) -> Result<&Self>
    where
        RO: Order,
    {
        if self.shape() != rhs.shape() {
            Err(Error::ShapeNotConformable)
        } else {
            Ok(self)
        }
    }
}

impl<L, LO> Matrix<L, LO>
where
    LO: Order,
{
    /// Performs multiplication-like operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.ncols() != rhs.nrows()`.
    /// - [`Error::SizeOverflow`] if the computed size of the output matrix exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The closure `op` is guaranteed to receive two non-empty, equal-length
    /// slices. It should always return a valid value derived from them.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// fn dot_product(left_row: &[i32], right_col: &[i32]) -> i32 {
    ///     left_row
    ///         .iter()
    ///         .zip(right_col)
    ///         .map(|(left, right)| left * right)
    ///         .reduce(|sum, product| sum + product)
    ///         .unwrap()
    /// }
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.multiplication_like_operation(rhs, dot_product);
    /// assert_eq!(result, Ok(matrix![[12, 12], [30, 30]]));
    /// ```
    pub fn multiplication_like_operation<R, RO, F, U>(
        self,
        rhs: Matrix<R, RO>,
        mut op: F,
    ) -> Result<Matrix<U, LO>>
    where
        RO: Order,
        F: FnMut(&[L], &[R]) -> U,
        U: Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let shape = Shape::new(nrows, ncols);
        let (layout, size) = Layout::from_shape_with_size(shape)?;
        let mut data = Vec::with_capacity(size);

        if self.ncols() == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { layout, data });
        }

        let lhs = self.with_order::<RowMajor>();
        let rhs = rhs.with_order::<ColMajor>();

        match LO::KIND {
            OrderKind::RowMajor => {
                for row in 0..nrows {
                    for col in 0..ncols {
                        let lhs = unsafe { lhs.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }

            OrderKind::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let lhs = unsafe { lhs.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { layout, data })
    }

    /// Ensures that two matrices are conformable for multiplication-like
    /// operation.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if `self.ncols() != rhs.nrows()`.
    fn ensure_multiplication_like_operation_conformable<R, RO>(
        &self,
        rhs: &Matrix<R, RO>,
    ) -> Result<&Self>
    where
        RO: Order,
    {
        if self.ncols() != rhs.nrows() {
            Err(Error::ShapeNotConformable)
        } else {
            Ok(self)
        }
    }
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    /// # Safety
    ///
    /// Calling this method when `n >= self.major()` is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline(always)]
    unsafe fn get_nth_major_axis_vector_unchecked(&self, n: usize) -> &[T] {
        let stride = self.stride();
        let lower = n * stride.major();
        let upper = lower + stride.major();
        unsafe { self.data.get_unchecked(lower..upper) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dispatch_binary, dispatch_unary, matrix};

    #[test]
    fn test_scalar_operation() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let output = matrix.scalar_operation(&scalar, |x, y| x + y).unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(output, expected);

            // Misuse but supposed to work.
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = {
                let scalar = 2;
                matrix.scalar_operation(&scalar, |x, _| x).unwrap()
            };
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            assert_eq!(output, expected);

            // Misuse but supposed to work.
            let scalar = 2;
            let output = {
                let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
                matrix.scalar_operation(&scalar, |_, y| y).unwrap()
            };
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            assert_eq!(output, expected);

            let matrix = matrix![[(); usize::MAX]; 1];
            let scalar = 2;
            let error = matrix.scalar_operation(&scalar, |_, _| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_scalar_operation_consume_self() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let output = matrix
                .scalar_operation_consume_self(&scalar, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(output, expected);

            // Misuse but supposed to work.
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let output = matrix
                .scalar_operation_consume_self(&scalar, |_, y| y)
                .unwrap();
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            assert_eq!(output, expected);

            let matrix = matrix![[(); usize::MAX]; 1];
            let scalar = 2;
            let error = matrix
                .scalar_operation_consume_self(&scalar, |_, _| 0)
                .unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }

    #[test]
    fn test_scalar_operation_assign() {
        dispatch_unary! {{
            let mut matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            matrix.scalar_operation_assign(&scalar, |x, y| *x += y);
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(matrix, expected);

            // Misuse but supposed to work.
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let mut matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            matrix.scalar_operation_assign(&scalar, |x, y| *x = y);
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            assert_eq!(matrix, expected);
        }}
    }

    #[test]
    fn test_elementwise_operation() {
        dispatch_binary! {{
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            let output = lhs.elementwise_operation(&rhs, |x, y| x + y).unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(output, expected);

            // Misuse but supposed to work.
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let output = {
                let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
                lhs.elementwise_operation(&rhs, |x, _| x).unwrap()
            };
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            assert_eq!(output, expected);

            // Misuse but supposed to work.
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            let output = {
                let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
                lhs.elementwise_operation(&rhs, |_, y| y).unwrap()
            };
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2]].with_order::<P>();
            let error = lhs.elementwise_operation(&rhs, |x, y| x + y).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]].with_order::<P>();
            let error = lhs.elementwise_operation(&rhs, |x, y| x + y).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 0]; 3].with_order::<O>();
            let rhs = matrix![[0; 0]; 3].with_order::<P>();
            let _ = lhs.elementwise_operation(&rhs, |_, _| ());

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 2]; 0].with_order::<O>();
            let rhs = matrix![[0; 2]; 0].with_order::<P>();
            let _ = lhs.elementwise_operation(&rhs, |_, _| ());
        }}
    }

    #[test]
    fn test_elementwise_operation_consume_self() {
        dispatch_binary! {{
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            let output = lhs
                .elementwise_operation_consume_self(&rhs, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(output, expected);

            // Misuse but supposed to work.
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            let output = lhs
                .elementwise_operation_consume_self(&rhs, |_, y| y)
                .unwrap();
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2]].with_order::<P>();
            let error = lhs
                .elementwise_operation_consume_self(&rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]].with_order::<P>();
            let error = lhs
                .elementwise_operation_consume_self(&rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 0]; 3].with_order::<O>();
            let rhs = matrix![[0; 0]; 3].with_order::<P>();
            let _ = lhs.elementwise_operation_consume_self(&rhs, |_, _| ());

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 2]; 0].with_order::<O>();
            let rhs = matrix![[0; 2]; 0].with_order::<P>();
            let _ = lhs.elementwise_operation_consume_self(&rhs, |_, _| ());
        }}
    }

    #[test]
    fn test_elementwise_operation_consume_rhs() {
        dispatch_binary! {{
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            let output = lhs
                .elementwise_operation_consume_rhs(rhs, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(output, expected);

            // Misuse but supposed to work.
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            let output = lhs
                .elementwise_operation_consume_rhs(rhs, |x, _| x)
                .unwrap();
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2]].with_order::<P>();
            let error = lhs
                .elementwise_operation_consume_rhs(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]].with_order::<P>();
            let error = lhs
                .elementwise_operation_consume_rhs(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 0]; 3].with_order::<O>();
            let rhs = matrix![[0; 0]; 3].with_order::<P>();
            let _ = lhs.elementwise_operation_consume_rhs(rhs, |_, _| ());

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 2]; 0].with_order::<O>();
            let rhs = matrix![[0; 2]; 0].with_order::<P>();
            let _ = lhs.elementwise_operation_consume_rhs(rhs, |_, _| ());
        }}
    }

    #[test]
    fn test_elementwise_operation_consume_both() {
        dispatch_binary! {{
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            let output = lhs
                .elementwise_operation_consume_both(rhs, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2]].with_order::<P>();
            let error = lhs
                .elementwise_operation_consume_both(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]].with_order::<P>();
            let error = lhs
                .elementwise_operation_consume_both(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 0]; 3].with_order::<O>();
            let rhs = matrix![[0; 0]; 3].with_order::<P>();
            let _ = lhs.elementwise_operation_consume_both(rhs, |_, _| ());

            // Assert no panic from unflattening indices occurs.
            let lhs = matrix![[0; 2]; 0].with_order::<O>();
            let rhs = matrix![[0; 2]; 0].with_order::<P>();
            let _ = lhs.elementwise_operation_consume_both(rhs, |_, _| ());
        }}
    }

    #[test]
    fn test_elementwise_operation_assign() {
        dispatch_binary! {{
            let mut lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            lhs.elementwise_operation_assign(&rhs, |x, y| *x += y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(lhs, expected);

            // Misuse but supposed to work.
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let mut lhs = lhs.map_ref(|x| x).unwrap();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            lhs.elementwise_operation_assign(&rhs, |x, y| *x = y)
                .unwrap();
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            assert_eq!(lhs, expected);

            let mut lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2]].with_order::<P>();
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign(&rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            assert_eq!(lhs, unchanged);

            let mut lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]].with_order::<P>();
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign(&rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            assert_eq!(lhs, unchanged);

            // Assert no panic from unflattening indices occurs.
            let mut lhs = matrix![[0; 0]; 3].with_order::<O>();
            let rhs = matrix![[0; 0]; 3].with_order::<P>();
            let _ = lhs.elementwise_operation_assign(&rhs, |_, _| ());

            // Assert no panic from unflattening indices occurs.
            let mut lhs = matrix![[0; 2]; 0].with_order::<O>();
            let rhs = matrix![[0; 2]; 0].with_order::<P>();
            let _ = lhs.elementwise_operation_assign(&rhs, |_, _| ());
        }}
    }

    #[test]
    fn test_elementwise_operation_assign_consume_rhs() {
        dispatch_binary! {{
            let mut lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2, 2], [2, 2, 2]].with_order::<P>();
            lhs.elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            assert_eq!(lhs, expected);

            let mut lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2]].with_order::<P>();
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            assert_eq!(lhs, unchanged);

            let mut lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[2, 2], [2, 2], [2, 2]].with_order::<P>();
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            assert_eq!(lhs, unchanged);

            // Assert no panic from unflattening indices occurs.
            let mut lhs = matrix![[0; 0]; 3].with_order::<O>();
            let rhs = matrix![[0; 0]; 3].with_order::<P>();
            let _ = lhs.elementwise_operation_assign_consume_rhs(rhs, |_, _| ());

            // Assert no panic from unflattening indices occurs.
            let mut lhs = matrix![[0; 2]; 0].with_order::<O>();
            let rhs = matrix![[0; 2]; 0].with_order::<P>();
            let _ = lhs.elementwise_operation_assign_consume_rhs(rhs, |_, _| ());
        }}
    }

    #[test]
    fn test_multiplication_like_operation() {
        fn dot_product(lhs_row: &[i32], rhs_col: &[i32]) -> i32 {
            lhs_row
                .iter()
                .zip(rhs_col)
                .map(|(lhs, rhs)| lhs * rhs)
                .reduce(|sum, product| sum + product)
                .unwrap()
        }

        dispatch_binary! {{
            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2], [3, 4], [5, 6]].with_order::<P>();
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[22, 28], [49, 64]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1], [2], [3]].with_order::<P>();
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[14], [32]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]].with_order::<P>();
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[30, 36, 42], [66, 81, 96]];
            assert_eq!(output, expected);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2], [3, 4]].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let rhs = matrix![[1, 2, 3], [4, 5, 6]].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);

            let lhs = matrix![[0; 0]; 2].with_order::<O>();
            let rhs = matrix![[0; usize::MAX]; 0].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, |_, _| 0)
                .unwrap_err();
            assert_eq!(error, Error::SizeOverflow);

            let lhs = matrix![[0; 0]; 1].with_order::<O>();
            let rhs = matrix![[0; usize::MAX]; 0].with_order::<P>();
            let error = lhs
                .multiplication_like_operation(rhs, |_, _| 0)
                .unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }}
    }
}
