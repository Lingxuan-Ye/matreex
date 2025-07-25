use crate::Matrix;
use crate::error::{Error, Result};
use crate::index::MemoryIndex;
use crate::order::Order;
use crate::shape::{MemoryShape, Shape};
use alloc::vec::Vec;
use core::ptr;

mod add;
mod div;
mod mul;
mod neg;
mod rem;
mod sub;

impl<L> Matrix<L> {
    /// Returns `true` if the matrix is a square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let matrix =matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// assert!(matrix.is_square());
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert!(!matrix.is_square());
    /// ```
    #[inline]
    pub fn is_square(&self) -> bool {
        self.major() == self.minor()
    }

    /// Returns `true` if two matrices are conformable for elementwise
    /// operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// assert!(lhs.is_elementwise_operation_conformable(&rhs));
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// assert!(!lhs.is_elementwise_operation_conformable(&rhs));
    /// ```
    #[inline]
    pub fn is_elementwise_operation_conformable<R>(&self, rhs: &Matrix<R>) -> bool {
        if self.order == rhs.order {
            self.shape == rhs.shape
        } else {
            self.major() == rhs.minor() && self.minor() == rhs.major()
        }
    }

    /// Returns `true` if two matrices are conformable for multiplication-like
    /// operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// assert!(!lhs.is_multiplication_like_operation_conformable(&rhs));
    /// ```
    #[inline]
    pub fn is_multiplication_like_operation_conformable<R>(&self, rhs: &Matrix<R>) -> bool {
        self.ncols() == rhs.nrows()
    }
}

impl<L> Matrix<L> {
    /// Ensures that the matrix is a square matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::SquareMatrixRequired`] if the matrix is not a square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let matrix =matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let result = matrix.ensure_square();
    /// assert!(result.is_ok());
    ///
    /// let matrix = matrix![[1, 2, 3], [4, 5, 6]];
    /// let result = matrix.ensure_square();
    /// assert_eq!(result, Err(Error::SquareMatrixRequired));
    /// ```
    #[inline]
    pub fn ensure_square(&self) -> Result<&Self> {
        if self.is_square() {
            Ok(self)
        } else {
            Err(Error::SquareMatrixRequired)
        }
    }

    /// Ensures that two matrices are conformable for elementwise operations.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.ensure_elementwise_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::ShapeNotConformable));
    /// ```
    #[inline]
    pub fn ensure_elementwise_operation_conformable<R>(&self, rhs: &Matrix<R>) -> Result<&Self> {
        if self.is_elementwise_operation_conformable(rhs) {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }

    /// Ensures that two matrices are conformable for multiplication-like
    /// operation.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::{Error, matrix};
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    ///
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert!(result.is_ok());
    ///
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
    /// assert_eq!(result, Err(Error::ShapeNotConformable));
    /// ```
    #[inline]
    pub fn ensure_multiplication_like_operation_conformable<R>(
        &self,
        rhs: &Matrix<R>,
    ) -> Result<&Self> {
        if self.is_multiplication_like_operation_conformable(rhs) {
            Ok(self)
        } else {
            Err(Error::ShapeNotConformable)
        }
    }
}

impl<L> Matrix<L> {
    /// Performs elementwise operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
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
    pub fn elementwise_operation<'a, 'b, R, F, U>(
        &'a self,
        rhs: &'b Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(&'a L, &'b R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        shape.size::<U>()?;
        let data = if self.order == rhs.order {
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = MemoryIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, consuming `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
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
    pub fn elementwise_operation_consume_self<'a, R, F, U>(
        self,
        rhs: &'a Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(L, &'a R) -> U,
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        let order = self.order;
        let shape = self.shape;
        shape.size::<U>()?;
        let data = if self.order == rhs.order {
            self.data
                .into_iter()
                .zip(&rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = MemoryIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { rhs.data.get_unchecked(index) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, consuming `rhs`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
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
    pub fn elementwise_operation_consume_rhs<'a, R, F, U>(
        &'a self,
        rhs: Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(&'a L, R) -> U,
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        let order = self.order;
        let shape = self.shape;
        shape.size::<U>()?;
        let data = if self.order == rhs.order {
            self.data
                .iter()
                .zip(rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let mut rhs = rhs;
            unsafe {
                rhs.data.set_len(0);
            }
            let rhs_base = rhs.data.as_ptr();
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = MemoryIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { ptr::read(rhs_base.add(index)) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, consuming both `self`
    /// and `rhs`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
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
    pub fn elementwise_operation_consume_both<R, F, U>(
        self,
        rhs: Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(L, R) -> U,
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        let order = self.order;
        let shape = self.shape;
        shape.size::<U>()?;
        let data = if self.order == rhs.order {
            self.data
                .into_iter()
                .zip(rhs.data)
                .map(|(left, right)| op(left, right))
                .collect()
        } else {
            let mut rhs = rhs;
            unsafe {
                rhs.data.set_len(0);
            }
            let rhs_base = rhs.data.as_ptr();
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data
                .into_iter()
                .enumerate()
                .map(|(index, left)| {
                    let index = MemoryIndex::from_flattened(index, lhs_stride)
                        .swap()
                        .to_flattened(rhs_stride);
                    let right = unsafe { ptr::read(rhs_base.add(index)) };
                    op(left, right)
                })
                .collect()
        };

        Ok(Matrix { order, shape, data })
    }

    /// Performs elementwise operation on two matrices, assigning the result
    /// to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// lhs.elementwise_operation_assign(&rhs, |x, y| *x += y)?;
    /// assert_eq!(lhs, matrix![[3, 4, 5], [6, 7, 8]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn elementwise_operation_assign<'a, R, F>(
        &mut self,
        rhs: &'a Matrix<R>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        F: FnMut(&mut L, &'a R),
    {
        self.ensure_elementwise_operation_conformable(rhs)?;

        if self.order == rhs.order {
            self.data
                .iter_mut()
                .zip(&rhs.data)
                .for_each(|(left, right)| op(left, right));
        } else {
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data.iter_mut().enumerate().for_each(|(index, left)| {
                let index = MemoryIndex::from_flattened(index, lhs_stride)
                    .swap()
                    .to_flattened(rhs_stride);
                let right = unsafe { rhs.data.get_unchecked(index) };
                op(left, right)
            });
        }

        Ok(self)
    }

    /// Performs elementwise operation on two matrices, consuming `rhs`
    /// and assigning the result to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// lhs.elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y)?;
    /// assert_eq!(lhs, matrix![[3, 4, 5], [6, 7, 8]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn elementwise_operation_assign_consume_rhs<R, F>(
        &mut self,
        rhs: Matrix<R>,
        mut op: F,
    ) -> Result<&mut Self>
    where
        F: FnMut(&mut L, R),
    {
        self.ensure_elementwise_operation_conformable(&rhs)?;

        if self.order == rhs.order {
            self.data
                .iter_mut()
                .zip(rhs.data)
                .for_each(|(left, right)| op(left, right));
        } else {
            let mut rhs = rhs;
            unsafe {
                rhs.data.set_len(0);
            }
            let rhs_base = rhs.data.as_ptr();
            let lhs_stride = self.stride();
            let rhs_stride = rhs.stride();
            self.data.iter_mut().enumerate().for_each(|(index, left)| {
                let index = MemoryIndex::from_flattened(index, lhs_stride)
                    .swap()
                    .to_flattened(rhs_stride);
                let right = unsafe { ptr::read(rhs_base.add(index)) };
                op(left, right)
            });
        }

        Ok(self)
    }
}

impl<L> Matrix<L> {
    /// Performs multiplication-like operation on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    /// - [`Error::SizeOverflow`] if the computed size of the output matrix exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if the required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The closure `op` is guaranteed to receive two non-empty, equal-length
    /// slices. It should always return a valid value derived from them.
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// fn dot_product(lhs: &[i32], rhs: &[i32]) -> i32 {
    ///     lhs.iter()
    ///         .zip(rhs)
    ///         .map(|(x, y)| x * y)
    ///         .reduce(|acc, p| acc + p)
    ///         .unwrap()
    /// }
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.multiplication_like_operation(rhs, dot_product);
    /// assert_eq!(result, Ok(matrix![[12, 12], [30, 30]]));
    /// ```
    pub fn multiplication_like_operation<R, F, U>(
        mut self,
        mut rhs: Matrix<R>,
        mut op: F,
    ) -> Result<Matrix<U>>
    where
        F: FnMut(&[L], &[R]) -> U,
        U: Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let order = self.order;
        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let shape = Shape::new(nrows, ncols);
        let shape = MemoryShape::from_shape(shape, order);
        let size = shape.size::<U>()?;
        let mut data = Vec::with_capacity(size);

        if self.ncols() == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { order, shape, data });
        }

        self.set_order(Order::RowMajor);
        rhs.set_order(Order::ColMajor);

        match order {
            Order::RowMajor => {
                for row in 0..nrows {
                    for col in 0..ncols {
                        let lhs = unsafe { self.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }

            Order::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let lhs = unsafe { self.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = op(lhs, rhs);
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { order, shape, data })
    }
}

impl<T> Matrix<T> {
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
    ) -> Result<Matrix<U>>
    where
        F: FnMut(&'a T, &'b S) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        shape.size::<U>()?;
        let data = self
            .data
            .iter()
            .map(|element| op(element, scalar))
            .collect();

        Ok(Matrix { order, shape, data })
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
    ) -> Result<Matrix<U>>
    where
        F: FnMut(T, &'a S) -> U,
    {
        let order = self.order;
        let shape = self.shape;
        shape.size::<U>()?;
        let data = self
            .data
            .into_iter()
            .map(|element| op(element, scalar))
            .collect();

        Ok(Matrix { order, shape, data })
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
    #[inline]
    pub fn scalar_operation_assign<'a, S, F>(&mut self, scalar: &'a S, mut op: F) -> &mut Self
    where
        F: FnMut(&mut T, &'a S),
    {
        self.data.iter_mut().for_each(|element| op(element, scalar));
        self
    }
}

impl<T> Matrix<T> {
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
    use crate::matrix;
    use crate::testkit;

    #[test]
    fn test_is_square() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert!(matrix.is_square());
        });

        let matrix = matrix![[1, 2], [3, 4]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert!(matrix.is_square());
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            assert!(!matrix.is_square());
        });
    }

    #[test]
    fn test_is_elementwise_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(lhs.is_elementwise_operation_conformable(&rhs));
            assert!(rhs.is_elementwise_operation_conformable(&lhs));
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(!lhs.is_elementwise_operation_conformable(&rhs));
            assert!(!rhs.is_elementwise_operation_conformable(&lhs));
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(!lhs.is_elementwise_operation_conformable(&rhs));
            assert!(!rhs.is_elementwise_operation_conformable(&lhs));
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(!lhs.is_elementwise_operation_conformable(&rhs));
            assert!(!rhs.is_elementwise_operation_conformable(&lhs));
        });
    }

    #[test]
    fn test_is_multiplication_like_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(!lhs.is_multiplication_like_operation_conformable(&rhs));
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(!lhs.is_multiplication_like_operation_conformable(&rhs));
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            assert!(lhs.is_multiplication_like_operation_conformable(&rhs));
        });
    }

    #[test]
    fn test_ensure_square() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let result = matrix.ensure_square();
            assert!(result.is_ok());
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let result = matrix.ensure_square();
            assert_eq!(result, Err(Error::SquareMatrixRequired));
        });
    }

    #[test]
    fn test_ensure_elementwise_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let result = lhs.ensure_elementwise_operation_conformable(&rhs);
            assert!(result.is_ok());
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let result = lhs.ensure_elementwise_operation_conformable(&rhs);
            assert_eq!(result, Err(Error::ShapeNotConformable));
        });
    }

    #[test]
    fn test_ensure_multiplication_like_operation_conformable() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
            assert!(result.is_ok());
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let result = lhs.ensure_multiplication_like_operation_conformable(&rhs);
            assert_eq!(result, Err(Error::ShapeNotConformable));
        });
    }

    #[test]
    fn test_elementwise_operation() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs.elementwise_operation(&rhs, |x, y| x + y).unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        // This is a misuse but should work.
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = {
                let rhs = rhs.clone();
                lhs.elementwise_operation(&rhs, |x, _| x).unwrap()
            };
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            testkit::assert_loose_eq(&output, &expected);
        });

        // This is a misuse but should work.
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = {
                let lhs = lhs.clone();
                lhs.elementwise_operation(&rhs, |_, y| y).unwrap()
            };
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs.elementwise_operation(&rhs, |x, y| x + y).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs.elementwise_operation(&rhs, |x, y| x + y).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 0]; 3];
        let rhs = matrix![[0; 0]; 3];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation(&rhs, |_, _| ());
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 2]; 0];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation(&rhs, |_, _| ());
        });
    }

    #[test]
    fn test_elementwise_operation_consume_self() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs
                .clone()
                .elementwise_operation_consume_self(&rhs, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        // This is a misuse but should work.
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs
                .elementwise_operation_consume_self(&rhs, |_, y| y)
                .unwrap();
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .elementwise_operation_consume_self(&rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .elementwise_operation_consume_self(&rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 0]; 3];
        let rhs = matrix![[0; 0]; 3];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation_consume_self(&rhs, |_, _| ());
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 2]; 0];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation_consume_self(&rhs, |_, _| ());
        });
    }

    #[test]
    fn test_elementwise_operation_consume_rhs() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs
                .elementwise_operation_consume_rhs(rhs, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        // This is a misuse but should work.
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs
                .elementwise_operation_consume_rhs(rhs, |x, _| x)
                .unwrap();
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .elementwise_operation_consume_rhs(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .elementwise_operation_consume_rhs(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 0]; 3];
        let rhs = matrix![[0; 0]; 3];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation_consume_rhs(rhs, |_, _| ());
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 2]; 0];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation_consume_rhs(rhs, |_, _| ());
        });
    }

    #[test]
    fn test_elementwise_operation_consume_both() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs
                .elementwise_operation_consume_both(rhs, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .elementwise_operation_consume_both(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .elementwise_operation_consume_both(rhs, |x, y| x + y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 0]; 3];
        let rhs = matrix![[0; 0]; 3];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation_consume_both(rhs, |_, _| ());
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 2]; 0];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let _ = lhs.elementwise_operation_consume_both(rhs, |_, _| ());
        });
    }

    #[test]
    fn test_elementwise_operation_assign() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            lhs.elementwise_operation_assign(&rhs, |x, y| *x += y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&lhs, &expected);
        });

        // This is a misuse but should work.
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let mut lhs = lhs.map_ref(|x| x).unwrap();
            lhs.elementwise_operation_assign(&rhs, |x, y| *x = y)
                .unwrap();
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            testkit::assert_loose_eq(&lhs, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign(&rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            testkit::assert_loose_eq(&lhs, &unchanged);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign(&rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            testkit::assert_loose_eq(&lhs, &unchanged);
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 0]; 3];
        let rhs = matrix![[0; 0]; 3];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let _ = lhs.elementwise_operation_assign(&rhs, |_, _| ());
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 2]; 0];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let _ = lhs.elementwise_operation_assign(&rhs, |_, _| ());
        });
    }

    #[test]
    fn test_elementwise_operation_assign_consume_rhs() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            lhs.elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&lhs, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            testkit::assert_loose_eq(&lhs, &unchanged);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2], [2, 2], [2, 2]];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let unchanged = lhs.clone();
            let error = lhs
                .elementwise_operation_assign_consume_rhs(rhs, |x, y| *x += y)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
            testkit::assert_loose_eq(&lhs, &unchanged);
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 0]; 3];
        let rhs = matrix![[0; 0]; 3];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let _ = lhs.elementwise_operation_assign_consume_rhs(rhs, |_, _| ());
        });

        // Assert no panic from unflattening indices occurs.
        let lhs = matrix![[0; 2]; 0];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            let _ = lhs.elementwise_operation_assign_consume_rhs(rhs, |_, _| ());
        });
    }

    #[test]
    fn test_multiplication_like_operation() {
        fn dot_product(lhs: &[i32], rhs: &[i32]) -> i32 {
            lhs.iter()
                .zip(rhs)
                .map(|(x, y)| x * y)
                .reduce(|acc, p| acc + p)
                .unwrap()
        }

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4], [5, 6]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[22, 28], [49, 64]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1], [2], [3]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[14], [32]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs.multiplication_like_operation(rhs, dot_product).unwrap();
            let expected = matrix![[30, 36, 42], [66, 81, 96]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let error = lhs
                .multiplication_like_operation(rhs, dot_product)
                .unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        });

        let lhs = matrix![[0; 0]; isize::MAX as usize + 1];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            // The size of the resulting matrix would be `2 * isize::MAX + 2`,
            // which is greater than `usize::MAX`.
            let error = lhs
                .multiplication_like_operation(rhs, |_, _| 0)
                .unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
        });

        let lhs = matrix![[0; 0]; isize::MAX as usize - 1];
        let rhs = matrix![[0; 2]; 0];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            // The required capacity of the resulting matrix would be
            // `2 * isize::MAX - 2`, which is greater than `isize::MAX`.
            let error = lhs
                .multiplication_like_operation(rhs, |_, _| 0u8)
                .unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        });
    }

    #[test]
    fn test_scalar_operation() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = matrix.scalar_operation(&scalar, |x, y| x + y).unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        // This is a misuse but should work.
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let output = {
                let scalar = 2;
                matrix.scalar_operation(&scalar, |x, _| x).unwrap()
            };
            let expected = matrix![[&1, &2, &3], [&4, &5, &6]];
            testkit::assert_loose_eq(&output, &expected);
        });

        // This is a misuse but should work.
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = {
                let matrix = matrix.clone();
                matrix.scalar_operation(&scalar, |_, y| y).unwrap()
            };
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[(); usize::MAX]; 1];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let error = matrix.scalar_operation(&scalar, |_, _| 0).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow)
        });
    }

    #[test]
    fn test_scalar_operation_consume_self() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = matrix
                .scalar_operation_consume_self(&scalar, |x, y| x + y)
                .unwrap();
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        // This is a misuse but should work.
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = matrix
                .scalar_operation_consume_self(&scalar, |_, y| y)
                .unwrap();
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[(); usize::MAX]; 1];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let error = matrix
                .scalar_operation_consume_self(&scalar, |_, _| 0)
                .unwrap_err();
            assert_eq!(error, Error::CapacityOverflow)
        });
    }

    #[test]
    fn test_scalar_operation_assign() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            let scalar = 2;
            matrix.scalar_operation_assign(&scalar, |x, y| *x += y);
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&matrix, &expected);
        });

        // This is a misuse but should work.
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let mut matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            matrix.scalar_operation_assign(&scalar, |x, y| *x = y);
            let expected = matrix![[&2, &2, &2], [&2, &2, &2]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }
}
