use super::super::order::Order;
use super::super::shape::Shape;
use super::super::Matrix;
use crate::error::Result;
use std::iter::zip;
use std::ops::{Add, Mul, MulAssign};

impl<L, R, U> Mul<Matrix<R>> for Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: Matrix<R>) -> Self::Output {
        match self.multiply(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Mul<&Matrix<R>> for Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: &Matrix<R>) -> Self::Output {
        self * rhs.clone()
    }
}

impl<L, R, U> Mul<Matrix<R>> for &Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: Matrix<R>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<L, R, U> Mul<&Matrix<R>> for &Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: &Matrix<R>) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<L> Matrix<L> {
    /// Performs elementwise multiplication on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_mul(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, 2, 4], [6, 8, 10]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_mul<R, U>(&self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Mul<R, Output = U> + Clone,
        R: Clone,
    {
        self.elementwise_operation(rhs, |left, right| left.clone() * right.clone())
    }

    /// Performs elementwise multiplication on two matrices, consuming `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_mul_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, 2, 4], [6, 8, 10]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_mul_consume_self<R, U>(self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Mul<R, Output = U>,
        R: Clone,
    {
        self.elementwise_operation_consume_self(rhs, |left, right| left * right.clone())
    }

    /// Performs elementwise multiplication on two matrices, assigning the result
    /// to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// lhs.elementwise_mul_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[0, 2, 4], [6, 8, 10]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_mul_assign<R>(&mut self, rhs: &Matrix<R>) -> Result<&mut Self>
    where
        L: MulAssign<R>,
        R: Clone,
    {
        self.elementwise_operation_assign(rhs, |left, right| *left *= right.clone())
    }
}

impl<L> Matrix<L> {
    /// Performs matrix multiplication on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// For performance reasons, this method consumes both `self` and `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[0, 1, 2], [3, 4, 5]];
    /// let rhs = matrix![[0, 1], [2, 3], [4, 5]];
    /// let result = lhs.multiply(rhs);
    /// assert_eq!(result, Ok(matrix![[10, 13], [28, 40]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    pub fn multiply<R, U>(mut self, mut rhs: Matrix<R>) -> Result<Matrix<U>>
    where
        L: Mul<R, Output = U> + Clone,
        R: Clone,
        U: Add<Output = U> + Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let order = self.order;
        let shape = Shape::new(nrows, ncols).try_to_axis_shape(order)?;
        let size = shape.size();
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
                        let element = unsafe {
                            zip(
                                self.iter_nth_major_axis_vector_unchecked(row),
                                rhs.iter_nth_major_axis_vector_unchecked(col),
                            )
                            .map(|(left, right)| left.clone() * right.clone())
                            .reduce(|accumulator, product| accumulator + product)
                            .unwrap_unchecked()
                        };
                        data.push(element);
                    }
                }
            }

            Order::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let element = unsafe {
                            zip(
                                self.iter_nth_major_axis_vector_unchecked(row),
                                rhs.iter_nth_major_axis_vector_unchecked(col),
                            )
                            .map(|(left, right)| left.clone() * right.clone())
                            .reduce(|accumulator, product| accumulator + product)
                            .unwrap_unchecked()
                        };
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { order, shape, data })
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Mul<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: $s) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element * *scalar)
                }
            }

            impl Mul<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: $s) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| *element * *scalar)
                }
            }

            impl Mul<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar * element)
                }
            }

            impl Mul<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: &Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| *scalar * *element)
                }
            }
        )*
    }
}

macro_rules! impl_primitive_scalar_mul {
    ($($t:ty)*) => {
        $(
            impl_helper! {
                ($t, $t, $t)
                ($t, &$t, $t)
                (&$t, $t, $t)
                (&$t, &$t, $t)
            }

            impl MulAssign<$t> for Matrix<$t> {
                #[inline]
                fn mul_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= *scalar);
                }
            }

            impl MulAssign<&$t> for Matrix<$t> {
                #[inline]
                fn mul_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_mul! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}

#[cfg(test)]
mod tests {
    use crate::error::Error;
    use crate::matrix;

    #[test]
    fn test_multiply() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[0, 1], [2, 3], [4, 5]];
        let expected = matrix![[10, 13], [28, 40]];

        // default order & default order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // default order &  alternative order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        // alternative order & alternative order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let mut output = lhs.multiply(rhs).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        rhs.switch_order();

        // alternative order & default order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let mut output = lhs.multiply(rhs).unwrap();
            output.switch_order();
            assert_eq!(output, expected);
        }

        lhs.switch_order();

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0], [1], [2]];
            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, matrix![[5], [14]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, matrix![[15, 18, 21], [42, 54, 66]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1], [2, 3]];
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[0, 1, 3], [4, 5, 6]];
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }
    }
}
