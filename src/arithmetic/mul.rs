use crate::Matrix;
use crate::error::Result;
use crate::order::Order;
use crate::shape::Shape;
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
    /// let result = lhs.elementwise_mul(&rhs);
    /// assert_eq!(result, Ok(matrix![[2, 4, 6], [8, 10, 12]]));
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
    /// let result = lhs.elementwise_mul_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[2, 4, 6], [8, 10, 12]]));
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
    /// # use matreex::Result;
    /// use matreex::matrix;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// lhs.elementwise_mul_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[2, 4, 6], [8, 10, 12]]);
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
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[1, 2], [3, 4], [5, 6]];
    /// let result = lhs.multiply(rhs);
    /// assert_eq!(result, Ok(matrix![[22, 28], [49, 64]]));
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
                        let lhs = unsafe { self.get_nth_major_axis_vector(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector(col) };
                        let element = unsafe { dot_product(lhs, rhs).unwrap_unchecked() };
                        data.push(element);
                    }
                }
            }

            Order::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let lhs = unsafe { self.get_nth_major_axis_vector(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector(col) };
                        let element = unsafe { dot_product(lhs, rhs).unwrap_unchecked() };
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { order, shape, data })
    }
}

#[inline(always)]
fn dot_product<L, R, U>(lhs: &[L], rhs: &[R]) -> Option<U>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U>,
{
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.clone() * right.clone())
        .reduce(|accumulator, product| accumulator + product)
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
    fn test_mul() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4], [5, 6]];
        let expected = matrix![[22, 28], [49, 64]];

        assert_eq!(lhs.clone() * rhs.clone(), expected);
        assert_eq!(lhs.clone() * &rhs, expected);
        assert_eq!(&lhs * rhs.clone(), expected);
        assert_eq!(&lhs * &rhs, expected);
    }

    #[test]
    fn test_elementwise_mul() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[2, 4, 6], [8, 10, 12]];

        let output = lhs.elementwise_mul(&rhs).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_elementwise_mul_consume_self() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[2, 4, 6], [8, 10, 12]];

        let output = lhs.elementwise_mul_consume_self(&rhs).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_elementwise_mul_assign() {
        let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[2, 4, 6], [8, 10, 12]];

        lhs.elementwise_mul_assign(&rhs).unwrap();
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_multiply() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4], [5, 6]];
        let expected = matrix![[22, 28], [49, 64]];

        // default order & default order
        {
            let lhs = lhs.clone();
            let rhs = rhs.clone();

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // default order &  alternative order
        {
            let lhs = lhs.clone();
            let mut rhs = rhs.clone();
            rhs.switch_order();

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & default order
        {
            let mut lhs = lhs.clone();
            let rhs = rhs.clone();
            lhs.switch_order();

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // alternative order & alternative order
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.switch_order();
            rhs.switch_order();

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // more test cases

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1], [2], [3]];

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, matrix![[14], [32]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, matrix![[30, 36, 42], [66, 81, 96]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2], [3, 4]];

            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2, 3], [4, 5, 6]];

            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_mul() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let matrix_ref = matrix.map_ref(|x| x);
        let scalar = 2;
        let expected = matrix![[2, 4, 6], [8, 10, 12]];

        assert_eq!(matrix.clone() * scalar, expected);
        assert_eq!(matrix.clone() * &scalar, expected);
        assert_eq!(&matrix * scalar, expected);
        assert_eq!(&matrix * &scalar, expected);
        assert_eq!(scalar * matrix.clone(), expected);
        assert_eq!(&scalar * matrix.clone(), expected);
        assert_eq!(scalar * &matrix, expected);
        assert_eq!(&scalar * &matrix, expected);

        assert_eq!(matrix_ref.clone() * scalar, expected);
        assert_eq!(matrix_ref.clone() * &scalar, expected);
        assert_eq!(&matrix_ref * scalar, expected);
        assert_eq!(&matrix_ref * &scalar, expected);
        assert_eq!(scalar * matrix_ref.clone(), expected);
        assert_eq!(&scalar * matrix_ref.clone(), expected);
        assert_eq!(scalar * &matrix_ref, expected);
        assert_eq!(&scalar * &matrix_ref, expected);

        {
            let mut matrix = matrix.clone();

            matrix *= scalar;
            assert_eq!(matrix, expected);
        }

        {
            let mut matrix = matrix.clone();

            matrix *= &scalar;
            assert_eq!(matrix, expected);
        }
    }
}
