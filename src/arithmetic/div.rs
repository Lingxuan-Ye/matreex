use crate::Matrix;
use crate::error::Result;
use std::ops::{Div, DivAssign};

impl<L> Matrix<L> {
    /// Performs elementwise division on two matrices.
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
    /// let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
    /// let result = lhs.elementwise_div(&rhs);
    /// assert_eq!(result, Ok(matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_div<R, U>(&self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Div<R, Output = U> + Clone,
        R: Clone,
    {
        self.elementwise_operation(rhs, |left, right| left.clone() / right.clone())
    }

    /// Performs elementwise division on two matrices, consuming `self`.
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
    /// let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
    /// let result = lhs.elementwise_div_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_div_consume_self<R, U>(self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Div<R, Output = U>,
        R: Clone,
    {
        self.elementwise_operation_consume_self(rhs, |left, right| left / right.clone())
    }

    /// Performs elementwise division on two matrices, assigning the result
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
    /// let mut lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
    /// lhs.elementwise_div_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_div_assign<R>(&mut self, rhs: &Matrix<R>) -> Result<&mut Self>
    where
        L: DivAssign<R>,
        R: Clone,
    {
        self.elementwise_operation_assign(rhs, |left, right| *left /= right.clone())
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Div<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: $s) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element / *scalar)
                }
            }

            impl Div<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: $s) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| *element / *scalar)
                }
            }

            impl Div<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar / element)
                }
            }

            impl Div<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: &Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| *scalar / *element)
                }
            }
        )*
    }
}

macro_rules! impl_primitive_scalar_div {
    ($($t:ty)*) => {
        $(
            impl_helper! {
                ($t, $t, $t)
                ($t, &$t, $t)
                (&$t, $t, $t)
                (&$t, &$t, $t)
            }

            impl DivAssign<$t> for Matrix<$t> {
                #[inline]
                fn div_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element /= *scalar);
                }
            }

            impl DivAssign<&$t> for Matrix<$t> {
                #[inline]
                fn div_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element /= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_div! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}

#[cfg(test)]
mod tests {
    use crate::matrix;

    #[test]
    fn test_elementwise_div() {
        let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
        let expected = matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]];

        let output = lhs.elementwise_div(&rhs).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_elementwise_div_consume_self() {
        let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
        let expected = matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]];

        let output = lhs.elementwise_div_consume_self(&rhs).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_elementwise_div_assign() {
        let mut lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
        let expected = matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]];

        lhs.elementwise_div_assign(&rhs).unwrap();
        assert_eq!(lhs, expected);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_div() {
        let matrix = matrix![[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]];
        let matrix_ref = matrix.map_ref(|x| x);
        let scalar = 2.0;
        let expected = matrix![[0.5, 1.0, 2.0], [4.0, 8.0, 16.0]];
        let rexpected = matrix![[2.0, 1.0, 0.5], [0.25, 0.125, 0.0625]];

        assert_eq!(matrix.clone() / scalar, expected);
        assert_eq!(matrix.clone() / &scalar, expected);
        assert_eq!(&matrix / scalar, expected);
        assert_eq!(&matrix / &scalar, expected);
        assert_eq!(scalar / matrix.clone(), rexpected);
        assert_eq!(&scalar / matrix.clone(), rexpected);
        assert_eq!(scalar / &matrix, rexpected);
        assert_eq!(&scalar / &matrix, rexpected);

        assert_eq!(matrix_ref.clone() / scalar, expected);
        assert_eq!(matrix_ref.clone() / &scalar, expected);
        assert_eq!(&matrix_ref / scalar, expected);
        assert_eq!(&matrix_ref / &scalar, expected);
        assert_eq!(scalar / matrix_ref.clone(), rexpected);
        assert_eq!(&scalar / matrix_ref.clone(), rexpected);
        assert_eq!(scalar / &matrix_ref, rexpected);
        assert_eq!(&scalar / &matrix_ref, rexpected);

        {
            let mut matrix = matrix.clone();

            matrix /= scalar;
            assert_eq!(matrix, expected);
        }

        {
            let mut matrix = matrix.clone();

            matrix /= &scalar;
            assert_eq!(matrix, expected);
        }
    }
}
