use super::super::Matrix;
use crate::error::Result;
use std::ops::{Rem, RemAssign};

impl<L> Matrix<L> {
    /// Performs elementwise remainder operation on two matrices.
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
    /// let result = lhs.elementwise_rem(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, 1, 0], [1, 0, 1]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_rem<R, U>(&self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Rem<R, Output = U> + Clone,
        R: Clone,
    {
        self.elementwise_operation(rhs, |left, right| left.clone() % right.clone())
    }

    /// Performs elementwise remainder operation on two matrices,
    /// consuming `self`.
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
    /// let result = lhs.elementwise_rem_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[0, 1, 0], [1, 0, 1]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_rem_consume_self<R, U>(self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Rem<R, Output = U>,
        R: Clone,
    {
        self.elementwise_operation_consume_self(rhs, |left, right| left % right.clone())
    }

    /// Performs elementwise remainder operation on two matrices,
    /// assigning the result to `self`.
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
    /// lhs.elementwise_rem_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[0, 1, 0], [1, 0, 1]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_rem_assign<R>(&mut self, rhs: &Matrix<R>) -> Result<&mut Self>
    where
        L: RemAssign<R>,
        R: Clone,
    {
        self.elementwise_operation_assign(rhs, |left, right| *left %= right.clone())
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Rem<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: $s) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element % *scalar)
                }
            }

            impl Rem<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: $s) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| *element % *scalar)
                }
            }

            impl Rem<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar % element)
                }
            }

            impl Rem<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: &Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| *scalar % *element)
                }
            }
        )*
    }
}

macro_rules! impl_primitive_scalar_rem {
    ($($t:ty)*) => {
        $(
            impl_helper! {
                ($t, $t, $t)
                ($t, &$t, $t)
                (&$t, $t, $t)
                (&$t, &$t, $t)
            }

            impl RemAssign<$t> for Matrix<$t> {
                #[inline]
                fn rem_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element %= *scalar);
                }
            }

            impl RemAssign<&$t> for Matrix<$t> {
                #[inline]
                fn rem_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element %= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_rem! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}

#[cfg(test)]
mod tests {
    use crate::matrix;

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_rem() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let scalar = 2;
        let expected = matrix![[1, 0, 1], [0, 1, 0]];
        let rexpected = matrix![[0, 0, 2], [2, 2, 2]];

        assert_eq!(matrix.clone() % scalar, expected);
        assert_eq!(matrix.clone() % &scalar, expected);
        assert_eq!(&matrix % scalar, expected);
        assert_eq!(&matrix % &scalar, expected);
        assert_eq!(scalar % matrix.clone(), rexpected);
        assert_eq!(&scalar % matrix.clone(), rexpected);
        assert_eq!(scalar % &matrix, rexpected);
        assert_eq!(&scalar % &matrix, rexpected);

        let matrix = matrix![[&1, &2, &3], [&4, &5, &6]];

        assert_eq!(matrix.clone() % scalar, expected);
        assert_eq!(matrix.clone() % &scalar, expected);
        assert_eq!(&matrix % scalar, expected);
        assert_eq!(&matrix % &scalar, expected);
        assert_eq!(scalar % matrix.clone(), rexpected);
        assert_eq!(&scalar % matrix.clone(), rexpected);
        assert_eq!(scalar % &matrix, rexpected);
        assert_eq!(&scalar % &matrix, rexpected);
    }
}
