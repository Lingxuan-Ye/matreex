use crate::Matrix;
use crate::error::Result;
use std::ops::{Sub, SubAssign};

impl<L, R, U> Sub<Matrix<R>> for Matrix<L>
where
    L: Sub<R, Output = U>,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn sub(self, rhs: Matrix<R>) -> Self::Output {
        self - &rhs
    }
}

impl<L, R, U> Sub<&Matrix<R>> for Matrix<L>
where
    L: Sub<R, Output = U>,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn sub(self, rhs: &Matrix<R>) -> Self::Output {
        match self.elementwise_sub_consume_self(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Sub<Matrix<R>> for &Matrix<L>
where
    L: Sub<R, Output = U> + Clone,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn sub(self, rhs: Matrix<R>) -> Self::Output {
        self - &rhs
    }
}

impl<L, R, U> Sub<&Matrix<R>> for &Matrix<L>
where
    L: Sub<R, Output = U> + Clone,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn sub(self, rhs: &Matrix<R>) -> Self::Output {
        match self.elementwise_sub(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R> SubAssign<Matrix<R>> for Matrix<L>
where
    L: SubAssign<R>,
    R: Clone,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Matrix<R>) {
        *self -= &rhs;
    }
}

impl<L, R> SubAssign<&Matrix<R>> for Matrix<L>
where
    L: SubAssign<R>,
    R: Clone,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Matrix<R>) {
        if let Err(error) = self.elementwise_sub_assign(rhs) {
            panic!("{error}");
        }
    }
}

impl<L> Matrix<L> {
    /// Performs elementwise subtraction on two matrices.
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
    /// let result = lhs.elementwise_sub(&rhs);
    /// assert_eq!(result, Ok(matrix![[-1, 0, 1], [2, 3, 4]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_sub<R, U>(&self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Sub<R, Output = U> + Clone,
        R: Clone,
    {
        self.elementwise_operation(rhs, |left, right| left.clone() - right.clone())
    }

    /// Performs elementwise subtraction on two matrices, consuming `self`.
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
    /// let result = lhs.elementwise_sub_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[-1, 0, 1], [2, 3, 4]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_sub_consume_self<R, U>(self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Sub<R, Output = U>,
        R: Clone,
    {
        self.elementwise_operation_consume_self(rhs, |left, right| left - right.clone())
    }

    /// Performs elementwise subtraction on two matrices, assigning the result
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
    /// lhs.elementwise_sub_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[-1, 0, 1], [2, 3, 4]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_sub_assign<R>(&mut self, rhs: &Matrix<R>) -> Result<&mut Self>
    where
        L: SubAssign<R>,
        R: Clone,
    {
        self.elementwise_operation_assign(rhs, |left, right| *left -= right.clone())
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Sub<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn sub(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element - *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Sub<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn sub(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element - *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Sub<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn sub(self, rhs: Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar - element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Sub<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn sub(self, rhs: &Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation(&self, |element, scalar| *scalar - *element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }
        )*
    }
}

macro_rules! impl_primitive_scalar_sub {
    ($($t:ty)*) => {
        $(
            impl_helper! {
                ($t, $t, $t)
                ($t, &$t, $t)
                (&$t, $t, $t)
                (&$t, &$t, $t)
            }

            impl SubAssign<$t> for Matrix<$t> {
                #[inline]
                fn sub_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element -= *scalar);
                }
            }

            impl SubAssign<&$t> for Matrix<$t> {
                #[inline]
                fn sub_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element -= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_sub! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}

#[cfg(test)]
mod tests {
    use crate::matrix;

    #[test]
    fn test_sub() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[-1, 0, 1], [2, 3, 4]];

        assert_eq!(lhs.clone() - rhs.clone(), expected);
        assert_eq!(lhs.clone() - &rhs, expected);
        assert_eq!(&lhs - rhs.clone(), expected);
        assert_eq!(&lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_assign() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[-1, 0, 1], [2, 3, 4]];

        {
            let mut lhs = lhs.clone();

            lhs -= rhs.clone();
            assert_eq!(lhs, expected);
        }

        {
            let mut lhs = lhs.clone();

            lhs -= &rhs;
            assert_eq!(lhs, expected);
        }
    }

    #[test]
    fn test_elementwise_sub() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[-1, 0, 1], [2, 3, 4]];

        let output = lhs.elementwise_sub(&rhs).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_elementwise_sub_consume_self() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[-1, 0, 1], [2, 3, 4]];

        let output = lhs.elementwise_sub_consume_self(&rhs).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_elementwise_sub_assign() {
        let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[-1, 0, 1], [2, 3, 4]];

        lhs.elementwise_sub_assign(&rhs).unwrap();
        assert_eq!(lhs, expected);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_sub() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let matrix_ref = matrix.map_ref(|x| x).unwrap();
        let scalar = 2;
        let expected = matrix![[-1, 0, 1], [2, 3, 4]];
        let rexpected = matrix![[1, 0, -1], [-2, -3, -4]];

        assert_eq!(matrix.clone() - scalar, expected);
        assert_eq!(matrix.clone() - &scalar, expected);
        assert_eq!(&matrix - scalar, expected);
        assert_eq!(&matrix - &scalar, expected);
        assert_eq!(scalar - matrix.clone(), rexpected);
        assert_eq!(&scalar - matrix.clone(), rexpected);
        assert_eq!(scalar - &matrix, rexpected);
        assert_eq!(&scalar - &matrix, rexpected);

        assert_eq!(matrix_ref.clone() - scalar, expected);
        assert_eq!(matrix_ref.clone() - &scalar, expected);
        assert_eq!(&matrix_ref - scalar, expected);
        assert_eq!(&matrix_ref - &scalar, expected);
        assert_eq!(scalar - matrix_ref.clone(), rexpected);
        assert_eq!(&scalar - matrix_ref.clone(), rexpected);
        assert_eq!(scalar - &matrix_ref, rexpected);
        assert_eq!(&scalar - &matrix_ref, rexpected);

        {
            let mut matrix = matrix.clone();

            matrix -= scalar;
            assert_eq!(matrix, expected);
        }

        {
            let mut matrix = matrix.clone();

            matrix -= &scalar;
            assert_eq!(matrix, expected);
        }
    }
}
