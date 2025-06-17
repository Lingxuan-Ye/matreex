use crate::Matrix;
use core::ops::{Div, DivAssign};

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Div<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element / *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Div<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element / *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Div<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar / element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Div<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn div(self, rhs: &Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation(&self, |element, scalar| *scalar / *element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
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
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_div() {
        let matrix = matrix![[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]];
        let matrix_ref = matrix.map_ref(|x| x).unwrap();
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
