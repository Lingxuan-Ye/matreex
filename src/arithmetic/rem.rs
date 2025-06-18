use crate::Matrix;
use core::ops::{Rem, RemAssign};

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Rem<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element % *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Rem<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element % *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Rem<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar % element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Rem<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn rem(self, rhs: &Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation(&self, |element, scalar| *scalar % *element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
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
    use crate::testkit;

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_rem() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = matrix % scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = matrix % &scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &matrix % scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &matrix % &scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = scalar % matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &scalar % matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = scalar % &matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &scalar % &matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = matrix % scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = matrix % &scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &matrix % scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &matrix % &scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = scalar % matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &scalar % matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = scalar % &matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &scalar % &matrix;
            let expected = matrix![[0, 0, 2], [2, 2, 2]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            let scalar = 2;
            matrix %= scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&matrix, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            let scalar = 2;
            matrix %= &scalar;
            let expected = matrix![[1, 0, 1], [0, 1, 0]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }
}
