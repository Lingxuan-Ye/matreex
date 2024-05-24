use super::super::Matrix;
use super::vector_dot_product;
use std::ops::{Add, Mul};

impl<L, R, U> Mul<&Matrix<R>> for &Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    fn mul(self, rhs: &Matrix<R>) -> Self::Output {
        let result = self.multiplication_like_operation(rhs, vector_dot_product);
        match result {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Mul<Matrix<R>> for &Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    fn mul(self, rhs: Matrix<R>) -> Self::Output {
        let result = self.multiplication_like_operation(&rhs, vector_dot_product);
        match result {
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

    fn mul(self, rhs: &Matrix<R>) -> Self::Output {
        let result = self.multiplication_like_operation(rhs, vector_dot_product);
        match result {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Mul<Matrix<R>> for Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    fn mul(self, rhs: Matrix<R>) -> Self::Output {
        let result = self.multiplication_like_operation(&rhs, vector_dot_product);
        match result {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

/// Implements scalar multiplication for [`Matrix`].
///
/// # Notes
///
/// Refer to [`impl_scalar_add!`] for more information.
///
/// [`impl_scalar_add!`]: crate::impl_scalar_add!
#[macro_export]
macro_rules! impl_scalar_mul {
    ($($t:ty)*) => {
        $(
            impl std::ops::Mul<&$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| (*element).clone() * scalar.clone())
                }
            }

            impl std::ops::Mul<$t> for &$crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| (*element).clone() * scalar.clone())
                }
            }

            impl std::ops::Mul<&$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl std::ops::Mul<$t> for $crate::matrix::Matrix<&$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl std::ops::Mul<&$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation(rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl std::ops::Mul<$t> for &$crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| element.clone() * scalar.clone())
                }
            }

            impl std::ops::Mul<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$t) -> Self::Output {
                    self.scalar_operation_consume_self(rhs, |element, scalar| element * scalar.clone())
                }
            }

            impl std::ops::Mul<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $t) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element * scalar.clone())
                }
            }

            impl std::ops::Mul<&$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() * (*element).clone())
                }
            }

            impl std::ops::Mul<$crate::matrix::Matrix<&$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl std::ops::Mul<&$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl std::ops::Mul<$crate::matrix::Matrix<$t>> for &$t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(self, |element, scalar| scalar.clone() * element)
                }
            }

            impl std::ops::Mul<&$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() * (*element).clone())
                }
            }

            impl std::ops::Mul<$crate::matrix::Matrix<&$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $crate::matrix::Matrix<&$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl std::ops::Mul<&$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: &$crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| scalar.clone() * element.clone())
                }
            }

            impl std::ops::Mul<$crate::matrix::Matrix<$t>> for $t
            where
                $t: Clone,
            {
                type Output = $crate::matrix::Matrix<$t>;

                fn mul(self, rhs: $crate::matrix::Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| scalar.clone() * element)
                }
            }

            impl std::ops::MulAssign<&$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                fn mul_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(rhs, |element, scalar| *element *= scalar.clone());
                }
            }

            impl std::ops::MulAssign<$t> for $crate::matrix::Matrix<$t>
            where
                $t: Clone,
            {
                fn mul_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= scalar.clone());
                }
            }
        )*
    }
}

impl_scalar_mul!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_mul() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[5, 4], [3, 2], [1, 0]];
        let expected = matrix![[5, 2], [32, 20]];

        {
            let result = &lhs * &rhs;
            assert_eq!(result, expected);
        }

        {
            rhs.switch_order();

            let result = &lhs * &rhs;
            assert_eq!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);

            rhs.switch_order();
        }

        {
            lhs.switch_order();

            let mut result = &lhs * &rhs;
            assert_ne!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);
            result.switch_order();
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_mul_fails() {
        let lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let rhs = matrix![[0, 1, 2], [3, 4, 5]];

        let _ = &lhs * &rhs;
    }

    #[test]
    fn test_mul_consume_rhs() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[5, 4], [3, 2], [1, 0]];
        let expected = matrix![[5, 2], [32, 20]];

        {
            let result = &lhs * rhs.clone();
            assert_eq!(result, expected);
        }

        {
            rhs.switch_order();

            let result = &lhs * rhs.clone();
            assert_eq!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);

            rhs.switch_order();
        }

        {
            lhs.switch_order();

            let mut result = &lhs * rhs.clone();
            assert_ne!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);
            result.switch_order();
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_mul_consume_rhs_fails() {
        let lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let rhs = matrix![[0, 1, 2], [3, 4, 5]];

        let _ = &lhs * rhs;
    }

    #[test]
    fn test_mul_consume_lhs() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[5, 4], [3, 2], [1, 0]];
        let expected = matrix![[5, 2], [32, 20]];

        {
            let result = lhs.clone() * &rhs;
            assert_eq!(result, expected);
        }

        {
            rhs.switch_order();

            let result = lhs.clone() * &rhs;
            assert_eq!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);

            rhs.switch_order();
        }

        {
            lhs.switch_order();

            let mut result = lhs.clone() * &rhs;
            assert_ne!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);
            result.switch_order();
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_mul_consume_lhs_fails() {
        let lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let rhs = matrix![[0, 1, 2], [3, 4, 5]];

        let _ = lhs * &rhs;
    }

    #[test]
    fn test_mul_consume_both() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let mut rhs = matrix![[5, 4], [3, 2], [1, 0]];
        let expected = matrix![[5, 2], [32, 20]];

        {
            let result = lhs.clone() * rhs.clone();
            assert_eq!(result, expected);
        }

        {
            rhs.switch_order();

            let result = lhs.clone() * rhs.clone();
            assert_eq!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);
            rhs.switch_order();
        }

        {
            lhs.switch_order();

            let mut result = lhs.clone() * rhs.clone();
            assert_ne!(result, expected);
            assert_eq!(result.order, lhs.order);
            assert_ne!(result.order, rhs.order);
            result.switch_order();
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_mul_consume_both_fails() {
        let lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let rhs = matrix![[0, 1, 2], [3, 4, 5]];

        let _ = lhs * rhs;
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_matrix_mul_scalar() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let rhs = 2;
        let expected = matrix![[0, 2, 4], [6, 8, 10]];

        {
            let result = &lhs * &rhs;
            assert_eq!(result, expected);

            let result = &lhs * rhs;
            assert_eq!(result, expected);

            let result = lhs.clone() * &rhs;
            assert_eq!(result, expected);

            let result = lhs.clone() * rhs;
            assert_eq!(result, expected);
        }

        {
            lhs.switch_order();

            let mut result: Matrix<i32> = &lhs * &rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);

            let mut result: Matrix<i32> = &lhs * rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);

            let mut result: Matrix<i32> = lhs.clone() * &rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);

            let mut result: Matrix<i32> = lhs.clone() * rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);
        }
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_scalar_mul_matrix() {
        let lhs = 2;
        let mut rhs = matrix![[0, 1, 2], [3, 4, 5]];
        let expected = matrix![[0, 2, 4], [6, 8, 10]];

        {
            let result = &lhs * &rhs;
            assert_eq!(result, expected);

            let result = lhs * &rhs;
            assert_eq!(result, expected);

            let result = &lhs * rhs.clone();
            assert_eq!(result, expected);

            let result = lhs * rhs.clone();
            assert_eq!(result, expected);
        }

        {
            rhs.switch_order();

            let mut result: Matrix<i32> = &lhs * &rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);

            let mut result: Matrix<i32> = lhs * &rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);

            let mut result: Matrix<i32> = &lhs * rhs.clone();
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);

            let mut result: Matrix<i32> = lhs * rhs.clone();
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_matrix_mul_scalar_assign() {
        let mut lhs = matrix![[0, 1, 2], [3, 4, 5]];
        let rhs = 2;
        let expected = matrix![[0, 2, 4], [6, 8, 10]];

        {
            let mut result = lhs.clone();
            result *= &rhs;
            assert_eq!(result, expected);

            let mut result = lhs.clone();
            result *= rhs;
            assert_eq!(result, expected);
        }

        {
            lhs.switch_order();

            let mut result = lhs.clone();
            result *= &rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);

            let mut result = lhs.clone();
            result *= rhs;
            assert_ne!(result, expected);
            result.switch_order();
            assert_eq!(result, expected);
        }
    }
}
