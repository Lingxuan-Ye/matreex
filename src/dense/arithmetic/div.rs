use super::super::Matrix;
use super::super::layout::Order;
use core::ops::{Div, DivAssign};

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl<O> Div<$s> for Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn div(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element / *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Div<$s> for &Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn div(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element / *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Div<Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn div(self, rhs: Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar / element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Div<&Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn div(self, rhs: &Matrix<$t, O>) -> Self::Output {
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

            impl<O> DivAssign<$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn div_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element /= *scalar);
                }
            }

            impl<O> DivAssign<&$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn div_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element /= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_div! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}
