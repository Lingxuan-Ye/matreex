use super::super::Matrix;
use super::super::layout::Order;
use core::ops::{Rem, RemAssign};

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl<O> Rem<$s> for Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn rem(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element % *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Rem<$s> for &Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn rem(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element % *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Rem<Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn rem(self, rhs: Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar % element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Rem<&Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn rem(self, rhs: &Matrix<$t, O>) -> Self::Output {
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

            impl<O> RemAssign<$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn rem_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element %= *scalar);
                }
            }

            impl<O> RemAssign<&$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn rem_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element %= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_rem! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}
