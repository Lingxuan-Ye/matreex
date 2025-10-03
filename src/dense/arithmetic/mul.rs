use super::super::Matrix;
use super::super::layout::Order;
use crate::error::Result;
use core::ops::{Add, Mul, MulAssign};

impl<L, R, U, LO, RO> Mul<Matrix<R, RO>> for Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.multiply(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U, LO, RO> Mul<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: &Matrix<R, RO>) -> Self::Output {
        self * rhs.clone()
    }
}

impl<L, R, U, LO, RO> Mul<Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: Matrix<R, RO>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<L, R, U, LO, RO> Mul<&Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: &Matrix<R, RO>) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<L, LO> Matrix<L, LO>
where
    LO: Order,
{
    pub fn multiply<R, U, RO>(self, rhs: Matrix<R, RO>) -> Result<Matrix<U, LO>>
    where
        L: Mul<R, Output = U> + Clone,
        R: Clone,
        U: Add<Output = U> + Default,
        RO: Order,
    {
        self.multiplication_like_operation(rhs, |left_row, right_col| unsafe {
            left_row
                .iter()
                .zip(right_col)
                .map(|(left, right)| left.clone() * right.clone())
                .reduce(|sum, product| sum + product)
                .unwrap_unchecked()
        })
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl<O> Mul<$s> for Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element * *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Mul<$s> for &Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element * *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Mul<Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar * element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Mul<&Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn mul(self, rhs: &Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation(&self, |element, scalar| *scalar * *element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
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

            impl<O> MulAssign<$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn mul_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= *scalar);
                }
            }

            impl<O> MulAssign<&$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn mul_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_mul! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}
