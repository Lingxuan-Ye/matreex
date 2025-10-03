use super::super::Matrix;
use super::super::layout::Order;
use core::ops::{Sub, SubAssign};

impl<L, R, U, LO, RO> Sub<Matrix<R, RO>> for Matrix<L, LO>
where
    L: Sub<R, Output = U>,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_both(rhs, |left, right| left - right) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U, LO, RO> Sub<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: Sub<R, Output = U>,
    R: Clone,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: &Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_self(rhs, |left, right| left - right.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U, LO, RO> Sub<Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Sub<R, Output = U> + Clone,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_rhs(rhs, |left, right| left.clone() - right) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U, LO, RO> Sub<&Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Sub<R, Output = U> + Clone,
    R: Clone,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: &Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation(rhs, |left, right| left.clone() - right.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, LO, RO> SubAssign<Matrix<R, RO>> for Matrix<L, LO>
where
    L: SubAssign<R>,
    LO: Order,
    RO: Order,
{
    fn sub_assign(&mut self, rhs: Matrix<R, RO>) {
        if let Err(error) =
            self.elementwise_operation_assign_consume_rhs(rhs, |left, right| *left -= right)
        {
            panic!("{error}");
        }
    }
}

impl<L, R, LO, RO> SubAssign<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: SubAssign<R>,
    R: Clone,
    LO: Order,
    RO: Order,
{
    fn sub_assign(&mut self, rhs: &Matrix<R, RO>) {
        if let Err(error) =
            self.elementwise_operation_assign(rhs, |left, right| *left -= right.clone())
        {
            panic!("{error}");
        }
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl<O> Sub<$s> for Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn sub(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element - *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Sub<$s> for &Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn sub(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element - *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Sub<Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn sub(self, rhs: Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar - element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Sub<&Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn sub(self, rhs: &Matrix<$t, O>) -> Self::Output {
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

            impl<O> SubAssign<$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn sub_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element -= *scalar);
                }
            }

            impl<O> SubAssign<&$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn sub_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element -= *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_sub! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}
