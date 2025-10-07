use super::super::Matrix;
use super::super::layout::Order;
use core::ops::{Sub, SubAssign};

impl<L, LO, R, RO, U> Sub<Matrix<R, RO>> for Matrix<L, LO>
where
    L: Sub<R, Output = U>,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_both(rhs, |lhs, rhs| lhs - rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Sub<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: Sub<R, Output = U>,
    LO: Order,
    R: Clone,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: &Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_self(rhs, |lhs, rhs| lhs - rhs.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Sub<Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Sub<R, Output = U> + Clone,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_rhs(rhs, |lhs, rhs| lhs.clone() - rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Sub<&Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Sub<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn sub(self, rhs: &Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation(rhs, |lhs, rhs| lhs.clone() - rhs.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO> SubAssign<Matrix<R, RO>> for Matrix<L, LO>
where
    L: SubAssign<R>,
    LO: Order,
    RO: Order,
{
    fn sub_assign(&mut self, rhs: Matrix<R, RO>) {
        if let Err(error) =
            self.elementwise_operation_assign_consume_rhs(rhs, |lhs, rhs| *lhs -= rhs)
        {
            panic!("{error}");
        }
    }
}

impl<L, LO, R, RO> SubAssign<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: SubAssign<R>,
    LO: Order,
    R: Clone,
    RO: Order,
{
    fn sub_assign(&mut self, rhs: &Matrix<R, RO>) {
        if let Err(error) = self.elementwise_operation_assign(rhs, |lhs, rhs| *lhs -= rhs.clone()) {
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
