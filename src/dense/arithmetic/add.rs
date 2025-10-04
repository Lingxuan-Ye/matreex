use super::super::Matrix;
use super::super::layout::Order;
use core::ops::{Add, AddAssign};

impl<L, LO, R, RO, U> Add<Matrix<R, RO>> for Matrix<L, LO>
where
    L: Add<R, Output = U>,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn add(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_both(rhs, |left, right| left + right) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Add<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: Add<R, Output = U>,
    LO: Order,
    R: Clone,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn add(self, rhs: &Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_self(rhs, |left, right| left + right.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Add<Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Add<R, Output = U> + Clone,
    LO: Order,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn add(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation_consume_rhs(rhs, |left, right| left.clone() + right) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Add<&Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Add<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
{
    type Output = Matrix<U, LO>;

    fn add(self, rhs: &Matrix<R, RO>) -> Self::Output {
        match self.elementwise_operation(rhs, |left, right| left.clone() + right.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO> AddAssign<Matrix<R, RO>> for Matrix<L, LO>
where
    L: AddAssign<R>,
    LO: Order,
    RO: Order,
{
    fn add_assign(&mut self, rhs: Matrix<R, RO>) {
        if let Err(error) =
            self.elementwise_operation_assign_consume_rhs(rhs, |left, right| *left += right)
        {
            panic!("{error}");
        }
    }
}

impl<L, LO, R, RO> AddAssign<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: AddAssign<R>,
    LO: Order,
    R: Clone,
    RO: Order,
{
    fn add_assign(&mut self, rhs: &Matrix<R, RO>) {
        if let Err(error) =
            self.elementwise_operation_assign(rhs, |left, right| *left += right.clone())
        {
            panic!("{error}");
        }
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl<O> Add<$s> for Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn add(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element + *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Add<$s> for &Matrix<$t, O>
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn add(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element + *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Add<Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn add(self, rhs: Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar + element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl<O> Add<&Matrix<$t, O>> for $s
            where
                O: Order
            {
                type Output = Matrix<$u, O>;

                fn add(self, rhs: &Matrix<$t, O>) -> Self::Output {
                    match rhs.scalar_operation(&self, |element, scalar| *scalar + *element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }
        )*
    }
}

macro_rules! impl_primitive_scalar_add {
    ($($t:ty)*) => {
        $(
            impl_helper! {
                ($t, $t, $t)
                ($t, &$t, $t)
                (&$t, $t, $t)
                (&$t, &$t, $t)
            }

            impl<O> AddAssign<$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn add_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element += *scalar);
                }
            }

            impl<O> AddAssign<&$t> for Matrix<$t, O>
            where
                O: Order
            {
                fn add_assign(&mut self, rhs: &$t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element += *scalar);
                }
            }
        )*
    }
}

impl_primitive_scalar_add! {u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64}
