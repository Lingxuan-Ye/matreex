use super::super::Matrix;
use super::super::layout::Order;
use core::ops::{Add, Mul, MulAssign};

impl<L, LO, R, RO, U> Mul<Matrix<R, RO>> for Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: Matrix<R, RO>) -> Self::Output {
        match self.multiplication_like_operation(rhs, |lhs_row, rhs_col| unsafe {
            lhs_row
                .iter()
                .zip(rhs_col)
                .map(|(lhs, rhs)| lhs.clone() * rhs.clone())
                .reduce(|sum, product| sum + product)
                .unwrap_unchecked()
        }) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, LO, R, RO, U> Mul<&Matrix<R, RO>> for Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: &Matrix<R, RO>) -> Self::Output {
        self * rhs.clone()
    }
}

impl<L, LO, R, RO, U> Mul<Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: Matrix<R, RO>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<L, LO, R, RO, U> Mul<&Matrix<R, RO>> for &Matrix<L, LO>
where
    L: Mul<R, Output = U> + Clone,
    LO: Order,
    R: Clone,
    RO: Order,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U, LO>;

    fn mul(self, rhs: &Matrix<R, RO>) -> Self::Output {
        self.clone() * rhs.clone()
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

#[cfg(test)]
mod tests {
    use crate::mock::{MockL, MockR, MockU};
    use crate::{dispatch_binary, dispatch_unary, matrix};

    #[test]
    fn test_mul() {
        dispatch_binary! {{
            let lhs = matrix![
                [MockL(1), MockL(2), MockL(3)],
                [MockL(4), MockL(5), MockL(6)],
            ].with_order::<O>();
            let rhs = matrix![
                [MockR(1), MockR(2)],
                [MockR(3), MockR(4)],
                [MockR(5), MockR(6)],
            ].with_order::<P>();
            let expected = matrix![[MockU(22), MockU(28)], [MockU(49), MockU(64)]];

            {
                let lhs = lhs.clone();
                let rhs = rhs.clone();
                let output = lhs * rhs;
                assert_eq!(output, expected);
            }

            {
                let lhs = lhs.clone();
                let output = lhs * &rhs;
                assert_eq!(output, expected);
            }

            {
                let rhs = rhs.clone();
                let output = &lhs * rhs;
                assert_eq!(output, expected);
            }

            {
                let output = &lhs * &rhs;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_mul() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[2, 4, 6], [8, 10, 12]];

            {
                let matrix = matrix.clone();
                let output = matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.clone();
                let output = matrix * &scalar;
                assert_eq!(output, expected);
            }

            {
                let output = &matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let output = &matrix * &scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = matrix * &scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &matrix * scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &matrix * &scalar;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_mul_rev() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[2, 4, 6], [8, 10, 12]];

            {
                let matrix = matrix.clone();
                let output = scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.clone();
                let output = &scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let output = scalar * &matrix;
                assert_eq!(output, expected);
            }

            {
                let output = &scalar * &matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &scalar * matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = scalar * &matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &scalar * &matrix;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    fn test_primitive_scalar_mul_assign() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[2, 4, 6], [8, 10, 12]];

            {
                let mut matrix = matrix.clone();
                matrix *= scalar;
                assert_eq!(matrix, expected);
            }

            {
                let mut matrix = matrix.clone();
                matrix *= &scalar;
                assert_eq!(matrix, expected);
            }
        }}
    }
}
