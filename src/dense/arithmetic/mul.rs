use super::super::Matrix;
use super::super::layout::Order;
use crate::error::Result;
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
        match self.multiply(rhs) {
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

impl<L, LO> Matrix<L, LO>
where
    LO: Order,
{
    pub fn multiply<R, RO, U>(self, rhs: Matrix<R, RO>) -> Result<Matrix<U, LO>>
    where
        L: Mul<R, Output = U> + Clone,
        R: Clone,
        RO: Order,
        U: Add<Output = U> + Default,
    {
        self.multiplication_like_operation(rhs, |lhs_row, rhs_col| unsafe {
            lhs_row
                .iter()
                .zip(rhs_col)
                .map(|(lhs, rhs)| lhs.clone() * rhs.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::{dispatch_binary, dispatch_unary, matrix};

    #[derive(Clone)]
    struct MockL(i32);

    #[derive(Clone)]
    struct MockR(i32);

    #[derive(Debug, Default, PartialEq)]
    struct MockU(i32);

    impl Mul<MockR> for MockL {
        type Output = MockU;

        fn mul(self, rhs: MockR) -> Self::Output {
            MockU(self.0 * rhs.0)
        }
    }

    impl Add for MockU {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    #[test]
    fn test_mul() {
        dispatch_binary! {{
            let lhs = matrix![
                [MockL(1), MockL(2), MockL(3)],
                [MockL(4), MockL(5), MockL(6)],
            ].with_order::<LO>();
            let rhs = matrix![
                [MockR(1), MockR(2)],
                [MockR(3), MockR(4)],
                [MockR(5), MockR(6)],
            ].with_order::<RO>();
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
    fn test_multiply() {
        dispatch_binary! {{
            {
                let lhs = matrix![
                    [MockL(1), MockL(2), MockL(3)],
                    [MockL(4), MockL(5), MockL(6)],
                ].with_order::<LO>();
                let rhs = matrix![
                    [MockR(1), MockR(2)],
                    [MockR(3), MockR(4)],
                    [MockR(5), MockR(6)],
                ].with_order::<RO>();
                let output = lhs.multiply(rhs).unwrap();
                let expected = matrix![
                    [MockU(22), MockU(28)],
                    [MockU(49), MockU(64)],
                ];
                assert_eq!(output, expected);
            }

            {
                let lhs = matrix![
                    [MockL(1), MockL(2), MockL(3)],
                    [MockL(4), MockL(5), MockL(6)],
                ].with_order::<LO>();
                let rhs = matrix![
                    [MockR(1)],
                    [MockR(2)],
                    [MockR(3)],
                ].with_order::<RO>();
                let output = lhs.multiply(rhs).unwrap();
                let expected = matrix![
                    [MockU(14)],
                    [MockU(32)],
                ];
                assert_eq!(output, expected);
            }

            {
                let lhs = matrix![
                    [MockL(1), MockL(2), MockL(3)],
                    [MockL(4), MockL(5), MockL(6)],
                ].with_order::<LO>();
                let rhs =  matrix![
                    [MockR(1), MockR(2), MockR(3)],
                    [MockR(4), MockR(5), MockR(6)],
                    [MockR(7), MockR(8), MockR(9)],
                ].with_order::<RO>();
                let output = lhs.multiply(rhs).unwrap();
                let expected = matrix![
                    [MockU(30), MockU(36), MockU(42)],
                    [MockU(66), MockU(81), MockU(96)],
                ];
                assert_eq!(output, expected);
            }

            {
                let lhs = matrix![
                    [MockL(1), MockL(2), MockL(3)],
                    [MockL(4), MockL(5), MockL(6)],
                ].with_order::<LO>();
                let rhs = matrix![
                    [MockR(1), MockR(2)],
                    [MockR(3), MockR(4)],
                ].with_order::<RO>();
                let error = lhs.multiply(rhs).unwrap_err();
                assert_eq!(error, Error::ShapeNotConformable);
            }

            {
                let lhs = matrix![
                    [MockL(1), MockL(2), MockL(3)],
                    [MockL(4), MockL(5), MockL(6)],
                ].with_order::<LO>();
                let rhs = matrix![
                    [MockR(1), MockR(2), MockR(3)],
                    [MockR(4), MockR(5), MockR(6)],
                ].with_order::<RO>();
                let error = lhs.multiply(rhs).unwrap_err();
                assert_eq!(error, Error::ShapeNotConformable);
            }

            {
                let lhs = matrix![[0; 0]; 2].with_order::<LO>();
                let rhs = matrix![[0; usize::MAX]; 0].with_order::<RO>();
                let error = lhs.multiply(rhs).unwrap_err();
                assert_eq!(error, Error::SizeOverflow);
            }

            {
                let lhs = matrix![[0; 0]; 1].with_order::<LO>();
                let rhs = matrix![[0; usize::MAX]; 0].with_order::<RO>();
                let error = lhs.multiply(rhs).unwrap_err();
                assert_eq!(error, Error::CapacityOverflow);
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
