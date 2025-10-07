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
        match self.elementwise_operation_consume_both(rhs, |lhs, rhs| lhs + rhs) {
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
        match self.elementwise_operation_consume_self(rhs, |lhs, rhs| lhs + rhs.clone()) {
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
        match self.elementwise_operation_consume_rhs(rhs, |lhs, rhs| lhs.clone() + rhs) {
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
        match self.elementwise_operation(rhs, |lhs, rhs| lhs.clone() + rhs.clone()) {
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
            self.elementwise_operation_assign_consume_rhs(rhs, |lhs, rhs| *lhs += rhs)
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
        if let Err(error) = self.elementwise_operation_assign(rhs, |lhs, rhs| *lhs += rhs.clone()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dispatch_binary, dispatch_unary, matrix};

    #[derive(Clone, Debug, PartialEq)]
    pub struct MockL(i32);

    #[derive(Clone)]
    struct MockR(i32);

    #[derive(Debug, PartialEq)]
    struct MockU(i32);

    impl Add<MockR> for MockL {
        type Output = MockU;

        fn add(self, rhs: MockR) -> Self::Output {
            MockU(self.0 + rhs.0)
        }
    }

    impl AddAssign<MockR> for MockL {
        fn add_assign(&mut self, rhs: MockR) {
            self.0 += rhs.0;
        }
    }

    #[test]
    fn test_add() {
        dispatch_binary! {{
            let lhs = matrix![
                [MockL(1), MockL(2), MockL(3)],
                [MockL(4), MockL(5), MockL(6)],
            ].with_order::<LO>();
            let rhs = matrix![
                [MockR(2), MockR(2), MockR(2)],
                [MockR(2), MockR(2), MockR(2)],
            ].with_order::<RO>();
            let expected = matrix![
                [MockU(3), MockU(4), MockU(5)],
                [MockU(6), MockU(7), MockU(8)],
            ];

            {
                let lhs = lhs.clone();
                let rhs = rhs.clone();
                let output = lhs + rhs;
                assert_eq!(output, expected);
            }

            {
                let lhs = lhs.clone();
                let output = lhs + &rhs;
                assert_eq!(output, expected);
            }

            {
                let rhs = rhs.clone();
                let output = &lhs + rhs;
                assert_eq!(output, expected);
            }

            {
                let output = &lhs + &rhs;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    fn test_add_assign() {
        dispatch_binary! {{
            let lhs = matrix![
                [MockL(1), MockL(2), MockL(3)],
                [MockL(4), MockL(5), MockL(6)],
            ].with_order::<LO>();
            let rhs = matrix![
                [MockR(2), MockR(2), MockR(2)],
                [MockR(2), MockR(2), MockR(2)],
            ].with_order::<RO>();
            let expected = matrix![
                [MockL(3), MockL(4), MockL(5)],
                [MockL(6), MockL(7), MockL(8)],
            ];

            {
                let mut lhs = lhs.clone();
                let rhs = rhs.clone();
                lhs += rhs;
                assert_eq!(lhs, expected);
            }

            {
                let mut lhs = lhs.clone();
                lhs += &rhs;
                assert_eq!(lhs, expected);
            }
        }}
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_add() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];

            {
                let matrix = matrix.clone();
                let output = matrix + scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.clone();
                let output = matrix + &scalar;
                assert_eq!(output, expected);
            }

            {
                let output = &matrix + scalar;
                assert_eq!(output, expected);
            }

            {
                let output = &matrix + &scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = matrix + scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = matrix + &scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &matrix + scalar;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &matrix + &scalar;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_add_rev() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];

            {
                let matrix = matrix.clone();
                let output = scalar + matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.clone();
                let output = &scalar + matrix;
                assert_eq!(output, expected);
            }

            {
                let output = scalar + &matrix;
                assert_eq!(output, expected);
            }

            {
                let output = &scalar + &matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = scalar + matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &scalar + matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = scalar + &matrix;
                assert_eq!(output, expected);
            }

            {
                let matrix = matrix.map_ref(|x| x).unwrap();
                let output = &scalar + &matrix;
                assert_eq!(output, expected);
            }
        }}
    }

    #[test]
    fn test_primitive_scalar_add_assign() {
        dispatch_unary! {{
            let matrix = matrix![[1, 2, 3], [4, 5, 6]].with_order::<O>();
            let scalar = 2;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];

            {
                let mut matrix = matrix.clone();
                matrix += scalar;
                assert_eq!(matrix, expected);
            }

            {
                let mut matrix = matrix.clone();
                matrix += &scalar;
                assert_eq!(matrix, expected);
            }
        }}
    }
}
