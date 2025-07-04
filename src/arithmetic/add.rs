use crate::Matrix;
use core::ops::{Add, AddAssign};

impl<L, R, U> Add<Matrix<R>> for Matrix<L>
where
    L: Add<R, Output = U>,
{
    type Output = Matrix<U>;

    #[inline]
    fn add(self, rhs: Matrix<R>) -> Self::Output {
        match self.elementwise_operation_consume_both(rhs, |left, right| left + right) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Add<&Matrix<R>> for Matrix<L>
where
    L: Add<R, Output = U>,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn add(self, rhs: &Matrix<R>) -> Self::Output {
        match self.elementwise_operation_consume_self(rhs, |left, right| left + right.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Add<Matrix<R>> for &Matrix<L>
where
    L: Add<R, Output = U> + Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn add(self, rhs: Matrix<R>) -> Self::Output {
        match self.elementwise_operation_consume_rhs(rhs, |left, right| left.clone() + right) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Add<&Matrix<R>> for &Matrix<L>
where
    L: Add<R, Output = U> + Clone,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn add(self, rhs: &Matrix<R>) -> Self::Output {
        match self.elementwise_operation(rhs, |left, right| left.clone() + right.clone()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R> AddAssign<Matrix<R>> for Matrix<L>
where
    L: AddAssign<R>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Matrix<R>) {
        if let Err(error) =
            self.elementwise_operation_assign_consume_rhs(rhs, |left, right| *left += right)
        {
            panic!("{error}");
        }
    }
}

impl<L, R> AddAssign<&Matrix<R>> for Matrix<L>
where
    L: AddAssign<R>,
    R: Clone,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Matrix<R>) {
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
            impl Add<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element + *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Add<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element + *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Add<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar + element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Add<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: &Matrix<$t>) -> Self::Output {
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

            impl AddAssign<$t> for Matrix<$t> {
                #[inline]
                fn add_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element += *scalar);
                }
            }

            impl AddAssign<&$t> for Matrix<$t> {
                #[inline]
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
    use self::mock::{MockL, MockR, MockU};
    use crate::matrix;
    use crate::testkit;

    mod mock {
        use core::ops::{Add, AddAssign};

        #[derive(Clone, Debug, PartialEq)]
        pub(super) struct MockL(pub(super) i32);

        #[derive(Clone)]
        pub(super) struct MockR(pub(super) i32);

        #[derive(Debug, PartialEq)]
        pub(super) struct MockU(pub(super) i32);

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
    }

    #[test]
    fn test_add() {
        let lhs = matrix![
            [MockL(1), MockL(2), MockL(3)],
            [MockL(4), MockL(5), MockL(6)],
        ];
        let rhs = matrix![
            [MockR(2), MockR(2), MockR(2)],
            [MockR(2), MockR(2), MockR(2)],
        ];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs + rhs;
            let expected = matrix![
                [MockU(3), MockU(4), MockU(5)],
                [MockU(6), MockU(7), MockU(8)],
            ];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![
            [MockL(1), MockL(2), MockL(3)],
            [MockL(4), MockL(5), MockL(6)],
        ];
        let rhs = matrix![
            [MockR(2), MockR(2), MockR(2)],
            [MockR(2), MockR(2), MockR(2)],
        ];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = lhs + &rhs;
            let expected = matrix![
                [MockU(3), MockU(4), MockU(5)],
                [MockU(6), MockU(7), MockU(8)],
            ];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![
            [MockL(1), MockL(2), MockL(3)],
            [MockL(4), MockL(5), MockL(6)],
        ];
        let rhs = matrix![
            [MockR(2), MockR(2), MockR(2)],
            [MockR(2), MockR(2), MockR(2)],
        ];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = &lhs + rhs;
            let expected = matrix![
                [MockU(3), MockU(4), MockU(5)],
                [MockU(6), MockU(7), MockU(8)],
            ];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![
            [MockL(1), MockL(2), MockL(3)],
            [MockL(4), MockL(5), MockL(6)],
        ];
        let rhs = matrix![
            [MockR(2), MockR(2), MockR(2)],
            [MockR(2), MockR(2), MockR(2)],
        ];
        testkit::for_each_order_binary(lhs, rhs, |lhs, rhs| {
            let output = &lhs + &rhs;
            let expected = matrix![
                [MockU(3), MockU(4), MockU(5)],
                [MockU(6), MockU(7), MockU(8)],
            ];
            testkit::assert_loose_eq(&output, &expected);
        });

        let lhs = matrix![
            [MockL(1), MockL(2), MockL(3)],
            [MockL(4), MockL(5), MockL(6)],
        ];
        let rhs = matrix![
            [MockR(2), MockR(2), MockR(2)],
            [MockR(2), MockR(2), MockR(2)],
        ];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            lhs += rhs;
            let expected = matrix![
                [MockL(3), MockL(4), MockL(5)],
                [MockL(6), MockL(7), MockL(8)],
            ];
            testkit::assert_loose_eq(&lhs, &expected);
        });

        let lhs = matrix![
            [MockL(1), MockL(2), MockL(3)],
            [MockL(4), MockL(5), MockL(6)],
        ];
        let rhs = matrix![
            [MockR(2), MockR(2), MockR(2)],
            [MockR(2), MockR(2), MockR(2)],
        ];
        testkit::for_each_order_binary(lhs, rhs, |mut lhs, rhs| {
            lhs += &rhs;
            let expected = matrix![
                [MockL(3), MockL(4), MockL(5)],
                [MockL(6), MockL(7), MockL(8)],
            ];
            testkit::assert_loose_eq(&lhs, &expected);
        });
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_add() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = matrix + scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = matrix + &scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &matrix + scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &matrix + &scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = scalar + matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &scalar + matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = scalar + &matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let scalar = 2;
            let output = &scalar + &matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = matrix + scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = matrix + &scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &matrix + scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &matrix + &scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = scalar + matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &scalar + matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = scalar + &matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |matrix| {
            let matrix = matrix.map_ref(|x| x).unwrap();
            let scalar = 2;
            let output = &scalar + &matrix;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            let scalar = 2;
            matrix += scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&matrix, &expected);
        });

        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        testkit::for_each_order_unary(matrix, |mut matrix| {
            let scalar = 2;
            matrix += &scalar;
            let expected = matrix![[3, 4, 5], [6, 7, 8]];
            testkit::assert_loose_eq(&matrix, &expected);
        });
    }
}
