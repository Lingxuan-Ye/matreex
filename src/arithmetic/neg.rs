use crate::Matrix;
use core::ops::Neg;

impl<T, U> Neg for Matrix<T>
where
    T: Neg<Output = U>,
{
    type Output = Matrix<U>;

    #[inline]
    fn neg(self) -> Self::Output {
        match self.map(|element| element.neg()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<T, U> Neg for &Matrix<T>
where
    T: Neg<Output = U> + Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn neg(self) -> Self::Output {
        match self.map_ref(|element| element.clone().neg()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

#[cfg(test)]
mod tests {
    use self::mock::{MockT, MockU};
    use crate::matrix;
    use crate::testkit;

    mod mock {
        use core::ops::Neg;

        #[derive(Clone)]
        pub(super) struct MockT(pub(super) i32);

        #[derive(Debug, PartialEq)]
        pub(super) struct MockU(pub(super) i32);

        impl Neg for MockT {
            type Output = MockU;

            fn neg(self) -> Self::Output {
                MockU(-self.0)
            }
        }
    }

    #[test]
    fn neg() {
        let matrix = matrix![
            [MockT(1), MockT(2), MockT(3)],
            [MockT(4), MockT(5), MockT(6)],
        ];
        testkit::for_each_order_unary(matrix, |matrix| {
            let output = -matrix;
            let expected = matrix![
                [MockU(-1), MockU(-2), MockU(-3)],
                [MockU(-4), MockU(-5), MockU(-6)],
            ];
            testkit::assert_loose_eq(&output, &expected);
        });

        let matrix = matrix![
            [MockT(1), MockT(2), MockT(3)],
            [MockT(4), MockT(5), MockT(6)],
        ];
        testkit::for_each_order_unary(matrix, |matrix| {
            let output = -&matrix;
            let expected = matrix![
                [MockU(-1), MockU(-2), MockU(-3)],
                [MockU(-4), MockU(-5), MockU(-6)],
            ];
            testkit::assert_loose_eq(&output, &expected);
        });
    }
}
