use super::super::Matrix;
use super::super::layout::Order;
use core::ops::Neg;

impl<T, O, U> Neg for Matrix<T, O>
where
    T: Neg<Output = U>,
    O: Order,
{
    type Output = Matrix<U, O>;

    fn neg(self) -> Self::Output {
        match self.map(|element| element.neg()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<T, O, U> Neg for &Matrix<T, O>
where
    T: Neg<Output = U> + Clone,
    O: Order,
{
    type Output = Matrix<U, O>;

    fn neg(self) -> Self::Output {
        match self.map_ref(|element| element.clone().neg()) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mock::{MockT, MockU};
    use crate::{dispatch_unary, matrix};

    #[test]
    fn test_neg() {
        dispatch_unary! {{
            let matrix = matrix![
                [MockT(1), MockT(2), MockT(3)],
                [MockT(4), MockT(5), MockT(6)],
            ].with_order::<O>();
            let expected = matrix![
                [MockU(-1), MockU(-2), MockU(-3)],
                [MockU(-4), MockU(-5), MockU(-6)],
            ];

            {
                let matrix = matrix.clone();
                let output = -matrix;
                assert_eq!(output, expected);
            }

            {
                let output = -&matrix;
                assert_eq!(output, expected);
            }
        }}
    }
}
