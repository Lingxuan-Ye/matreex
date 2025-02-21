use super::super::Matrix;
use std::ops::Neg;

impl<T, U> Neg for Matrix<T>
where
    T: Neg<Output = U>,
{
    type Output = Matrix<U>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.map(|element| element.neg())
    }
}

impl<T, U> Neg for &Matrix<T>
where
    T: Neg<Output = U> + Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.map_ref(|element| element.clone().neg())
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix;

    #[test]
    fn neg() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let expected = matrix![[-1, -2, -3], [-4, -5, -6]];

        assert_eq!(-matrix.clone(), expected);
        assert_eq!(-&matrix, expected);
    }
}
