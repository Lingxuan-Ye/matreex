use super::super::Matrix;
use std::ops::Neg;

impl<T, U> Neg for Matrix<T>
where
    T: Neg<Output = U>,
{
    type Output = Matrix<U>;

    fn neg(self) -> Self::Output {
        self.map(|element| element.neg())
    }
}

impl<T, U> Neg for &Matrix<T>
where
    T: Neg<Output = U> + Clone,
{
    type Output = Matrix<U>;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}
