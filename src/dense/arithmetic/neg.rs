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
