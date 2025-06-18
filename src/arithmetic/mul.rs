use crate::Matrix;
use crate::error::Result;
use crate::order::Order;
use crate::shape::{AxisShape, Shape};
use alloc::vec::Vec;
use core::ops::{Add, Mul, MulAssign};

impl<L, R, U> Mul<Matrix<R>> for Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: Matrix<R>) -> Self::Output {
        match self.multiply(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Mul<&Matrix<R>> for Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: &Matrix<R>) -> Self::Output {
        self * rhs.clone()
    }
}

impl<L, R, U> Mul<Matrix<R>> for &Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: Matrix<R>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<L, R, U> Mul<&Matrix<R>> for &Matrix<L>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U> + Default,
{
    type Output = Matrix<U>;

    #[inline]
    fn mul(self, rhs: &Matrix<R>) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<L> Matrix<L> {
    /// Performs matrix multiplication on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    /// - [`Error::SizeOverflow`] if size exceeds [`usize::MAX`].
    /// - [`Error::CapacityOverflow`] if required capacity in bytes exceeds [`isize::MAX`].
    ///
    /// # Notes
    ///
    /// The order of the resulting matrix will always be the same as that
    /// of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2], [2, 2], [2, 2]];
    /// let result = lhs.multiply(rhs);
    /// assert_eq!(result, Ok(matrix![[12, 12], [30, 30]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    /// [`Error::SizeOverflow`]: crate::error::Error::SizeOverflow
    /// [`Error::CapacityOverflow`]: crate::error::Error::CapacityOverflow
    pub fn multiply<R, U>(mut self, mut rhs: Matrix<R>) -> Result<Matrix<U>>
    where
        L: Mul<R, Output = U> + Clone,
        R: Clone,
        U: Add<Output = U> + Default,
    {
        self.ensure_multiplication_like_operation_conformable(&rhs)?;

        let order = self.order;
        let nrows = self.nrows();
        let ncols = rhs.ncols();
        let shape = Shape::new(nrows, ncols);
        let shape = AxisShape::from_shape(shape, order);
        let size = shape.size::<U>()?;
        let mut data = Vec::with_capacity(size);

        if self.ncols() == 0 {
            data.resize_with(size, U::default);
            return Ok(Matrix { order, shape, data });
        }

        self.set_order(Order::RowMajor);
        rhs.set_order(Order::ColMajor);

        match order {
            Order::RowMajor => {
                for row in 0..nrows {
                    for col in 0..ncols {
                        let lhs = unsafe { self.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = unsafe { dot_product(lhs, rhs).unwrap_unchecked() };
                        data.push(element);
                    }
                }
            }

            Order::ColMajor => {
                for col in 0..ncols {
                    for row in 0..nrows {
                        let lhs = unsafe { self.get_nth_major_axis_vector_unchecked(row) };
                        let rhs = unsafe { rhs.get_nth_major_axis_vector_unchecked(col) };
                        let element = unsafe { dot_product(lhs, rhs).unwrap_unchecked() };
                        data.push(element);
                    }
                }
            }
        }

        Ok(Matrix { order, shape, data })
    }
}

#[inline(always)]
fn dot_product<L, R, U>(lhs: &[L], rhs: &[R]) -> Option<U>
where
    L: Mul<R, Output = U> + Clone,
    R: Clone,
    U: Add<Output = U>,
{
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.clone() * right.clone())
        .reduce(|accumulator, product| accumulator + product)
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Mul<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation_consume_self(&rhs, |element, scalar| element * *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Mul<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: $s) -> Self::Output {
                    match self.scalar_operation(&rhs, |element, scalar| *element * *scalar) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Mul<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: Matrix<$t>) -> Self::Output {
                    match rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar * element) {
                        Err(error) => panic!("{error}"),
                        Ok(output) => output,
                    }
                }
            }

            impl Mul<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn mul(self, rhs: &Matrix<$t>) -> Self::Output {
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

            impl MulAssign<$t> for Matrix<$t> {
                #[inline]
                fn mul_assign(&mut self, rhs: $t) {
                    self.scalar_operation_assign(&rhs, |element, scalar| *element *= *scalar);
                }
            }

            impl MulAssign<&$t> for Matrix<$t> {
                #[inline]
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
    use crate::matrix;

    #[test]
    fn test_mul() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4], [5, 6]];
        let expected = matrix![[22, 28], [49, 64]];

        assert_eq!(lhs.clone() * rhs.clone(), expected);
        assert_eq!(lhs.clone() * &rhs, expected);
        assert_eq!(&lhs * rhs.clone(), expected);
        assert_eq!(&lhs * &rhs, expected);
    }

    #[test]
    fn test_multiply() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[1, 2], [3, 4], [5, 6]];
        let expected = matrix![[22, 28], [49, 64]];

        // row-major & row-major
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.set_order(Order::RowMajor);
            rhs.set_order(Order::RowMajor);

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // row-major & col-major
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.set_order(Order::RowMajor);
            rhs.set_order(Order::ColMajor);

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // col-major & row-major
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.set_order(Order::ColMajor);
            rhs.set_order(Order::RowMajor);

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // col-major & col-major
        {
            let mut lhs = lhs.clone();
            let mut rhs = rhs.clone();
            lhs.set_order(Order::ColMajor);
            rhs.set_order(Order::ColMajor);

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, expected);
        }

        // more test cases

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1], [2], [3]];

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, matrix![[14], [32]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

            let output = lhs.multiply(rhs).unwrap();
            assert_eq!(output, matrix![[30, 36, 42], [66, 81, 96]]);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2], [3, 4]];

            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = lhs.clone();
            let rhs = matrix![[1, 2, 3], [4, 5, 6]];

            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::ShapeNotConformable);
        }

        {
            let lhs = matrix![[0; 0]; isize::MAX as usize + 1];
            let rhs = matrix![[0; 2]; 0];

            // the size of the resulting matrix would be `2 * isize::MAX + 2`,
            // which is greater than `usize::MAX`
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::SizeOverflow);
        }

        {
            let lhs = matrix![[0u8; 0]; isize::MAX as usize - 1];
            let rhs = matrix![[0u8; 2]; 0];

            // the required capacity of the resulting matrix would be
            // `2 * isize::MAX - 2`, which is greater than `isize::MAX`
            let error = lhs.multiply(rhs).unwrap_err();
            assert_eq!(error, Error::CapacityOverflow);
        }
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_mul() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let matrix_ref = matrix.map_ref(|x| x).unwrap();
        let scalar = 2;
        let expected = matrix![[2, 4, 6], [8, 10, 12]];

        assert_eq!(matrix.clone() * scalar, expected);
        assert_eq!(matrix.clone() * &scalar, expected);
        assert_eq!(&matrix * scalar, expected);
        assert_eq!(&matrix * &scalar, expected);
        assert_eq!(scalar * matrix.clone(), expected);
        assert_eq!(&scalar * matrix.clone(), expected);
        assert_eq!(scalar * &matrix, expected);
        assert_eq!(&scalar * &matrix, expected);

        assert_eq!(matrix_ref.clone() * scalar, expected);
        assert_eq!(matrix_ref.clone() * &scalar, expected);
        assert_eq!(&matrix_ref * scalar, expected);
        assert_eq!(&matrix_ref * &scalar, expected);
        assert_eq!(scalar * matrix_ref.clone(), expected);
        assert_eq!(&scalar * matrix_ref.clone(), expected);
        assert_eq!(scalar * &matrix_ref, expected);
        assert_eq!(&scalar * &matrix_ref, expected);

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
    }
}
