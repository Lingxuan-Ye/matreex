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
    use crate::matrix;

    #[test]
    fn test_add() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        assert_eq!(lhs.clone() + rhs.clone(), expected);
        assert_eq!(lhs.clone() + &rhs, expected);
        assert_eq!(&lhs + rhs.clone(), expected);
        assert_eq!(&lhs + &rhs, expected);
    }

    #[test]
    fn test_add_assign() {
        let lhs = matrix![[1, 2, 3], [4, 5, 6]];
        let rhs = matrix![[2, 2, 2], [2, 2, 2]];
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        {
            let mut lhs = lhs.clone();

            lhs += rhs.clone();
            assert_eq!(lhs, expected);
        }

        {
            let mut lhs = lhs.clone();

            lhs += &rhs;
            assert_eq!(lhs, expected);
        }
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_add() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
        let matrix_ref = matrix.map_ref(|x| x).unwrap();
        let scalar = 2;
        let expected = matrix![[3, 4, 5], [6, 7, 8]];

        assert_eq!(matrix.clone() + scalar, expected);
        assert_eq!(matrix.clone() + &scalar, expected);
        assert_eq!(&matrix + scalar, expected);
        assert_eq!(&matrix + &scalar, expected);
        assert_eq!(scalar + matrix.clone(), expected);
        assert_eq!(&scalar + matrix.clone(), expected);
        assert_eq!(scalar + &matrix, expected);
        assert_eq!(&scalar + &matrix, expected);

        assert_eq!(matrix_ref.clone() + scalar, expected);
        assert_eq!(matrix_ref.clone() + &scalar, expected);
        assert_eq!(&matrix_ref + scalar, expected);
        assert_eq!(&matrix_ref + &scalar, expected);
        assert_eq!(scalar + matrix_ref.clone(), expected);
        assert_eq!(&scalar + matrix_ref.clone(), expected);
        assert_eq!(scalar + &matrix_ref, expected);
        assert_eq!(&scalar + &matrix_ref, expected);

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
    }
}
