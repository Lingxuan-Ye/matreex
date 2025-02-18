use super::super::Matrix;
use crate::error::Result;
use std::ops::{Add, AddAssign};

impl<L, R, U> Add<Matrix<R>> for Matrix<L>
where
    L: Add<R, Output = U>,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn add(self, rhs: Matrix<R>) -> Self::Output {
        self + &rhs
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
        match self.elementwise_add_consume_self(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R, U> Add<Matrix<R>> for &Matrix<L>
where
    L: Add<R, Output = U> + Clone,
    R: Clone,
{
    type Output = Matrix<U>;

    #[inline]
    fn add(self, rhs: Matrix<R>) -> Self::Output {
        self + &rhs
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
        match self.elementwise_add(rhs) {
            Err(error) => panic!("{error}"),
            Ok(output) => output,
        }
    }
}

impl<L, R> AddAssign<Matrix<R>> for Matrix<L>
where
    L: AddAssign<R>,
    R: Clone,
{
    #[inline]
    fn add_assign(&mut self, rhs: Matrix<R>) {
        *self += &rhs
    }
}

impl<L, R> AddAssign<&Matrix<R>> for Matrix<L>
where
    L: AddAssign<R>,
    R: Clone,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Matrix<R>) {
        if let Err(error) = self.elementwise_add_assign(rhs) {
            panic!("{error}");
        }
    }
}

impl<L> Matrix<L> {
    /// Performs elementwise addition on two matrices.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_add(&rhs);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_add<R, U>(&self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Add<R, Output = U> + Clone,
        R: Clone,
    {
        self.elementwise_operation(rhs, |left, right| left.clone() + right.clone())
    }

    /// Performs elementwise addition on two matrices, consuming `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Notes
    ///
    /// The resulting matrix will always have the same order as `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    ///
    /// let lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// let result = lhs.elementwise_add_consume_self(&rhs);
    /// assert_eq!(result, Ok(matrix![[3, 4, 5], [6, 7, 8]]));
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_add_consume_self<R, U>(self, rhs: &Matrix<R>) -> Result<Matrix<U>>
    where
        L: Add<R, Output = U>,
        R: Clone,
    {
        self.elementwise_operation_consume_self(rhs, |left, right| left + right.clone())
    }

    /// Performs elementwise addition on two matrices, assigning the result
    /// to `self`.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeNotConformable`] if the matrices are not conformable.
    ///
    /// # Examples
    ///
    /// ```
    /// use matreex::matrix;
    /// # use matreex::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut lhs = matrix![[1, 2, 3], [4, 5, 6]];
    /// let rhs = matrix![[2, 2, 2], [2, 2, 2]];
    /// lhs.elementwise_add_assign(&rhs)?;
    /// assert_eq!(lhs, matrix![[3, 4, 5], [6, 7, 8]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Error::ShapeNotConformable`]: crate::error::Error::ShapeNotConformable
    #[inline]
    pub fn elementwise_add_assign<R>(&mut self, rhs: &Matrix<R>) -> Result<&mut Self>
    where
        L: AddAssign<R>,
        R: Clone,
    {
        self.elementwise_operation_assign(rhs, |left, right| *left += right.clone())
    }
}

macro_rules! impl_helper {
    ($(($t:ty, $s:ty, $u:ty))*) => {
        $(
            impl Add<$s> for Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: $s) -> Self::Output {
                    self.scalar_operation_consume_self(&rhs, |element, scalar| element + *scalar)
                }
            }

            impl Add<$s> for &Matrix<$t> {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: $s) -> Self::Output {
                    self.scalar_operation(&rhs, |element, scalar| *element + *scalar)
                }
            }

            impl Add<Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation_consume_self(&self, |element, scalar| *scalar + element)
                }
            }

            impl Add<&Matrix<$t>> for $s {
                type Output = Matrix<$u>;

                #[inline]
                fn add(self, rhs: &Matrix<$t>) -> Self::Output {
                    rhs.scalar_operation(&self, |element, scalar| *scalar + *element)
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
    #[allow(clippy::op_ref)]
    fn test_primitive_scalar_add() {
        let matrix = matrix![[1, 2, 3], [4, 5, 6]];
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

        let matrix = matrix![[&1, &2, &3], [&4, &5, &6]];

        assert_eq!(matrix.clone() + scalar, expected);
        assert_eq!(matrix.clone() + &scalar, expected);
        assert_eq!(&matrix + scalar, expected);
        assert_eq!(&matrix + &scalar, expected);
        assert_eq!(scalar + matrix.clone(), expected);
        assert_eq!(&scalar + matrix.clone(), expected);
        assert_eq!(scalar + &matrix, expected);
        assert_eq!(&scalar + &matrix, expected);
    }
}
